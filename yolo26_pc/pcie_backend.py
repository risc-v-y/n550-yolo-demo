"""Single-owner PCIe DDR transport through the platform's pbcopy/pbload commands."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import secrets
import re
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time
import zlib

BASE=0x80000000
MAILBOX=0x81f00000
MODEL_DDR_BASE,MODEL_DDR_END=0xf0000000,0xf1000000
TOOL_FAILURE=re.compile(r'(?i)(?:\b(?:failed|failure)\b|\bfail!|\[x\]\s*ERROR|EXIT_WITH_NO_LICENSE)')
MAGIC=0x50363259
INPUT_BYTES=3*416*416*4
OUTPUT_BYTES=7200
BOOT,READY,BUSY,DONE,ERROR,NODE=range(6)
INFO=struct.Struct('<4I3Q6I')
STATUS=struct.Struct('<4I2Q3Ii4I')
STATUS_KEYS='magic version state error session frame sequence output_bytes output_crc detail generation reserved reserved2 crc'.split()


def checked(data):
    if len(data)!=64 or zlib.crc32(data[:60])!=struct.unpack_from('<I',data,60)[0]:
        raise ValueError('PCIe snapshot length/CRC mismatch')
    return data


class PCIeToolError(RuntimeError):
    def __init__(self, command, returncode, output):
        self.returncode=returncode
        self.command=str(command)
        super().__init__(f'{Path(command).name} failed ({returncode}): {output[-2000:]}')


class PBTools:
    def __init__(self, pbcopy='pbcopy', pbload='pbload', stop=None, timeout=30, diagnostic=None):
        self.stop=stop or threading.Event()
        self.timeout=timeout
        self.diagnostic=diagnostic or (lambda event: None)
        self.commands=[]
        for command in (pbcopy,pbload):
            resolved=shutil.which(command)
            if resolved is None:
                raise FileNotFoundError(f'PCIe executable not found: {command}')
            self.commands.append(resolved)
        # A second local controller must not overwrite an in-flight frame.
        import fcntl
        self.lock=open(Path(tempfile.gettempdir())/'yolo26-pcie.lock','a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError('Another host process owns the YOLO PCIe link') from None
        self.cwd=Path.cwd()
        try:
            handle,name=tempfile.mkstemp(prefix='p',suffix='.bin',dir=self.cwd)
            os.close(handle)
            self.path=Path(name)
        except BaseException:
            self.lock.close()
            raise

    def _run(self,command,argument,deadline):
        if self.stop.is_set():
            raise InterruptedError('PCIe operation cancelled')
        limit=min(deadline,time.monotonic()+self.timeout)
        if time.monotonic()>=limit:
            raise TimeoutError('PCIe command deadline exceeded')
        started=time.monotonic()
        process=subprocess.Popen([command,'-d',argument],cwd=self.cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            while True:
                if self.stop.is_set():
                    raise InterruptedError('PCIe operation cancelled; board inference may still be running')
                remaining=limit-time.monotonic()
                if remaining<=0:
                    raise TimeoutError(f'{Path(command).name} timed out')
                try:
                    output=process.communicate(timeout=min(.1,remaining))[0]
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as exc:
            exc.pcie_operation='read' if command==self.commands[1] else 'write'
            process.kill()
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass  # A blocked kernel driver may require platform intervention.
            raise
        text=output.decode('utf-8',errors='replace')
        failed=bool(process.returncode or TOOL_FAILURE.search(text))
        self.diagnostic({'type':'pcie_tool','message':f'{Path(command).name}: exit {process.returncode}',
                         'argument':argument,'elapsed_seconds':time.monotonic()-started,
                         'returncode':process.returncode,'output':text if failed else text[-8192:]})
        if failed:
            exc=PCIeToolError(command,process.returncode,text)
            exc.pcie_operation='read' if command==self.commands[1] else 'write'
            raise exc

    def write(self,address,data,deadline):
        self._range(address,len(data))
        self.path.write_bytes(data)
        self._run(self.commands[0],f'{self.path.name}:0x{address-BASE:x}',deadline)

    def read(self,address,size,deadline):
        self._range(address,size)
        # Retry only crashed reads, at most twice, within the original deadline.
        # A crashed process may have left a full-sized file; never accept it.
        for attempt in range(1,4):
            self.path.unlink(missing_ok=True)
            try:
                self._run(self.commands[1],f'{self.path.name}:0x{address-BASE:x}:0x{size:x}',deadline)
                break
            except PCIeToolError as exc:
                self.path.unlink(missing_ok=True)
                if exc.returncode!=-signal.SIGSEGV or attempt==3:
                    raise
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise
                self.diagnostic({'type':'pcie_read_retry','address':address,'bytes':size,
                                 'attempt':attempt,'next_attempt':attempt+1,
                                 'message':f'pbload crashed; retry {attempt}/2, address=0x{address:x}, bytes={size}'})
                if self.stop.wait(min(.2,remaining)):
                    raise InterruptedError('PCIe read retry cancelled') from exc
        data=self.path.read_bytes()
        if len(data)!=size:
            raise ValueError(f'PCIe read returned {len(data)} bytes, expected {size}')
        return data

    def close(self):
        self.path.unlink(missing_ok=True)
        self.lock.close()

    @staticmethod
    def _range(address,size):
        if size<=0 or not (BASE<=address and address+size<=0x82000000 or
                          MODEL_DDR_BASE<=address and address+size<=MODEL_DDR_END):
            raise ValueError('PCIe access outside reserved DDR')


class PCIeBackend:
    output_bytes=OUTPUT_BYTES
    node_count=334
    required_capability=0
    firmware_name='demo-pcie.bin'

    def validate_result(self,result):
        for candidate in struct.iter_unpack('<6f',result):
            category=candidate[5]
            if not all(map(math.isfinite,candidate)) or category!=round(category) or not 0<=category<80:
                raise ValueError('Invalid PCIe result values/classes')

    def __init__(self, pbcopy='pbcopy', pbload='pbload', stop=None, infer_timeout=900,
                 progress=None, diagnostic=None, transport=None, tool_timeout=30,
                 poll_interval=.1, check_finite=False, trace=False, capture=False):
        if infer_timeout<=0 or tool_timeout<=0 or poll_interval<=0:
            raise ValueError('PCIe timeouts and polling interval must be positive')
        self.stop=stop or threading.Event()
        self.infer_timeout=infer_timeout
        self.poll_interval=poll_interval
        self.progress=progress or (lambda text: None)
        self.diagnostic=diagnostic or (lambda event: None)
        self.link=transport or PBTools(pbcopy,pbload,self.stop,tool_timeout,self.diagnostic)
        self.session=secrets.randbits(64) or 1
        self.frame=0
        self.options=(4 if check_finite else 0) | (2 if trace else 0) | (8 if capture else 0)
        self.failed=False
        self.last_timing={}
        self.last_diagnostic=None
        self.last_status=None
        self.profile_descriptor=None
        try:
            deadline=time.monotonic()+infer_timeout
            values=INFO.unpack(checked(self.link.read(MAILBOX,64,deadline)))
            if ((values[0],values[2])!=(MAGIC,256) or values[1] not in (1,2) or
                    (getattr(self,'required_version',None) is not None and values[1]!=self.required_version) or
                    (values[7],values[8],values[10])!=(INPUT_BYTES,self.output_bytes,self.node_count)
                    or values[9] not in (816,832)
                    or (values[3]&self.required_capability)!=self.required_capability):
                raise ValueError('Incompatible PCIe firmware; load firmware/'+self.firmware_name)
            self.input_address,self.output_address,self.diag_address=values[4:7]
            self.diag_bytes=values[9]
            self.capture_address=values[11] if values[3]&1 else 0
            self.protocol_version=values[1]
            self.stream=None
            if self.protocol_version==2:
                if not values[3]&2 or values[11]%64 or not BASE<=values[11]<=MAILBOX-128:
                    raise ValueError('Invalid version-2 capability address')
                raw=self.link.read(values[11],128,deadline)
                fields=struct.unpack('<16Q',raw)
                if fields[:3]!=(MAGIC,2,128) or fields[-1]!=zlib.crc32(raw[:120]):
                    raise ValueError('Invalid version-2 capability CRC/schema')
                self.stream=dict(zip('magic version bytes slots result profile snapshot input_bytes slot_bytes result_bytes max_rows accumulator_bytes vector_bytes option_mask reserved crc'.split(),fields))
                self.capture_address=self.stream['snapshot']
            if capture and (not self.capture_address or self.capture_address%64 or
                            not BASE<=self.capture_address<=MAILBOX-512):
                raise ValueError('Firmware lacks node capture; load the updated demo-pcie.bin')
            for address,size in ((self.input_address,INPUT_BYTES),(self.output_address,self.output_bytes)):
                if address%64 or not (BASE<=address and address+size<=MAILBOX or
                                     MODEL_DDR_BASE<=address and address+size<=MODEL_DDR_END):
                    raise ValueError('Firmware reports a buffer outside the reserved model DDR')
            if self.diag_address<BASE or self.diag_address%64 or self.diag_address+self.diag_bytes>MAILBOX:
                raise ValueError('Firmware reports diagnostics outside reserved normal DDR')
            self.info={'input_address':self.input_address,'output_address':self.output_address,
                       'diagnostic_address':self.diag_address,'input_bytes':INPUT_BYTES,
                       'output_bytes':self.output_bytes,'nodes':self.node_count,'mailbox':MAILBOX,'version':self.protocol_version,
                       'capture_address':self.capture_address}
            while True:
                state=self._status(deadline)
                if state['state']==NODE:
                    raise RuntimeError('Board capture is paused; run node_capture.py --recover-paused')
                doorbell=struct.unpack_from('<I',self.link.read(MAILBOX+128,64,deadline))[0]
                if state['state'] not in (BOOT,BUSY) and doorbell==state['sequence']:
                    if state['state']==ERROR and state['error'] not in range(0x40,0x45):
                        self._snapshot(deadline)
                        raise RuntimeError(f'Board startup/trap failure: {state}')
                    break
                self._pause(deadline)
            self.sequence=state['sequence']
            self._submit(1,0,0,deadline)
            self._wait(READY,deadline)
        except BaseException:
            self.link.close()
            raise

    def close(self):
        self.link.close()

    def _check(self,deadline):
        if self.stop.is_set():
            raise InterruptedError('PCIe operation cancelled; board may still be processing')
        if time.monotonic()>=deadline:
            raise TimeoutError(f'PCIe timed out; last status={self.last_status}; do not overwrite an active input')

    def _pause(self,deadline):
        self._check(deadline)
        self.stop.wait(min(self.poll_interval,max(0,deadline-time.monotonic())))
        self._check(deadline)

    def _status(self,deadline):
        while True:
            self._check(deadline)
            blob=self.link.read(MAILBOX+192,64,deadline)
            try:
                state=dict(zip(STATUS_KEYS,STATUS.unpack(checked(blob))))
            except ValueError:
                self._pause(deadline)  # May have read during a board publication.
                continue
            if state['magic']!=MAGIC or state['version']!=getattr(self,'protocol_version',1) or state['state'] not in range(6):
                raise ValueError('Invalid PCIe status; firmware may have restarted')
            if state!=self.last_status:
                self.diagnostic({'type':'pcie_status','message':f"PCIe state={state['state']} frame={state['frame']}",
                                 'status':state})
            self.last_status=state
            return state

    def _submit(self,command,size,crc,deadline):
        self._check(deadline)
        self.sequence=(self.sequence+1)&0xffffffff or 1
        body=struct.pack('<QQ11I',self.session,self.frame,command,self.sequence,size,crc,self.options,*([0]*6))
        body+=struct.pack('<I',zlib.crc32(body))
        self.link.write(MAILBOX+64,body,deadline)
        if self.link.read(MAILBOX+64,64,deadline)!=body:
            raise ValueError('PCIe request readback mismatch; command was not committed')
        self.link.write(MAILBOX+128,struct.pack('<16I',self.sequence,*([0]*15)),deadline)

    def _snapshot(self,deadline):
        from board_diagnostics import decode_snapshot
        self.profile_descriptor=None
        try:
            raw=self.link.read(self.diag_address,self.diag_bytes,deadline)
            self.last_diagnostic=decode_snapshot(raw[:816])
            if self.diag_bytes==832:
                self.profile_descriptor=struct.unpack_from('<QQ',raw,816)
            self.diagnostic({'type':'board','diagnostic':self.last_diagnostic})
        except Exception as exc:
            self.diagnostic({'type':'pcie_diagnostic_error','message':str(exc)})

    def _wait(self,target,deadline):
        while True:
            state=self._status(deadline)
            # OPEN may follow a cancelled/failed session. Its old ERROR can
            # remain visible until the new committed command is consumed.
            if state['state']==ERROR and (state['sequence']==self.sequence or state['error'] not in range(0x40,0x45)):
                self._snapshot(deadline)
                raise RuntimeError(f'Board PCIe error: {state}; diagnostic={self.last_diagnostic}')
            if state['sequence']==self.sequence:
                if (state['session'],state['frame'])!=(self.session,self.frame):
                    raise ValueError('PCIe result session/frame mismatch')
                if state['state']==target:
                    return state
            self._pause(deadline)

    def infer_bytes(self,data):
        if self.failed:
            raise RuntimeError('PCIe session failed; reconnect before sending another frame')
        if not isinstance(data,bytes) or len(data)!=INPUT_BYTES:
            raise ValueError(f'Expected {INPUT_BYTES} bytes of RGB CHW FP32 input')
        if not all(math.isfinite(value) for (value,) in struct.iter_unpack('<f',data)):
            raise ValueError('Input contains NaN/Inf')
        started=time.monotonic()
        deadline=started+self.infer_timeout
        try:
            previous=self._status(deadline)
            if previous['session']!=self.session or previous['sequence']!=self.sequence or previous['state'] not in (READY,DONE):
                raise RuntimeError('PCIe ownership/state changed; refusing to overwrite DDR')
            self.frame+=1
            self.progress(f'PCIe uploading frame {self.frame}')
            self.link.write(self.input_address,data,deadline)
            self._submit(2,len(data),zlib.crc32(data),deadline)
            uploaded=time.monotonic()
            self.progress(f'Board processing frame {self.frame}')
            state=self._wait(DONE,deadline)
            inferred=time.monotonic()
            if state['output_bytes']!=self.output_bytes:
                raise ValueError('PCIe result length mismatch')
            result=self.link.read(self.output_address,self.output_bytes,deadline)
            if len(result)!=self.output_bytes or zlib.crc32(result)!=state['output_crc']:
                raise ValueError('PCIe result CRC/length mismatch')
            confirmed=self._status(deadline)
            if confirmed!=state:
                raise ValueError('PCIe status changed during result download')
            self.validate_result(result)
            self.last_timing={'upload_seconds':uploaded-started,'run_roundtrip_seconds':inferred-uploaded,
                              'download_seconds':time.monotonic()-inferred,'pcie_frame':self.frame}
            self._snapshot(deadline)
            if self.diag_bytes==832:
                if self.profile_descriptor is None:
                    raise ValueError('Missing board timing descriptor')
                address,size=self.profile_descriptor
                if address or size:
                    from board_profile import decode_profile,PROFILE_LENGTHS
                    if size not in PROFILE_LENGTHS or address%64 or not BASE<=address<=MAILBOX-size:
                        raise ValueError('Invalid board timing address/length')
                    profile_start=time.monotonic()
                    self.last_timing['board_profile']=decode_profile(
                        self.link.read(address,size,deadline),self.frame,self.session)
                    if self._status(deadline)!=state:
                        raise ValueError('PCIe ownership changed during timing read')
                    self.last_timing['profile_read_seconds']=time.monotonic()-profile_start
            return result
        except BaseException:
            self.failed=True
            raise

    def infer(self,tensor):
        import numpy as np
        if tensor.shape!=(3,416,416) or tensor.dtype!=np.float32 or not np.isfinite(tensor).all():
            raise ValueError('Expected finite RGB CHW FP32 416x416 input')
        return np.frombuffer(self.infer_bytes(tensor.astype('<f4',copy=False).tobytes()),
                             dtype='<f4').reshape(300,6).copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pbcopy',default='pbcopy')
    parser.add_argument('--pbload',default='pbload')
    parser.add_argument('--infer-timeout',type=float,default=900)
    parser.add_argument('--tool-timeout',type=float,default=30)
    parser.add_argument('--poll-interval',type=float,default=.1)
    parser.add_argument('--check-intermediates',action='store_true')
    parser.add_argument('--board-trace',action='store_true',help='Print every node and tensor shape on the debug UART')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    from board_diagnostics import DiagnosticLog
    output=args.output or Path(__file__).resolve().parent/'outputs'/'board-pcie'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True,exist_ok=False)
    log=DiagnosticLog(output)
    backend=None
    report={'status':'starting','transport':'PCIe DDR'}
    try:
        backend=PCIeBackend(args.pbcopy,args.pbload,infer_timeout=args.infer_timeout,
                            tool_timeout=args.tool_timeout,poll_interval=args.poll_interval,
                            progress=print,diagnostic=log.emit,check_finite=args.check_intermediates,trace=args.board_trace)
        report['board']=backend.info
        if args.image:
            from video_demo import read_image,prepare_input,postprocess,annotate,np
            from live_demo import save_image
            image=read_image(args.image)
            tensor,transform=prepare_input(image)
            candidates=backend.infer(tensor)
            boxes=postprocess(candidates,transform,.25)
            np.save(output/'candidates.npy',candidates)
            save_image(output/'source.png',image)
            save_image(output/'annotated.png',annotate(image,boxes))
            report.update(timing=backend.last_timing,transform=transform,detections=boxes.tolist())
            if args.reference:
                blob=args.reference.read_bytes()
                if len(blob)!=7232 or blob[:8]!=b'Y26VDET\0':
                    raise ValueError('Invalid reference candidate file')
                expected=np.frombuffer(blob,dtype='<f4',offset=32).reshape(300,6)
                report['reference']={'path':str(args.reference),'byte_identical':candidates.tobytes()==blob[32:],
                    'class_sequence_match':bool(np.array_equal(candidates[:,5],expected[:,5])),
                    'max_coordinate_difference':float(np.max(np.abs(candidates[:,:4]-expected[:,:4]))),
                    'max_score_difference':float(np.max(np.abs(candidates[:,4]-expected[:,4])))}
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        log.exception(exc)
        raise
    finally:
        if backend is not None:
            report['last_diagnostic']=backend.last_diagnostic
            backend.close()
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        log.close()
        print(output,flush=True)


if __name__=='__main__':
    main()
