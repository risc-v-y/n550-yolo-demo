"""Single-owner PCIe DDR transport through the platform's pbcopy/pbload commands."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zlib

BASE=0x80000000
MAILBOX=0x81f00000
MAGIC=0x50363259
INPUT_BYTES=3*416*416*4
OUTPUT_BYTES=7200
BOOT,READY,BUSY,DONE,ERROR=range(5)
INFO=struct.Struct('<4I3Q6I')
STATUS=struct.Struct('<4I2Q3Ii4I')
STATUS_KEYS='magic version state error session frame sequence output_bytes output_crc detail generation reserved reserved2 crc'.split()


def checked(data):
    if len(data)!=64 or zlib.crc32(data[:60])!=struct.unpack_from('<I',data,60)[0]:
        raise ValueError('PCIe snapshot length/CRC mismatch')
    return data


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
        self.temp=tempfile.TemporaryDirectory(prefix='yolo26-pcie-')
        self.path=Path(self.temp.name)/'transfer.bin'

    def _run(self,command,argument,deadline):
        if self.stop.is_set():
            raise InterruptedError('PCIe operation cancelled')
        limit=min(deadline,time.monotonic()+self.timeout)
        if time.monotonic()>=limit:
            raise TimeoutError('PCIe command deadline exceeded')
        started=time.monotonic()
        process=subprocess.Popen([command,'-d',argument],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
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
        except BaseException:
            process.kill()
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass  # A blocked kernel driver may require platform intervention.
            raise
        text=output.decode('utf-8',errors='replace')
        self.diagnostic({'type':'pcie_tool','message':f'{Path(command).name}: exit {process.returncode}',
                         'argument':argument,'elapsed_seconds':time.monotonic()-started,
                         'returncode':process.returncode,'output':text[-8192:]})
        if process.returncode:
            raise RuntimeError(f'{Path(command).name} failed ({process.returncode}): {text[-2000:]}')

    def write(self,address,data,deadline):
        self.path.write_bytes(data)
        self._run(self.commands[0],f'{self.path}:0x{address-BASE:x}',deadline)

    def read(self,address,size,deadline):
        self.path.unlink(missing_ok=True)  # Never accept a stale output file.
        self._run(self.commands[1],f'{self.path}:0x{address-BASE:x}:0x{size:x}',deadline)
        data=self.path.read_bytes()
        if len(data)!=size:
            raise ValueError(f'PCIe read returned {len(data)} bytes, expected {size}')
        return data

    def close(self):
        self.temp.cleanup()
        self.lock.close()


class PCIeBackend:
    def __init__(self, pbcopy='pbcopy', pbload='pbload', stop=None, infer_timeout=900,
                 progress=None, diagnostic=None, transport=None, tool_timeout=30,
                 poll_interval=.1, check_finite=False):
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
        self.options=4 if check_finite else 0
        self.failed=False
        self.last_timing={}
        self.last_diagnostic=None
        self.last_status=None
        try:
            deadline=time.monotonic()+infer_timeout
            values=INFO.unpack(checked(self.link.read(MAILBOX,64,deadline)))
            if values[:3]!=(MAGIC,1,256) or values[7:11]!=(INPUT_BYTES,OUTPUT_BYTES,816,334):
                raise ValueError('Incompatible PCIe firmware; load demo-pcie.elf')
            self.input_address,self.output_address,self.diag_address=values[4:7]
            for address,size in ((self.input_address,INPUT_BYTES),(self.output_address,OUTPUT_BYTES),(self.diag_address,816)):
                if address<BASE or address%64 or address+size>MAILBOX:
                    raise ValueError('Firmware reports a buffer outside the reserved model DDR')
            self.info={'input_address':self.input_address,'output_address':self.output_address,
                       'diagnostic_address':self.diag_address,'input_bytes':INPUT_BYTES,
                       'output_bytes':OUTPUT_BYTES,'nodes':334,'mailbox':MAILBOX,'version':1}
            while True:
                state=self._status(deadline)
                doorbell=struct.unpack_from('<I',self.link.read(MAILBOX+128,64,deadline))[0]
                if state['state'] not in (BOOT,BUSY) and doorbell==state['sequence']:
                    if state['state']==ERROR and state['error'] not in range(0x40,0x44):
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
            if state['magic']!=MAGIC or state['version']!=1 or state['state'] not in range(5):
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
        try:
            self.last_diagnostic=decode_snapshot(self.link.read(self.diag_address,816,deadline))
            self.diagnostic({'type':'board','diagnostic':self.last_diagnostic})
        except Exception as exc:
            self.diagnostic({'type':'pcie_diagnostic_error','message':str(exc)})

    def _wait(self,target,deadline):
        while True:
            state=self._status(deadline)
            if state['state']==ERROR:
                self._snapshot(deadline)
                raise RuntimeError(f'Board PCIe error: {state}; diagnostic={self.last_diagnostic}')
            if state['sequence']==self.sequence:
                if (state['session'],state['frame'])!=(self.session,self.frame):
                    raise ValueError('PCIe result session/frame mismatch')
                if state['state']==target:
                    return state
            self._pause(deadline)

    def infer(self,tensor):
        import numpy as np
        if self.failed:
            raise RuntimeError('PCIe session failed; reconnect before sending another frame')
        if tensor.shape!=(3,416,416) or tensor.dtype!=np.float32 or not np.isfinite(tensor).all():
            raise ValueError('Expected finite RGB CHW FP32 416x416 input')
        started=time.monotonic()
        deadline=started+self.infer_timeout
        try:
            previous=self._status(deadline)
            if previous['session']!=self.session or previous['sequence']!=self.sequence or previous['state'] not in (READY,DONE):
                raise RuntimeError('PCIe ownership/state changed; refusing to overwrite DDR')
            self.frame+=1
            data=tensor.astype('<f4',copy=False).tobytes()
            self.progress(f'PCIe uploading frame {self.frame}')
            self.link.write(self.input_address,data,deadline)
            self._submit(2,len(data),zlib.crc32(data),deadline)
            uploaded=time.monotonic()
            self.progress(f'Board processing frame {self.frame}')
            state=self._wait(DONE,deadline)
            inferred=time.monotonic()
            if state['output_bytes']!=OUTPUT_BYTES:
                raise ValueError('PCIe result length mismatch')
            result=self.link.read(self.output_address,OUTPUT_BYTES,deadline)
            if len(result)!=OUTPUT_BYTES or zlib.crc32(result)!=state['output_crc']:
                raise ValueError('PCIe result CRC/length mismatch')
            confirmed=self._status(deadline)
            if confirmed!=state:
                raise ValueError('PCIe status changed during result download')
            candidates=np.frombuffer(result,dtype='<f4').reshape(300,6).copy()
            classes=candidates[:,5]
            if not np.isfinite(candidates).all() or np.any(classes!=np.rint(classes)) or np.any((classes<0)|(classes>=80)):
                raise ValueError('Invalid PCIe result values/classes')
            self.last_timing={'upload_seconds':uploaded-started,'run_roundtrip_seconds':inferred-uploaded,
                              'download_seconds':time.monotonic()-inferred,'pcie_frame':self.frame}
            self._snapshot(deadline)
            return candidates
        except BaseException:
            self.failed=True
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pbcopy',default='pbcopy')
    parser.add_argument('--pbload',default='pbload')
    parser.add_argument('--infer-timeout',type=float,default=900)
    parser.add_argument('--tool-timeout',type=float,default=30)
    parser.add_argument('--poll-interval',type=float,default=.1)
    parser.add_argument('--check-intermediates',action='store_true')
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
                            progress=print,diagnostic=log.emit,check_finite=args.check_intermediates)
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
