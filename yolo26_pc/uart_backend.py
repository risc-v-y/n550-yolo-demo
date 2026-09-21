"""N550 UART transport: framed CRC32 packets, stop-and-wait, bounded retries."""
import argparse
from pathlib import Path
import secrets
import struct
import threading
import time
import zlib

HEADER = struct.Struct('<4sHHIIIIII')
HELLO, INFO, BEGIN, DATA, RUN, ACK, GET, RESULT, ERROR, PING, PONG, STOP = range(1,13)
LIMIT = 1024


def packet(kind, seq, frame=0, offset=0, payload=b''):
    if len(payload) > LIMIT:
        raise ValueError('Packet exceeds 1024 bytes')
    prefix = struct.pack('<4sHHIIIII', b'Y26B',1,kind,seq,frame,offset,len(payload),zlib.crc32(payload))
    return prefix + struct.pack('<I',zlib.crc32(prefix)) + payload


class UARTBackend:
    def __init__(self, port, baud=115200, stop=None, infer_timeout=900, progress=None, transport=None):
        if transport is None:
            import serial
            transport = serial.Serial(port, baudrate=baud, bytesize=8, parity='N', stopbits=1,
                                      timeout=.1, write_timeout=3, xonxoff=False,
                                      rtscts=False, dsrdtr=False)
        self.link = transport
        self.stop = stop or threading.Event()
        self.infer_timeout = infer_timeout
        self.progress = progress or (lambda text: None)
        self.seq = secrets.randbelow(0x7fffffff)+1
        self.frame = 0
        self.buffer = bytearray()
        self.last_timing = {}
        try:
            data = self.request(HELLO,INFO)
            if len(data)!=24:
                raise ValueError('Invalid board INFO length')
            fields = struct.unpack('<6I',data)
            self.info = dict(zip(('input_bytes','output_bytes','nodes','actual_baud','dlf_bits','cache_line'),fields))
            if fields[:3] != (3*416*416*4,7200,334) or fields[5]!=64:
                raise ValueError(f'Incompatible board firmware: {self.info}')
        except Exception:
            self.close()
            raise

    def close(self):
        self.link.close()

    def _check(self, deadline):
        if self.stop.is_set():
            raise InterruptedError('UART operation cancelled')
        if time.monotonic() >= deadline:
            raise TimeoutError('UART packet timed out; check board state, baud and cable')

    def _fill(self, n, deadline):
        while len(self.buffer)<n:
            self._check(deadline)
            self.buffer.extend(self.link.read(n-len(self.buffer)))

    def _response(self, seq, frame, offset, deadline):
        while True:
            self._check(deadline)
            self._fill(4,deadline)
            if self.buffer[:4] != b'Y26B':
                del self.buffer[0]
                continue
            self._fill(32,deadline)
            h = HEADER.unpack_from(self.buffer)
            if h[1]!=1 or h[6]>LIMIT or zlib.crc32(self.buffer[:28])!=h[8]:
                del self.buffer[0]
                continue
            self._fill(32+h[6],deadline)
            body=bytes(self.buffer[32:32+h[6]])
            del self.buffer[:32+h[6]]
            if zlib.crc32(body)!=h[7]:
                raise ValueError('UART payload CRC mismatch')
            if (h[3],h[4],h[5]) != (seq,frame,offset):
                continue  # late reply to an earlier request
            return h[2],body

    def request(self, kind, expected, payload=b'', offset=0, timeout=3):
        self.seq = (self.seq+1)&0xffffffff
        seq=self.seq
        encoded=packet(kind,seq,self.frame,offset,payload)
        failure=None
        for attempt in range(3):
            deadline=time.monotonic()+timeout
            try:
                cursor=0
                while cursor<len(encoded):
                    self._check(deadline)
                    written=self.link.write(encoded[cursor:])
                    if not written:
                        raise TimeoutError('UART write made no progress')
                    cursor+=written
                response,body=self._response(seq,self.frame,offset,deadline)
                if response==ERROR:
                    code=struct.unpack('<I',body)[0] if len(body)==4 else 'malformed'
                    raise RuntimeError(f'Board rejected command {kind}: error {code}')
                if response!=expected:
                    raise RuntimeError(f'Unexpected response {response}, expected {expected}')
                if expected==ACK and body:
                    raise RuntimeError('Unexpected ACK payload')
                return body
            except (TimeoutError,ValueError) as exc:
                failure=exc
                self.buffer.clear()
                self.progress(f'UART retry {attempt+1}/3: {exc}')
        raise TimeoutError(f'UART failed after 3 attempts: {failure}')

    def ping(self, data):
        reply=self.request(PING,PONG,data)
        if reply!=data:
            raise ValueError('UART loopback differs')

    def infer(self, tensor):
        import numpy as np
        if tensor.shape!=(3,416,416) or tensor.dtype!=np.float32 or not np.isfinite(tensor).all():
            raise ValueError('Expected finite RGB CHW FP32 416x416 input')
        self.request(BEGIN,ACK)
        data=tensor.astype('<f4',copy=False).tobytes()
        started=time.monotonic()
        previous_percent=-1
        for offset in range(0,len(data),LIMIT):
            self.request(DATA,ACK,data[offset:offset+LIMIT],offset)
            percent=(offset+min(LIMIT,len(data)-offset))*100//len(data)
            if percent!=previous_percent:
                self.progress(f'UART frame {self.frame}: upload {percent}%')
                previous_percent=percent
        uploaded=time.monotonic()
        self.progress(f'Board processing frame {self.frame}')
        self.request(RUN,ACK,timeout=self.infer_timeout)
        inferred=time.monotonic()
        result=bytearray()
        while len(result)<7200:
            block=self.request(GET,RESULT,offset=len(result))
            if len(block)!=min(LIMIT,7200-len(result)):
                raise ValueError('Invalid result block length')
            result.extend(block)
        candidates=np.frombuffer(result,dtype='<f4').reshape(300,6).copy()
        classes=candidates[:,5]
        if (not np.isfinite(candidates).all() or np.any(classes!=np.rint(classes)) or
                np.any((classes<0)|(classes>=80))):
            raise ValueError('Invalid board result values/classes')
        self.last_timing={'upload_seconds':uploaded-started,'run_roundtrip_seconds':inferred-uploaded,
                          'download_seconds':time.monotonic()-inferred}
        self.frame+=1
        return candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--baud',type=int,default=115200)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--reference',type=Path,help='Historical Y26VDET candidate .bin for the SAME source image')
    parser.add_argument('--output',type=Path,default=Path('outputs/board-image'))
    parser.add_argument('--ping',type=int,default=1,help='Number of 1024-byte loopback checks')
    args=parser.parse_args()
    backend=UARTBackend(args.port,args.baud,progress=print)
    try:
        print(backend.info)
        for i in range(args.ping):
            backend.ping(bytes((j+i)%256 for j in range(LIMIT)))
        print(f'{args.ping} UART ping checks passed')
        if args.image:
            from video_demo import read_image, prepare_input, postprocess, annotate, np, write_json
            from live_demo import save_image
            args.output.mkdir(parents=True,exist_ok=False)
            image=read_image(args.image)
            tensor,transform=prepare_input(image)
            candidates=backend.infer(tensor)
            boxes=postprocess(candidates,transform,.25)
            np.save(args.output/'candidates.npy',candidates)
            save_image(args.output/'annotated.png',annotate(image,boxes))
            report={'board':backend.info,'timing':backend.last_timing,'detections':boxes.tolist()}
            if args.reference:
                blob=args.reference.read_bytes()
                if len(blob)!=7232 or blob[:8]!=b'Y26VDET\0':
                    raise ValueError('Invalid reference candidate file')
                expected=np.frombuffer(blob,dtype='<f4',offset=32).reshape(300,6)
                report['reference']={'path':str(args.reference),
                                     'byte_identical':bool(np.array_equal(candidates,expected)),
                                     'class_sequence_match':bool(np.array_equal(candidates[:,5],expected[:,5])),
                                     'max_coordinate_difference':float(np.max(np.abs(candidates[:,:4]-expected[:,:4]))),
                                     'max_score_difference':float(np.max(np.abs(candidates[:,4]-expected[:,4])))}
            write_json(args.output/'report.json',report)
    finally:
        backend.close()


if __name__=='__main__':
    main()
