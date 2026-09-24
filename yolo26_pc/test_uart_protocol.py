"""Linux-native C parser integration test; does not require pyserial or a board."""
import argparse
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
import unittest

from uart_backend import UARTBackend, HEADER, BEGIN, DATA, RUN, ACK, GET, RESULT, STOP, PING, PONG, DIAG_GET, DIAG, packet


class PipeTransport:
    def __init__(self, binary):
        self.process=subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        self.raw=bytearray()
        self.ready=bytearray()
        self.drop=False
        self.drop_kind=None
        self.corrupt=False
        self.stale=False
    def write(self, data):
        part=data[:127]
        self.process.stdin.write(part)
        self.process.stdin.flush()
        return len(part)
    def read(self, n):
        if not self.ready:
            if select.select([self.process.stdout],[],[],.02)[0]:
                self.raw.extend(os.read(self.process.stdout.fileno(),4096))
            if len(self.raw)>=32:
                h=HEADER.unpack_from(self.raw)
                count=32+h[6]
                if len(self.raw)>=count:
                    response=bytes(self.raw[:count]); del self.raw[:count]
                    if self.drop and (self.drop_kind is None or h[2]==self.drop_kind):
                        self.drop=False
                    elif self.corrupt:
                        self.corrupt=False
                        self.ready.extend(response[:-1]+bytes([response[-1]^1]))
                    else:
                        if self.stale:
                            self.ready.extend(packet(PONG,h[3]-1,h[4],h[5],b'late'))
                            self.stale=False
                        self.ready.extend(response)
        answer=bytes(self.ready[:min(n,7)])
        del self.ready[:len(answer)]
        return answer
    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=3)
        self.process.stdout.close()


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.link=PipeTransport(BINARY)
        self.client=UARTBackend('test',transport=self.link)
    def tearDown(self):
        self.client.close()
    def test_fragmentation_crc_and_stale_reply(self):
        self.link.stale=True
        self.client.ping(bytes(range(256))*4)
        self.link.corrupt=True
        self.assertEqual(self.client.request(PING,PONG,b'crc-check',timeout=.15),b'crc-check')
    def test_frame_and_offset_errors(self):
        self.client.request(BEGIN,ACK)
        with self.assertRaises(RuntimeError):
            self.client.request(DATA,ACK,b'x',offset=1)
        self.client.frame+=1
        with self.assertRaises(RuntimeError):
            self.client.request(DATA,ACK,b'x',offset=0)
    def test_full_frame_retry_does_not_repeat_inference(self):
        self.client.request(BEGIN,ACK)
        total=3*416*416*4
        for offset in range(0,total,1024):
            self.client.request(DATA,ACK,bytes(min(1024,total-offset)),offset)
        self.link.drop=True
        self.client.request(RUN,ACK,timeout=.15)
        result=bytearray()
        while len(result)<7200:
            result.extend(self.client.request(GET,RESULT,offset=len(result)))
        self.assertEqual(len(result),7200)
        self.assertEqual(struct.unpack_from('<f',result)[0],1.0)
        self.client.request(STOP,ACK)
    def test_cancel_and_malformed_header_resync(self):
        broken=bytearray(packet(PING,123,payload=b'')); broken[-1]^=1
        cursor=0
        while cursor<len(broken): cursor+=self.link.write(broken[cursor:])
        self.client.ping(b'recovered')
        self.client.stop.set()
        with self.assertRaises(InterruptedError): self.client.ping(b'cancelled')

    def test_diagnostic_snapshot_and_crc_counter(self):
        events=[]
        self.client.diagnostic=events.append
        self.client.configure_diagnostics(trace=True,check_finite=True)
        self.assertEqual(self.client.last_diagnostic['options'],7)
        broken=bytearray(packet(PING,123)); broken[-1]^=1
        self.link.write(broken)
        self.client.request(DIAG_GET,DIAG)
        self.assertEqual(self.client.last_diagnostic['crc_errors'],1)
        self.assertTrue(events)

    def test_model_error_keeps_node_and_signed_detail(self):
        events=[]
        self.client.diagnostic=events.append
        self.client.configure_diagnostics()
        self.client.request(BEGIN,ACK)
        total=3*416*416*4
        for offset in range(0,total,1024):
            block=bytearray(min(1024,total-offset))
            if not offset: block[0]=255
            self.client.request(DATA,ACK,block,offset)
        with self.assertRaisesRegex(RuntimeError,'error 3'):
            self.client.request(RUN,ACK)
        d=self.client.last_diagnostic
        self.assertEqual((d['node'],d['error'],d['detail']),(17,0x31,-7))
        self.assertEqual(d['stage_name'],'failed')
        self.assertEqual(d['received'],total)
        self.assertTrue(any(e['diagnostic']['stage_name']=='run' for e in events))

    def test_trace_shapes_and_retry_do_not_repeat_model(self):
        events=[]
        self.client.diagnostic=events.append
        self.client.configure_diagnostics(trace=True)
        self.client.request(BEGIN,ACK)
        total=3*416*416*4
        for offset in range(0,total,1024):
            self.client.request(DATA,ACK,bytes(min(1024,total-offset)),offset)
        self.link.drop=True; self.link.drop_kind=ACK
        self.client.request(RUN,ACK,timeout=.5)
        result=self.client.request(GET,RESULT)
        self.assertEqual(struct.unpack_from('<f',result)[0],1.0)
        starts=[e['diagnostic'] for e in events if e.get('type')=='board' and e['diagnostic']['phase_name']=='node_begin']
        self.assertEqual(len(starts),1)
        self.assertEqual(starts[0]['tensors'][0]['shape'],[1,3,416,416])
        self.assertEqual(starts[0]['tensors'][1]['shape'],[1,300,6])
        self.assertGreaterEqual(self.client.last_diagnostic['retries'],1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('binary')
    args,remaining=parser.parse_known_args()
    BINARY=str(Path(args.binary).resolve())
    unittest.main(argv=[sys.argv[0]]+remaining)
