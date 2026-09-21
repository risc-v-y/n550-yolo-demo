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

from uart_backend import UARTBackend, HEADER, BEGIN, DATA, RUN, ACK, GET, RESULT, STOP, PING, PONG, packet


class PipeTransport:
    def __init__(self, binary):
        self.process=subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        self.raw=bytearray()
        self.ready=bytearray()
        self.drop=False
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
                    if self.drop:
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


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('binary')
    args,remaining=parser.parse_known_args()
    BINARY=str(Path(args.binary).resolve())
    unittest.main(argv=[sys.argv[0]]+remaining)
