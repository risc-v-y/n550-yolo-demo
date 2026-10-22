import struct
import sys
from pathlib import Path
import unittest
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'yolo26_pc'))
from pcie_backend import INFO, STATUS, STATUS_KEYS, MAGIC, MAILBOX, INPUT_BYTES
from pcie_stream import StreamBackend, RequestReadbackError


class Memory:
    def __init__(self,version=2):
        self.memory={}; self.writes=[]; self.corrupt_request=False; self.corrupt_output=False
        self.closed=False; self.version=version; self.generation=0
        info=[MAGIC,version,256,3,0xf0200000,0xf0500000,0x80020000,INPUT_BYTES,7200,832,334,0x80010000,0]
        raw=INFO.pack(*info); self.memory[MAILBOX]=raw[:60]+struct.pack('<I',zlib.crc32(raw[:60]))
        caps=[MAGIC,2,128,0xf0600000,0x80040000,0x80050000,0x80030000,519168,519232,7336,128,65536,64,30,0,0]
        raw=struct.pack('<16Q',*caps); self.memory[0x80010000]=raw[:120]+struct.pack('<Q',zlib.crc32(raw[:120]))
        self.memory[MAILBOX+128]=bytes(64)
        self.status(1,0,0,0)

    def status(self,state,session,frame,sequence):
        self.generation+=1
        raw_output=bytes(7200)
        values=[MAGIC,self.version,state,0,session,frame,sequence,7200 if state==3 else 0,
                zlib.crc32(raw_output) if state==3 else 0,0,self.generation,0,0,0]
        raw=STATUS.pack(*values); raw=raw[:60]+struct.pack('<I',zlib.crc32(raw[:60]))
        self.memory[MAILBOX+192]=raw
        if state==3:
            packet=raw+struct.pack('<8Q',100,200,10,20,0,0,0,0)+raw_output
            self.memory[0x80040000]=packet+struct.pack('<Q',zlib.crc32(packet))

    def write(self,address,data,deadline):
        self.writes.append((address,len(data))); self.memory[address]=data
        if address==MAILBOX+128:
            seq,slot=struct.unpack_from('<2I',data)
            request=self.memory[0xf0600000+(slot-1)*519232][:64] if slot else self.memory[MAILBOX+64]
            session,frame,command=struct.unpack_from('<QQI',request)
            self.status(3 if command==5 else 1,session,frame,seq)

    def read(self,address,size,deadline):
        if self.corrupt_request and address>=0xf0600000 and size==64: return bytes(64)
        raw=self.memory[address][:size]
        if self.corrupt_output and address==0x80040000: return raw[:-1]+bytes([raw[-1]^1])
        return raw

    def close(self): self.closed=True


class StreamTests(unittest.TestCase):
    def test_pipeline_slot_order_and_single_packet(self):
        memory=Memory(); board=StreamBackend(transport=memory)
        board.configure({'m':128,'n':64})
        board.upload(bytes(519168),1); board.begin()
        board.upload(bytes([1])*519168,2)
        with self.assertRaises(RuntimeError): board.begin()
        self.assertEqual(len(board.finish()),7200)
        self.assertIsNone(board.last_timing['profile_read_seconds'])
        board.begin(); self.assertEqual(len(board.finish()),7200)
        uploads=[p for p in memory.writes if p[1]>64]
        self.assertEqual(uploads,[(0xf0600000,519232),(0xf0600000+519232,519232)])
        board.close(); self.assertTrue(memory.closed)

    def test_wrong_firmware_rejected_before_any_write(self):
        memory=Memory(1)
        with self.assertRaises(ValueError): StreamBackend(transport=memory)
        self.assertEqual(memory.writes,[])
        self.assertTrue(memory.closed)

    def test_pack_capability_is_checked_before_config_write_and_commit_is_timed(self):
        memory=Memory(); board=StreamBackend(transport=memory)
        before=len(memory.writes)
        with self.assertRaises(ValueError): board.configure({'pack':29})
        self.assertEqual(len(memory.writes),before)
        board.stream['pack_mask']=31
        board.configure({'m':128,'n':32,'pack':29,'compute':0,'graph':11})
        board.upload(bytes(519168),1); board.begin(); board.finish()
        self.assertGreaterEqual(board.last_timing['commit_seconds'],0)

    def test_readback_failure_never_commits_and_cannot_retry(self):
        memory=Memory(); board=StreamBackend(transport=memory)
        count=len(memory.writes); memory.corrupt_request=True
        with self.assertRaises(ValueError): board.upload(bytes(519168),1)
        self.assertTrue(board.failed)
        self.assertEqual(len(memory.writes),count+1)
        with self.assertRaises(RuntimeError): board.upload(bytes(519168),1)

    def test_corrupt_result_and_capacity(self):
        memory=Memory(); board=StreamBackend(transport=memory)
        with self.assertRaises(ValueError): board.configure({'m':256})
        board.upload(bytes(519168),1); board.begin(); memory.corrupt_output=True
        with self.assertRaises(ValueError): board.finish()
        self.assertTrue(board.failed)

    def test_pending_frame_mismatch_preserves_both_headers_and_active_frame(self):
        memory=Memory(); events=[]; board=StreamBackend(transport=memory,diagnostic=events.append)
        board.upload(bytes(519168),1); board.begin()
        count=len(memory.writes); memory.corrupt_request=True
        with self.assertRaises(RequestReadbackError) as caught:
            board.upload(bytes([1])*519168,2)
        detail=caught.exception.details
        self.assertEqual((detail['frame_id'],detail['active_frame_id'],detail['slot']),(1,0,2))
        self.assertEqual(detail['cpu_address'],0xf0600000+519232)
        self.assertEqual(len(bytes.fromhex(detail['expected_hex'])),64)
        self.assertEqual(detail['actual_hex'],'00'*64)
        self.assertFalse(detail['actual_crc_valid'])
        self.assertEqual(detail['different_offsets'],[i for i,b in enumerate(bytes.fromhex(detail['expected_hex'])) if b])
        self.assertIn(detail,events)
        self.assertTrue(board.failed)
        self.assertEqual(len(memory.writes),count+1)  # Exactly one upload, no commit/rewrite.
        with self.assertRaises(RuntimeError): board.begin()


if __name__=='__main__': unittest.main()
