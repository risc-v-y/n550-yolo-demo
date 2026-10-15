"""Native arithmetic edge cases and protocol checks; no physical board."""
from fractions import Fraction
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from full_replay import Replay,ROOT
from tile_probe import fp32_rne

BINARY=Path(os.environ.get('YOLO_AUDITOR',str(ROOT/'yolo26_riscv/build/host-verify/host-verify')))


@unittest.skipUnless(sys.platform=='linux' and BINARY.exists(),'Build the Linux static auditor first')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/'reference/nodes/manifest.json').read_text())

    def convolution(self,terms,bias=0.0,mode=16,weights=None,k=64):
        # A real 1x1 convolution with K=64 exercises multiple AMU instructions.
        tensors=self.manifest['tensors']
        node=next(n for n in self.manifest['nodes'] if n['kind']=='CONV' and n['params'][4]==1
                  and tensors[n['inputs'][1]]['shape'][1:]==[k,1,1])
        x,w,b=node['inputs']; out=node['outputs'][0]
        area=tensors[x]['shape'][2]*tensors[x]['shape'][3]
        channels=tensors[w]['shape'][0]
        values=list(terms)+[0.0]*(k-len(terms))
        weights=weights or [1.0]*k
        data={x:b''.join(struct.pack('<f',v)*area for v in values),
              w:struct.pack('<{}f'.format(k),*weights)*channels,b:struct.pack('<f',bias)*channels}
        replay=Replay(mode,binary=BINARY)
        try: return replay.run(node,tensors,data)[out]
        finally: replay.close()

    def test_boundaries_cancellation_and_bias(self):
        cases=[([1.0,2**-24]+[0.0]*14+[2**-24],-1.0),
               ([1.0]+[0.0]*15+[2**-24,2**-48],0.0),
               ([65504.0,-65504.0]*31+[1.0,2**-24],-1.0),
               ([2**-24]*64,2**-24)]
        # The input is packed to half before multiplication, including tails.
        for terms,bias in cases:
            packed=[struct.unpack('<e',struct.pack('<e',v))[0] for v in terms]
            acc=0.0
            for i in range(0,64,16):
                acc=fp32_rne(sum(map(Fraction,packed[i:i+16]),Fraction(acc)))
            expected=struct.pack('<f',fp32_rne(Fraction(acc)+Fraction(bias)))
            raw=self.convolution(terms,bias)
            self.assertEqual(raw,expected*(len(raw)//4))

    def test_once32_and_split16_are_distinct(self):
        terms=[1.0,2**-24]+[0.0]*14+[2**-24]
        self.assertNotEqual(self.convolution(terms,mode=16),self.convolution(terms,mode=32))

    def test_tail_combines_with_accumulator_before_rounding(self):
        terms=[1.0]+[0.0]*15+[2**-24,2**-24]
        weights=[1.0]*64; weights[17]=2**-24
        raw=self.convolution(terms,weights=weights)
        self.assertEqual(raw,struct.pack('<I',0x3f800001)*(len(raw)//4))

    def test_long_k_spans_multiple_32_item_instructions(self):
        terms=[65504.0,-65504.0]*127+[1.0,2**-24]
        raw=self.convolution(terms,-1.0,k=256)
        self.assertEqual(raw,bytes(len(raw)))

    def test_first_layer_five_saved_screenshot_words(self):
        from tile_probe import assets
        manifest,inputs,_,_=assets(); replay=Replay(16,binary=BINARY)
        try: raw=replay.run(manifest['nodes'][0],manifest['tensors'],inputs)[3]
        finally: replay.close()
        expected={28:0x3fa9d738,29:0x3f2baaf4,30:0x3fdca53e,15357:0x38570000,624567:0x4280f2e9}
        for element,word in expected.items(): self.assertEqual(struct.unpack_from('<I',raw,element*4)[0],word)

    def test_bad_input_length_closes_session(self):
        node=self.manifest['nodes'][0]; replay=Replay(binary=BINARY)
        with self.assertRaises(ValueError): replay.run(node,self.manifest['tensors'],{0:b'x'})
        self.assertTrue(replay.failed)
        self.assertIsNotNone(replay.process.poll())

    def test_packaged_binary_source_binding_and_selftest(self):
        replay=Replay()
        replay.close()

    def test_topk_ties_and_pruned_score_output(self):
        for node_id in (318,323):
            node=self.manifest['nodes'][node_id]; tensors=self.manifest['tensors']
            width=tensors[node['inputs'][0]]['count']; values=[1.0]*width; values[-1]=2.0
            replay=Replay(binary=BINARY)
            try: result=replay.run(node,tensors,{node['inputs'][0]:struct.pack('<{}f'.format(width),*values)})
            finally: replay.close()
            self.assertEqual(list(struct.unpack('<300q',result[node['outputs'][1]])),[width-1]+list(range(299)))
            if node['outputs'][0]>=0:
                self.assertEqual(result[node['outputs'][0]],struct.pack('<300f',2,*([1]*299)))

    def test_gather_int64_preserves_indices_and_large_values(self):
        node=self.manifest['nodes'][329]; tensors=self.manifest['tensors']; ids=node['inputs']
        values=[2**60+i for i in range(300)]; selected=list(reversed(range(300)))
        replay=Replay(binary=BINARY)
        try:
            result=replay.run(node,tensors,{ids[0]:struct.pack('<300q',*values),ids[1]:struct.pack('<300q',*selected)})
        finally: replay.close()
        self.assertEqual(struct.unpack('<300q',result[node['outputs'][0]]),tuple(reversed(values)))


if __name__=='__main__': unittest.main()
