"""Full-value metrics, reference framing and compact return bundle checks."""
import gzip
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib
from node_compare import ReferenceArchive, compare_tensor, replay_convolution
from node_capture import HEADER, TENSOR, parse_snapshot, make_return_bundle, collect
from pcie_backend import MAGIC, PBTools, MAILBOX, READY, NODE, DONE, ERROR


class CollectionTests(unittest.TestCase):
    """Exercise host collection, including an input arena slot reused by output."""
    def run_capture(self, missing_input=False):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/'model').mkdir()
            (root/'model/constants.bin').write_bytes(b'')
            prepared=struct.pack('<f',1); changed=struct.pack('<f',1.5)
            (root/'input.bin').write_bytes(prepared)
            result=struct.pack('<6f',1.5,0,1,1,.5,0)*300
            reference_result=struct.pack('<6f',1,0,1,1,.5,0)*300
            tensors=[None]*549
            for tid in (0,1,548):
                tensors[tid]={'dtype':1,'shape':[300,6] if tid==548 else [1],
                              'bytes':7200 if tid==548 else 4,'constant':0,'offset':0}
            nodes=[{'op':5,'kind':'COPY','module':'test','inputs':[0],'outputs':[1]},
                   {'op':5,'kind':'COPY','module':'test','inputs':[1],'outputs':[548]}]
            expected={0:prepared,1:prepared,548:reference_result}

            class Archive:
                def take(self,tid): return expected.pop(tid)
                def finish(self):
                    if expected: raise AssertionError('Missing tensor reference')

            class Backend:
                frame=0; session=123; sequence=1; step=0; state=READY
                input_address=0xf0000000; output_address=0xf0000100
                last_command=None; last_diagnostic=None
                def __init__(self): self.link=self; self.timeout=.01; self.data=prepared
                def write(self,address,data,deadline): self.data=data
                def read(self,address,size,deadline):
                    if address==MAILBOX+128: return struct.pack('<I',self.sequence)+bytes(60)
                    data=result if address==self.output_address else self.data
                    return data[:size]
                def _status(self,deadline):
                    return {'state':self.state,'session':self.session,'frame':self.frame,
                            'sequence':self.sequence,'detail':self.step,'output_bytes':7200,
                            'output_crc':zlib.crc32(result)}
                def _submit(self,command,size,token,deadline):
                    self.last_command=command; self.sequence+=1
                    if command==2: self.state=NODE; self.data=changed
                    elif command==3:
                        if token!=self.step+1: raise AssertionError('Wrong ACK token')
                        self.step+=1; self.state=NODE if self.step<2 else DONE
                    elif command==4: self.state=ERROR
                def _snapshot(self,deadline): self.last_diagnostic={'stage':20}
                def _pause(self,deadline): raise AssertionError('Fixture unexpectedly waiting')

            backend=Backend()
            def snapshot(backend,state,deadline):
                step=backend.step; inp=step; out=1 if step==0 else 548
                def descriptor(tid,direction):
                    t=tensors[tid]
                    raw=result if tid==548 else changed
                    return {'id':tid,'direction':direction,'address':backend.output_address if tid==548 else backend.input_address,
                            **t,'count':t['bytes']//4,
                            'crc_before':zlib.crc32(prepared if step==0 and direction=='input' else raw),
                            'crc_after':zlib.crc32(raw)}
                ds=[descriptor(inp,'input'),descriptor(out,'output')]
                if missing_input and step==0: ds=ds[1:]
                return {'node':step,'op':5,'token':step+1,'amu_flags':0,'node_cycles':10,'tensors':ds}
            output=root/'result'; output.mkdir()
            with patch('node_capture.ROOT',root),patch('node_capture.read_node',snapshot):
                if missing_input:
                    with self.assertRaisesRegex(ValueError,'node inputs'):
                        collect(backend,Archive(),{'tensors':tensors,'nodes':nodes},
                                {'input':'input.bin'},prepared,output,2,lambda event:None)
                    self.assertEqual(backend.last_command,4)
                    self.assertEqual(backend.state,ERROR)
                    self.assertTrue((output/'partial-nodes.json').exists())
                else:
                    rows,first,actual=collect(backend,Archive(),{'tensors':tensors,'nodes':nodes},
                                             {'input':'input.bin'},prepared,output,2,lambda event:None)
                    self.assertEqual(len(rows),2); self.assertEqual(actual,result)
                    self.assertEqual(first['node'],0)
                    self.assertTrue(first['same_input_full_bytes_verified'])
                    self.assertIn('storage was reused',first['inputs'][0]['source'])
                    self.assertEqual((output/'first-input-0000.bin').read_bytes(),prepared)
                    self.assertEqual((output/'tensors/0548.bin').read_bytes(),result)
                    self.assertFalse(rows[1]['same_input_reference_crc'])
                    self.assertEqual(backend.state,DONE)

    def test_complete_collection_and_overwritten_input_reconstruction(self):
        self.run_capture()

    def test_missing_input_aborts_and_preserves_partial_report(self):
        self.run_capture(missing_input=True)


class CompareTests(unittest.TestCase):
    def test_full_tensor_not_only_sampled_values(self):
        expected=struct.pack('<1024f',*([1.0]*1024))
        actual=bytearray(expected)
        struct.pack_into('<f',actual,900,1.5)
        result=compare_tensor(bytes(actual),expected,1,[1,1024])
        self.assertEqual(result['different_elements'],1)
        self.assertEqual(result['samples'][0]['element'],225)
        self.assertEqual(result['max_absolute_difference'],.5)
        self.assertEqual(result['mean_absolute_difference'],.5/1024)
        self.assertFalse(result['byte_identical'])

    def test_int64_preserves_large_integer_difference(self):
        expected=struct.pack('<q',2**60)
        actual=struct.pack('<q',2**60+1)
        self.assertEqual(compare_tensor(actual,expected,2,[1])['max_absolute_difference'],1)

    def test_signed_zero_is_a_byte_difference(self):
        result=compare_tensor(struct.pack('<I',0x80000000),struct.pack('<f',0),1,[1])
        self.assertFalse(result['byte_identical'])
        self.assertEqual(result['different_elements'],1)
        self.assertEqual(result['max_absolute_difference'],0)

    def test_convolution_replays_same_input_and_labels_candidate_rules(self):
        tensors=[{'shape':[1,1,1,3]},{'shape':[1,1,1,3]},{'shape':[1]},
                 {'shape':[1,1,1,1]}]
        node={'kind':'CONV','inputs':[0,1,2],'outputs':[3],'params':[1,1,0,0,1]}
        inputs={0:struct.pack('<3f',1,1,1),1:struct.pack('<3f',1,2**-24,-1),2:struct.pack('<f',0)}
        actual=struct.pack('<f',0); reference=struct.pack('<f',2**-24)
        replay=replay_convolution(node,tensors,inputs,actual,reference,[{'element':0}])
        rules=replay['results'][0]['rules']
        self.assertTrue(rules['qemu_fp64_dot_fp32_per_k32']['matches_reference_bits'])
        self.assertTrue(rules['sequential_fp32']['matches_actual_bits'])
        self.assertIn('do not establish',replay['note'])

    def test_chunked_lossless_reference_and_corrupt_part(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            data=struct.pack('<2f',1,2)
            raw=struct.pack('<8sII',b'Y26SCAL1',1,1)+struct.pack('<3I',0,1,8)+data
            compressed=gzip.compress(raw,mtime=0)
            parts=[]
            for i,block in enumerate((compressed[:13],compressed[13:])):
                path=root/f'part{i}'; path.write_bytes(block)
                parts.append({'path':path.name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
            manifest={'tensors':[{'dtype':1,'bytes':8}],'samples':{'bus':{'parts':parts,
                      'records':[{'id':0,'sha256':hashlib.sha256(data).hexdigest()}]}}}
            archive=ReferenceArchive(root,manifest,'bus')
            try:
                self.assertEqual(archive.take(0),data); archive.finish()
            finally:
                archive.close()
            (root/'part1').write_bytes(bytes(len(compressed)-13))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                ReferenceArchive(root,manifest,'bus')

    def test_return_bundle_never_contains_full_tensors_or_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/'tensors').mkdir()
            (root/'tensors/0003.bin').write_bytes(bytes(2*1024*1024))
            (root/'first-input-0000.bin').write_bytes(bytes(2*1024*1024))
            (root/'summary.json').write_text('{}')
            (root/'nodes.csv').write_text('node,tensor\n0,3\n')
            (root/'first_difference.json').write_text('{}')
            bundle=make_return_bundle(root)
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(set(archive.namelist()),{'summary.json','nodes.csv','first_difference.json'})
            self.assertLess(bundle.stat().st_size,1024*1024)


class DescriptorTests(unittest.TestCase):
    def snapshot(self):
        blob=bytearray(512)
        HEADER.pack_into(blob,0,MAGIC,1,512,0,123,1,2,5,1,1,1,12,0,0)
        for slot in range(6):
            struct.pack_into('<i',blob,64+slot*64,-1)
        TENSOR.pack_into(blob,64,0,1,1,8,0xf0000000,0,2,2,0,0,0,123,123,0,0)
        TENSOR.pack_into(blob,320,1,1,1,8,0xf0000040,0,2,2,0,0,0,456,456,0,0)
        struct.pack_into('<I',blob,508,zlib.crc32(blob[:508]))
        return bytes(blob)

    def test_identity_crc_shape_and_reserved_address_validation(self):
        backend=type('Backend',(),{'session':123,'frame':1,'sequence':2,'options':12,
                                 'link':type('Link',(),{'_range':staticmethod(PBTools._range)})()})()
        blob=self.snapshot()
        state={'detail':0,'output_bytes':512,'output_crc':zlib.crc32(blob)}
        self.assertEqual(parse_snapshot(blob,state,backend)['tensors'][1]['shape'],[2])
        corrupted=bytearray(blob); corrupted[100]^=1
        with self.assertRaisesRegex(ValueError,'CRC'):
            parse_snapshot(bytes(corrupted),state,backend)
        backend.frame=2
        with self.assertRaisesRegex(ValueError,'identity'):
            parse_snapshot(blob,state,backend)


if __name__=='__main__':
    unittest.main()
