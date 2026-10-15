"""Offline first-layer diagnostics: known arithmetic, provenance and page bounds."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import analyze_first_node as analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run=self.root/'capture'; (self.run/'tensors').mkdir(parents=True)
        for name in ('model','reference/nodes','reference/prepared'):
            (self.root/name).mkdir(parents=True)
        self.x=struct.pack('<75f',*([1]*75))
        self.w=struct.pack('<27f',1,*([0]*26)); self.bias=struct.pack('<f',1)
        constants=self.w+self.bias
        (self.root/'model/constants.bin').write_bytes(constants)
        files=[{'name':'constants.bin','bytes':len(constants),'sha256':analysis.digest(constants)}]
        (self.root/'model/manifest.json').write_text(json.dumps({'files':files}))
        self.tensors=[{'id':0,'dtype':1,'bytes':300,'constant':0,'offset':0,'shape':[1,3,5,5]},
                      {'id':1,'dtype':1,'bytes':108,'constant':1,'offset':0,'shape':[1,3,3,3]},
                      {'id':2,'dtype':1,'bytes':4,'constant':1,'offset':108,'shape':[1]},
                      {'id':3,'dtype':1,'bytes':36,'constant':0,'offset':0,'shape':[1,1,3,3]}]
        self.node={'id':0,'kind':'CONV','module':'model.0.conv','inputs':[0,1,2],
                   'outputs':[3],'params':[1,1,0,0,1]}
        reference=struct.pack('<9f',*([2]*9))
        values=[2]*9; values[4]=2+2**-20
        actual=struct.pack('<9f',*values)
        manifest={'nodes':[self.node],'tensors':self.tensors,
                  'model_files_sha256':{'constants.bin':analysis.digest(constants)},
                  'samples':{'bus':{'records':[{'id':0,'sha256':analysis.digest(self.x)},
                                             {'id':3,'sha256':analysis.digest(reference)}]}}}
        (self.root/'reference/nodes/manifest.json').write_text(json.dumps(manifest))
        (self.root/'reference/prepared/input.bin').write_bytes(self.x)
        (self.root/'reference/prepared/manifest.json').write_text(json.dumps({'samples':{'bus':{
            'input':'reference/prepared/input.bin','input_sha256':analysis.digest(self.x)}}}))
        first={'node':0,'kind':'CONV','module':self.node['module'],'tensor':3,
               'shape':self.tensors[3]['shape'],'inputs':[],
               'metrics':{'sha256_actual':analysis.digest(actual),'sha256_reference':analysis.digest(reference)}}
        for tid,data in enumerate((self.x,self.w,self.bias)):
            (self.run/f'first-input-{tid:04d}.bin').write_bytes(data)
            first['inputs'].append({'tensor':tid,'sha256_actual':analysis.digest(data),
                                    'sha256_reference':analysis.digest(data)})
        (self.run/'first_difference.json').write_text(json.dumps(first))
        (self.run/'summary.json').write_text(json.dumps({'status':'failed','sample':'bus',
                                                       'input_sha256':analysis.digest(self.x)}))
        (self.run/'first-reference.bin').write_bytes(reference)
        (self.run/'tensors/0003.bin').write_bytes(actual)

    def test_verified_old_capture_math_and_screenshot_pages(self):
        originals={p:p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        report=analysis.analyze(self.run,self.root)
        self.assertEqual(report['max_ulp']['element'],4)
        self.assertEqual(report['max_ulp']['ulp'],4)
        self.assertEqual(len(report['terms']),27)
        self.assertEqual(report['terms'][0]['input_coordinate'],[0,0,1,1])
        self.assertEqual(report['terms'][0]['input_bits'],'3f800000')
        self.assertEqual(report['terms'][0]['input_half'],0x3c00)
        self.assertEqual(report['dot_fp64'],1)
        self.assertEqual(report['bias'],1)
        for rule in report['rules']:
            self.assertEqual(rule['before_bias'],1)
            self.assertEqual(rule['after_bias'],2)
            self.assertTrue(rule['matches_reference'])
            self.assertFalse(rule['matches_board'])
        for page in range(1,5):
            output=analysis.render(report,page)
            self.assertLessEqual(len(output.splitlines()),22)
            self.assertTrue(all(len(line)<=80 for line in output.splitlines()))
            self.assertIn('capture=failed',output)
        self.assertIn('NOT captured',analysis.render(report,2))
        self.assertIn('NOT measured AMU buffers',analysis.render(report,3))
        self.assertEqual(originals,{p:p.read_bytes() for p in self.run.rglob('*') if p.is_file()})

    def test_max_ulp_and_max_absolute_can_be_different_elements(self):
        ref=struct.pack('<2f',1e-8,10)
        first=analysis.word(ref,0)
        actual=struct.pack('<I',first+32768)+struct.pack('<f',10.001)
        ulp,absolute=analysis.extrema(actual,ref,[1,1,1,2])
        self.assertEqual(ulp['element'],0)
        self.assertEqual(ulp['ulp'],32768)
        self.assertEqual(absolute['element'],1)

    def test_nonfinite_output_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            analysis.extrema(struct.pack('<f',float('nan')),struct.pack('<f',1),[1])

    def test_tampered_saved_files_rejected(self):
        for name in ('first-input-0000.bin','first-reference.bin','tensors/0003.bin'):
            path=self.run/name; old=path.read_bytes()
            path.write_bytes(bytes(len(old)))
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'hash/length mismatch'):
                analysis.analyze(self.run,self.root)
            path.write_bytes(old)

    def test_changed_model_rejected(self):
        (self.root/'model/constants.bin').write_bytes(bytes(112))
        with self.assertRaisesRegex(ValueError,'hash/length mismatch'):
            analysis.analyze(self.run,self.root)

    def test_missing_input_rejected(self):
        (self.run/'first-input-0002.bin').unlink()
        with self.assertRaises(FileNotFoundError):
            analysis.analyze(self.run,self.root)

    def test_padding_and_fp16_ties_to_even(self):
        tensors=copy.deepcopy(self.tensors)
        tensors[3]['shape']=[1,1,5,5]
        node=dict(self.node,params=[1,1,1,1,1])
        x=struct.pack('<75f',2**-25,3*2**-25,*([1]*73))
        terms,bias=analysis.dot_terms(node,tensors,{0:x,1:self.w,2:self.bias},0)
        self.assertTrue(terms[0]['padding'])
        self.assertIsNone(terms[0]['input_index'])
        self.assertEqual(terms[0]['input_half'],0)
        self.assertEqual(terms[4]['input_half'],0)
        self.assertEqual(terms[5]['input_half'],2)
        self.assertEqual(terms[26]['weight_index'],26)


if __name__=='__main__': unittest.main()
