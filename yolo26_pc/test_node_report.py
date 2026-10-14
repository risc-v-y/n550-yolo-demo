"""Screenshot readability, failure reporting and backdated-host selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from node_compare import compare_tensor,replay_convolution
from node_report import remember_report,render_page
import struct


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.run=self.root/'2021-run'; self.run.mkdir()
        self.summary={'status':'completed','sample':'bus','nodes_captured':334,
                      'tensors_captured':335,'different_nodes':1,'input_sha256':'a'*64,
                      'normal_repeat':{'byte_identical_to_capture':False},'elapsed_seconds':123.456}
        self.save()

    def tearDown(self): self.temp.cleanup()

    def save(self):
        (self.run/'summary.json').write_text(json.dumps(self.summary),encoding='utf-8')

    def assert_screen(self,page):
        self.assertLessEqual(len(page.splitlines()),22)
        self.assertTrue(all(len(line)<=80 for line in page.splitlines()))

    def write_first_difference(self):
        tensors=[{'id':0,'dtype':1,'shape':[1,1,1,3]},
                 {'id':1,'dtype':1,'shape':[1,1,1,3]},
                 {'id':2,'dtype':1,'shape':[1]},{'shape':[1,1,1,1]}]
        node={'kind':'CONV','inputs':[0,1,2],'outputs':[3],'params':[1,1,0,0,1]}
        inputs={0:struct.pack('<3f',1,1,1),1:struct.pack('<3f',1,2**-24,-1),2:struct.pack('<f',0)}
        actual=struct.pack('<f',0); reference=struct.pack('<f',2**-24)
        metrics=compare_tensor(actual,reference,1,[1,1,1,1])
        first={'node':0,'kind':'CONV','tensor':3,'module':'model.0','shape':[1,1,1,1],
               'input_descriptors':tensors[:3],'inputs':[{'tensor':i,'byte_identical_to_reference':True} for i in range(3)],
               'same_input_full_bytes_verified':True,'metrics':metrics,
               'replay':replay_convolution(node,tensors,inputs,actual,reference,metrics['samples'])}
        (self.run/'first_difference.json').write_text(json.dumps(first))
        return first

    def test_all_pages_with_real_replay_and_input_shapes(self):
        self.write_first_difference()
        pages=[render_page(self.run,page) for page in (1,2,3)]
        for page in pages: self.assert_screen(page)
        self.assertEqual(len({page.splitlines()[1] for page in pages}),1)
        self.assertIn('Ordinary repeat == capture: NO',pages[0])
        self.assertIn('Input t0 FP32 1x1x1x3 full-match=YES',pages[1])
        self.assertIn('[33800000]',pages[1])
        self.assertIn('Q: 0/1 / 1/1',pages[2])
        self.assertIn('S: 1/1 / 0/1',pages[2])
        self.assertIn('33800000',pages[2])
        self.assertIn('do not establish',pages[2])

    def test_failed_capture_can_show_saved_difference_without_hiding_failure(self):
        first=self.write_first_difference()
        # Exercise the screen limit with four inputs and three differing samples.
        first['input_descriptors'].append({'id':4,'dtype':1,'shape':[]})
        first['metrics']['samples']*=3
        (self.run/'first_difference.json').write_text(json.dumps(first))
        self.summary={'status':'failed','sample':'bus','exception':'PCIeToolError',
                      'error':'pbload failed (-11)','elapsed_seconds':2026.75}
        self.save()
        before={p.name:p.read_bytes() for p in self.run.iterdir()}
        pages=[render_page(self.run,page,view='difference') for page in (1,2,3)]
        for page in pages:
            self.assert_screen(page)
            self.assertIn('PARTIAL CAPTURE',page)
            self.assertIn('status=failed',page)
        self.assertIn('Final bytes match QEMU: unknown',pages[0])
        self.assertIn('Different elements:',pages[1])
        self.assertIn('Input t4 FP32 scalar',pages[1])
        self.assertIn('Q: 0/1 / 1/1',pages[2])
        self.assertIn('do not establish',pages[2])
        self.assertIn('pbload failed (-11)',render_page(self.run,2))
        self.assertEqual(len({page.splitlines()[1] for page in pages}),1)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.run.iterdir()})

    def test_difference_view_without_saved_difference_is_explicit(self):
        self.summary={'status':'failed','sample':'bus','error':'early failure'}
        self.save()
        for page in (2,3):
            output=render_page(self.run,page,view='difference')
            self.assert_screen(output)
            self.assertIn('No first differing tensor was recorded',output)
            self.assertIn('PARTIAL CAPTURE',output)

    def test_partial_failure_and_long_error_fit_screens(self):
        self.summary={'status':'failed','sample':'bus','exception':'TimeoutError',
                      'error':'PCIe timeout. '+'long tool failure '*200,
                      'last_status':{'state':2,'sequence':5,'frame':1},
                      'last_diagnostic':{'node':3,'error':0,'phase_name':'compute'}}
        self.save()
        (self.run/'nodes.csv').write_text('node,byte_identical\n0,True\n1,False\n')
        for page in (1,2,3): self.assert_screen(render_page(self.run,page))
        self.assertIn('Nodes captured: 2/334',render_page(self.run,1))
        self.assertIn('Last captured node: 1',render_page(self.run,1))
        self.assertIn('Last node=3 phase=compute',render_page(self.run,2))
        self.assertIn('Further error text remains',render_page(self.run,3))
        # An earlier numerical difference must not hide a later capture error.
        (self.run/'first_difference.json').write_text(json.dumps({'node':0,'kind':'COPY','tensor':3}))
        self.assertIn('TimeoutError',render_page(self.run,2))

    def test_latest_pointer_uses_completion_not_calendar(self):
        future=self.root/'2026-run'; future.mkdir()
        (future/'summary.json').write_text(json.dumps({'status':'completed','sample':'zidane'}))
        with patch('node_report.LATEST',self.root/'latest.json'):
            remember_report(future)
            remember_report(self.run)
            self.assertIn('sample=bus',render_page())

    def test_missing_repeat_and_no_difference_are_not_claimed_passed(self):
        del self.summary['normal_repeat']; self.save()
        self.assertIn('Ordinary repeat == capture: unknown',render_page(self.run,1))
        self.assertIn('No first differing tensor',render_page(self.run,2))


if __name__=='__main__': unittest.main()
