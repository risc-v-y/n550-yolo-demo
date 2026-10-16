"""Full workflow failures, actual-input provenance and screenshot bounds."""
import argparse
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib
import full_check as job
import full_evidence as ev
from node_capture import HEADER,TENSOR
from pcie_backend import MAGIC
from full_replay import sha


class SavedReportTests(unittest.TestCase):
    def test_explicit_failed_reanalysis_is_not_hidden_by_original_report(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)
            (directory/'summary.json').write_text('{"status":"passed"}')
            (directory/'bus-analysis.json').write_text('{"id":"latest","status":"failed","errors":["damaged input"]}')
            report=job.saved_report(directory)
            self.assertEqual(report['status'],'failed')
            self.assertIn('damaged input',job.render(report,2))

    def test_incomplete_bus_never_inherits_parent_pass_and_pages_are_bounded(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)
            bus={'passed':True,'nodes_captured':333,'rows':[],
                 'outputs_captured':334,'inputs_verified_nodes':333,'final':{'normal_repeat_exact':True}}
            path=directory/'summary.json'
            path.write_text(json.dumps({'status':'passed','samples':{'bus':bus},'tile_gate':{'status':'passed'}}))
            original=path.read_bytes()
            report=job.saved_report(directory)
            self.assertEqual(report['status'],'not_passed')
            self.assertEqual(job.saved_report(directory)['id'],report['id'])
            for page in (1,2,3):
                lines=job.render(report,page).splitlines()
                self.assertLessEqual(len(lines),22)
                self.assertTrue(all(len(line)<=80 for line in lines))
            self.assertEqual(path.read_bytes(),original)


class FinalResultTests(unittest.TestCase):
    def check(self,raw,sample):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name); (directory/'tensors').mkdir()
            (directory/'tensors/0548.bin').write_bytes(raw)
            (directory/'normal-candidates.bin').write_bytes(raw)
            with patch.object(ev,'save_annotation') as drawing:
                result=ev.final_check(directory,sample,{'normal_repeat':{'sha256':sha(raw),'byte_identical_to_capture':True}})
                self.assertEqual(drawing.call_args.args[0].name,'full-check-annotated.png')
                self.assertEqual(result['annotated'],'full-check-annotated.png')
                return result

    def test_bus_frozen_candidates_and_drawing(self):
        samples=ev.read_json(ev.ROOT/'reference/prepared/manifest.json')['samples']
        for name,count in (('bus',6),):
            sample=samples[name]; raw=(ev.ROOT/sample['reference']).read_bytes()[32:]
            result=self.check(raw,sample)
            self.assertTrue(result['normal_repeat_exact']); self.assertEqual(len(result['detections']),count)
            self.assertTrue(result['qemu']['byte_identical'])

    def test_no_detections_is_valid_and_bad_values_are_rejected(self):
        sample=ev.read_json(ev.ROOT/'reference/prepared/manifest.json')['samples']['bus']
        candidate=[0,0,1,1,0,0]
        self.assertEqual(self.check(struct.pack('<6f',*candidate)*300,sample)['detections'],[])
        for index,value in ((0,float('nan')),(0,2),(4,1.1),(5,80),(5,.5)):
            bad=list(candidate); bad[index]=value
            with self.assertRaises(ValueError): self.check(struct.pack('<6f',*bad)*300,sample)
        with self.assertRaises(ValueError): self.check(bytes(7196),sample)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.directory=self.root/'capture'; self.directory.mkdir()
        (self.directory/'snapshots').mkdir(); (self.directory/'tensors').mkdir()
        (self.root/'reference/nodes').mkdir(parents=True)
        (self.root/'firmware').mkdir(); (self.root/'firmware/demo-pcie.bin').write_bytes(b'firmware')
        self.input=struct.pack('<f',1.0); self.changed=struct.pack('<f',1.5)
        self.final=struct.pack('<6f',1.5,0,2,2,.5,0)*300
        self.tensors=[{'dtype':1,'shape':[1],'bytes':4,'constant':0,'count':1} for _ in range(549)]
        self.tensors[548].update(shape=[300,6],bytes=7200,count=1800)
        self.nodes=[{'id':0,'op':5,'kind':'COPY','inputs':[0],'outputs':[1]},
                    {'id':1,'op':5,'kind':'COPY','inputs':[1],'outputs':[548]}]
        self.manifest={'tensors':self.tensors,'nodes':self.nodes,'samples':{'bus':{'records':[
            {'id':1,'sha256':sha(self.input)},{'id':548,'sha256':sha(bytes(7200))}]}}}
        (self.root/'reference/nodes/manifest.json').write_text(json.dumps(self.manifest))
        self.prepared={'input_sha256':sha(self.input)}
        self.summary={'status':'completed','sample':'bus','evidence_version':1,'input_sha256':sha(self.input),
                      'firmware_sha256':sha(b'firmware'),
                      'reference_manifest_sha256':sha((self.root/'reference/nodes/manifest.json').read_bytes())}
        (self.directory/'summary.json').write_text(json.dumps(self.summary))
        (self.directory/'input.bin').write_bytes(self.input)
        (self.directory/'tensors/0001.bin').write_bytes(self.changed)
        (self.directory/'tensors/0548.bin').write_bytes(self.final)
        for node in self.nodes: self.snapshot(node)
        p=patch.object(ev,'ROOT',self.root); p.start(); self.addCleanup(p.stop)

    def snapshot(self,node):
        i=node['inputs'][0]; o=node['outputs'][0]; blob=bytearray(512)
        HEADER.pack_into(blob,0,MAGIC,1,512,node['id'],123,1,node['id']+2,5,1,1,node['id']+1,12,0,0)
        for slot in range(6): struct.pack_into('<i',blob,64+slot*64,-1)
        proof=[]
        for slot,tid in ((0,i),(4,o)):
            data=self.input if tid==0 else self.changed if tid==1 else self.final
            t=self.tensors[tid]; s=t['shape']; crc=zlib.crc32(data)
            # Node 0 overwrites input storage, so pre/post CRC differ.
            after=zlib.crc32(self.changed) if tid==0 else crc
            TENSOR.pack_into(blob,64+slot*64,tid,1,len(s),len(data),0xf0000000,0,t['count'],
                             *(s+[0]*(4-len(s))),crc,after,0,0)
            if slot==0: proof=[{'id':tid,'verified':True,'crc32_before':crc,
                               'sha256':sha(data),'path':'input.bin' if tid==0 else 'tensors/0001.bin'}]
        struct.pack_into('<I',blob,508,zlib.crc32(blob[:508])); raw=bytes(blob)
        h=HEADER.unpack_from(raw)
        backend=type('B',(),{'session':123,'frame':1,'sequence':h[6],'options':12,
                 'link':type('L',(),{'_range':staticmethod(ev.PBTools._range)})()})()
        state={'detail':node['id'],'output_bytes':512,'output_crc':zlib.crc32(raw)}
        snap=ev.parse_snapshot(raw,state,backend)
        prefix=self.directory/'snapshots/{:04d}'.format(node['id'])
        prefix.with_suffix('.bin').write_bytes(raw)
        prefix.with_suffix('.json').write_text(json.dumps({'version':1,'snapshot':snap,'inputs':proof,
            'snapshot_sha256':sha(raw),'outputs':[{'id':o,'sha256':sha(self.changed if o==1 else self.final)}]}))

    def audit(self):
        owner=self
        class Native:
            def __init__(self,mode): pass
            def run(self,node,tensors,inputs):
                if node['id']==0: return {1:owner.input}
                # Correct local output from a DIFFERENT upstream input.
                owner.assertEqual(inputs[1],owner.changed)
                return {548:owner.final}
            def close(self): pass
        with patch.object(ev,'final_check',return_value={'normal_repeat_exact':True}):
            return ev.audit(self.directory,'bus',self.manifest,self.prepared,replay_factory=Native)

    def test_local_error_vs_upstream_propagation_and_reused_input(self):
        r=self.audit()
        self.assertEqual([n['status'] for n in r['rows']],['unexplained','exact'])
        self.assertTrue(r['rows'][1]['upstream_propagation'])
        self.assertEqual(r['nodes_captured'],2)
        self.assertFalse(r['passed'])

    def test_single_bit_corruption_is_not_tolerated_and_does_not_stop_later_nodes(self):
        path=self.directory/'tensors/0001.bin'; raw=bytearray(path.read_bytes()); raw[0]^=1; path.write_bytes(raw)
        r=self.audit()
        self.assertEqual(r['rows'][0]['status'],'missing')
        self.assertEqual(r['rows'][1]['status'],'input_unverified')

    def test_incomplete_and_legacy_capture_cannot_be_reused(self):
        (self.directory/'snapshots/0001.bin').unlink()
        self.assertFalse(ev.reusable(self.directory,'bus',self.manifest,self.prepared))
        r=self.audit(); self.assertEqual(r['rows'][1]['status'],'missing')
        del self.summary['evidence_version']
        (self.directory/'summary.json').write_text(json.dumps(self.summary))
        self.assertFalse(ev.reusable(self.directory,'bus',self.manifest,self.prepared))

    def test_input_before_crc_is_required_even_when_output_matches(self):
        path=self.directory/'snapshots/0001.json'; r=json.loads(path.read_text())
        r['inputs'][0]['crc32_before']^=1; path.write_text(json.dumps(r))
        self.assertEqual(self.audit()['rows'][1]['status'],'input_unverified')

    def test_missing_and_duplicate_snapshot_rejected(self):
        prefix=self.directory/'snapshots'
        (prefix/'0001.bin').write_bytes((prefix/'0000.bin').read_bytes())
        self.assertEqual(self.audit()['rows'][1]['status'],'missing')

    def test_capture_identity_accepts_normalized_line_endings_but_not_changed_graph(self):
        path=self.root/'reference/nodes/manifest.json'
        lf=(json.dumps(self.manifest,indent=2)+'\n').encode()
        self.summary['reference_manifest_sha256']=sha(lf)
        (self.directory/'summary.json').write_text(json.dumps(self.summary))
        for data in (lf,lf.replace(b'\n',b'\r\n')):
            path.write_bytes(data)
            ev.identity(self.directory,'bus',self.manifest,self.prepared)
        path.write_bytes(lf.replace(b'"COPY"',b'"ADD"',1))
        with self.assertRaisesRegex(ValueError,'graph/sample/input evidence'):
            ev.identity(self.directory,'bus',self.manifest,self.prepared)

    def test_complete_node_capture_without_ordinary_repeat_cannot_be_reused(self):
        prepared=job.read_json(job.ROOT/'reference/prepared/manifest.json')['samples']['bus'].copy()
        prepared['input_sha256']=self.prepared['input_sha256']
        self.summary['normal_repeat']={'sha256':sha(self.final),'byte_identical_to_capture':True}
        (self.directory/'summary.json').write_text(json.dumps(self.summary))
        normal=self.directory/'normal-candidates.bin'; normal.write_bytes(self.final)
        self.assertTrue(ev.reusable(self.directory,'bus',self.manifest,prepared))
        normal.unlink()
        self.assertFalse(ev.reusable(self.directory,'bus',self.manifest,prepared))


class WorkflowTests(unittest.TestCase):
    def run_job(self,stop=False,numerical_error=False,reuse=False):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); (root/'reference/nodes').mkdir(parents=True)
            (root/'reference/prepared').mkdir(parents=True); (root/'firmware').mkdir()
            (root/'reference/nodes/manifest.json').write_text('{}')
            (root/'reference/prepared/manifest.json').write_text('{"samples":{"bus":{}}}')
            (root/'firmware/host-verify.json').write_text('{}')
            args=argparse.Namespace(output=root/'result',fresh=False,probe=None)
            def collect(args,sample,output): return output/'attempt-1',stop
            def analyze(directory,sample,*args,**kwargs):
                return {'passed':not numerical_error,'rows':[],'counts':{},'final':{},'errors':[],
                        'nodes_captured':334,'outputs_captured':335,'inputs_verified_nodes':334}
            with patch.object(job,'ROOT',root),patch.object(job,'Replay'),patch.object(job,'check_renderer'), \
                 patch.object(job,'verify_reference_manifest'), \
                 patch.object(job,'probe_gate',return_value={'status':'passed'}), \
                 patch.object(job,'find_reuse',return_value=root/'old/bus/attempt-1' if reuse else None), \
                 patch.object(job,'audit',side_effect=analyze), \
                 patch.object(job,'capture',side_effect=collect) as capture:
                code=job.run(args)
            return code,json.loads((root/'result/summary.json').read_text()),capture.call_count

    def test_numeric_difference_finishes_bus_without_starting_another_image(self):
        code,report,count=self.run_job(numerical_error=True)
        self.assertEqual(count,1); self.assertEqual(code,1)
        self.assertEqual(set(report['samples']),{'bus'})
        self.assertEqual(report['status'],'not_passed')

    def test_terminal_board_failure_saves_bus_summary(self):
        code,report,count=self.run_job(stop=True)
        self.assertEqual(count,1); self.assertEqual(code,1)
        self.assertTrue(report['errors']); self.assertEqual(set(report['samples']),{'bus'})

    def test_complete_bus_result_passes(self):
        code,report,count=self.run_job()
        self.assertEqual((code,count,report['status']),(0,1,'passed'))

    def test_validated_bus_reuse_never_loads_or_captures_again(self):
        with patch.object(job,'load') as load:
            code,report,count=self.run_job(reuse=True)
        load.assert_not_called()
        self.assertEqual((code,count,report['status']),(0,0,'passed'))

    def test_only_one_whole_frame_retry_for_tagged_read_failure(self):
        for retryable,expected in ((True,2),(False,1)):
            with tempfile.TemporaryDirectory() as name:
                path=Path(name); args=argparse.Namespace(pbcopy='copy',pbload='read',node_timeout=300,tool_timeout=30)
                def child(command,log):
                    d=Path(command[-1]); d.mkdir()
                    (d/'summary.json').write_text(json.dumps({'status':'failed','retryable_capture_failure':retryable}))
                    return 1
                with patch.object(job,'load') as load,patch.object(job,'command',side_effect=child):
                    _,stop=job.capture(args,'bus',path)
                self.assertTrue(stop); self.assertEqual(load.call_count,expected)

    def test_write_or_reset_failure_never_starts_capture(self):
        with patch.object(job,'load',side_effect=RuntimeError('write failed')),patch.object(job,'command') as command:
            with self.assertRaises(RuntimeError): job.capture(argparse.Namespace(),'bus',Path('unused'))
            command.assert_not_called()

    def test_three_pages_are_bounded_even_with_all_nodes_bad(self):
        r={'id':'123456789abc','status':'not_passed','samples':{},'errors':['x'*200]*20}
        for sample in ('bus',):
            r['samples'][sample]={'rows':[{'node':i,'kind':'CONV','status':'unexplained','outputs':[]} for i in range(334)],
                                  'errors':['long error '*20]}
        for page in (1,2,3):
            lines=job.render(r,page).splitlines()
            self.assertLessEqual(len(lines),22); self.assertLessEqual(max(map(len,lines)),80)

    def test_successful_page_does_not_invent_missing_nodes(self):
        r={'id':'123456789abc','status':'passed','samples':{
            name:{'nodes_captured':334,'outputs_captured':335,'inputs_verified_nodes':334,
                  'counts':{'exact':334}} for name in ('bus',)}}
        self.assertEqual(job.render(r,1).count('missing=0'),1)


class LegacyBusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.capture=self.directory/'bus/attempt-1'; self.capture.mkdir(parents=True)
        self.original={'id':'legacy','samples':{'bus':{'directory':str(self.capture)},
                       'zidane':{'passed':False}},'errors':['zidane failed'],'status':'not_passed'}

    def complete(self):
        (self.capture/'summary.json').write_text('{"sample":"bus","status":"completed"}')

    def reanalyze(self,passed=True):
        args=argparse.Namespace(report=self.directory,probe=None)
        result={'passed':passed,'rows':[],'counts':{},'final':{},'errors':[]}
        with patch.object(job,'assets',return_value=({},{})), \
             patch.object(job,'probe_gate',return_value={'status':'passed'}), \
             patch.object(job,'audit',return_value=result) as audit, \
             patch.object(job,'capture') as capture,patch.object(job,'load') as load:
            report=job.reanalyze(args)
        capture.assert_not_called(); load.assert_not_called()
        return report,audit.call_args.args[0]

    def test_old_second_image_failure_is_excluded_after_raw_bus_reaudit(self):
        self.complete()
        original=json.dumps(self.original); (self.directory/'summary.json').write_text(original)
        report,raw=self.reanalyze()
        self.assertEqual(raw,self.capture); self.assertEqual(report['status'],'passed')
        self.assertEqual(report['errors'],[]); self.assertEqual(set(report['samples']),{'bus'})
        self.assertEqual((self.directory/'summary.json').read_text(),original)
        self.assertTrue((self.directory/'bus-analysis.json').is_file())
        for page in (1,2,3): self.assertNotIn('zidane',job.render(report,page).lower())

    def test_completed_bus_is_found_before_parent_summary_exists(self):
        self.complete()
        self.assertEqual(job.bus_capture(self.directory),self.capture)
        report,raw=self.reanalyze(); self.assertEqual(raw,self.capture)
        self.assertEqual(report['status'],'passed')

    def test_derived_analysis_without_raw_summary_is_not_selected(self):
        (self.directory/'bus/analysis.json').write_text(json.dumps({'passed':True,'directory':str(self.capture)}))
        self.assertEqual(job.bus_capture(self.directory),self.directory/'bus/not-collected')
        report,raw=self.reanalyze(passed=False)
        self.assertEqual(report['status'],'not_passed')
        self.assertEqual(raw,self.directory/'bus/not-collected')

    def test_bus_with_failed_reaudit_is_not_accepted_from_old_passed_report(self):
        self.complete(); self.original['status']='passed'
        (self.directory/'summary.json').write_text(json.dumps(self.original))
        report,_=self.reanalyze(passed=False)
        self.assertEqual(report['status'],'not_passed')


if __name__=='__main__': unittest.main()
