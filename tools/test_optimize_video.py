"""Exercise the complete acceptance controller without a physical board."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import optimize_video as app


class AcceptanceTests(unittest.TestCase):
    def test_selection_final_gate_and_cleanup(self):
        video=app.dependencies()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'acceptance'; output.mkdir()
            (root/'firmware').mkdir(); (root/'firmware/demo-pcie.bin').write_bytes(b'firmware')
            image=video.np.zeros((48,64,3),dtype=video.np.uint8)
            raw,mapping=app.Preprocessor(video).prepare(image)
            logical=(video.np.frombuffer(raw,dtype=video.np.uint8).astype('<f4')/video.np.float32(255)).tobytes()
            frames=[dict(input=raw,logical=logical,transform=mapping,
                         input_sha256=hashlib.sha256(logical).hexdigest()) for _ in range(3)]
            class Golden:
                def __init__(self,*args): pass
                def predict(self,data): return bytes(7200),{}
                def close(self): pass
            class Board:
                def __init__(self,*args,**kwargs):
                    self.stream={'max_rows':32,'accumulator_bytes':4096}
                    self.last_timing={}
                def configure(self,c): self.config=c
                def infer_u8(self,data):
                    self.last_profile={'model_cycles':90-sum(self.config.values()),'service_cycles':100}
                    return bytes(7200)
                def close(self): pass
            def baseline(args,directory):
                records=[]
                for i in range(3):
                    (directory/('candidates-{:06d}.bin'.format(i))).write_bytes(bytes(7200))
                    records.append({'frame_id':i,'input_sha256':frames[i]['input_sha256'],
                                    'elapsed_seconds':1,'pcie_timing':{'board_profile':{'model_cycles':100}}})
                return {'status':'completed','results':records}
            def pipeline(args,directory,**kwargs):
                rows=[]
                for i in range(args.max_frames):
                    prefix='frame-{:06d}'.format(i)
                    (directory/(prefix+'-input.bin')).write_bytes(raw)
                    (directory/(prefix+'-output.bin')).write_bytes(bytes(7200))
                    video.cv2.imencode('.png',image)[1].tofile(str(directory/(prefix+'-source.png')))
                    rows.append({'frame_id':i,'input_sha256':hashlib.sha256(raw).hexdigest(),
                        'output_sha256':hashlib.sha256(bytes(7200)).hexdigest(),'transform':mapping,
                        'completed_interval_seconds':.1 if kwargs['upload_mode']=='overlap' else .2})
                return {'status':'completed','results':rows,'recorded_frames':args.max_frames,
                        'upload_mode':kwargs['upload_mode']}
            args=argparse.Namespace(source=Path('video'),reset_dir='',pbcopy='',pbload='',
                                   infer_timeout=10,tool_timeout=2)
            with patch.object(app,'ROOT',root),patch.object(app,'samples',return_value=frames), \
                 patch.object(app,'baseline_image',return_value=(Path('old.bin'),'hash')), \
                 patch.object(app,'load'),patch.object(app,'baseline_video',side_effect=baseline), \
                 patch.object(app,'pipeline',side_effect=pipeline),patch.object(app,'verify_movie',side_effect=lambda v,p,n,s:n), \
                 patch('pcie_stream.StreamBackend',Board),patch('video_golden.VideoGolden',Golden):
                report=app.run(args,output)
                args.resume=output; args.upload_mode='serial'
                continued=root/'continued'; continued.mkdir()
                with patch.object(app,'checked_selection',return_value=(report,{'path':str(output),'sha256':'old'})), \
                     patch.object(app,'baseline_image',side_effect=AssertionError('Must not extract baseline')), \
                     patch.object(app,'benchmark',side_effect=AssertionError('Must not rerun candidate')), \
                     patch.object(app,'load') as reload:
                    resumed=app.run(args,continued)
                self.assertEqual(resumed['status'],'passed',resumed.get('error'))
                self.assertEqual(resumed['upload_mode'],'serial')
                self.assertEqual(len(resumed['frames']),30)
                reload.assert_called_once_with(args)
                installed=root/'yolo26_pc/outputs/optimization/selected.json'
                accepted_bytes=installed.read_bytes()
                failed=root/'failed'; failed.mkdir()
                failure={'stage':'input_header_readback','frame_id':1,'active_frame_id':0,'slot':2,
                         'expected_hex':'11'*64,'actual_hex':'00'*64}
                child={'status':'failed','results':[],'recorded_frames':0,'submitted_frames':[0],
                       'failure':failure,'error':'Input request readback mismatch frame_id=1 slot=2'}
                args.upload_mode='overlap'
                with patch.object(app,'checked_selection',return_value=(report,{})), \
                     patch.object(app,'pipeline',return_value=child):
                    broken=app.run(args,failed)
                self.assertEqual(broken['status'],'failed')
                self.assertIn('Input request readback mismatch',broken['error'])
                self.assertEqual(broken['failure'],failure)
                self.assertEqual(broken['completed_frames'],0)
                self.assertEqual(installed.read_bytes(),accepted_bytes)
            self.assertEqual(report['status'],'passed',report.get('error'))
            self.assertEqual(len(report['frames']),30)
            selected=json.loads((root/'yolo26_pc/outputs/optimization/selected.json').read_text())
            self.assertEqual(selected['status'],'passed')
            self.assertEqual(selected['upload_mode'],'serial')
            self.assertFalse(list(output.glob('baseline-*')))
            self.assertFalse(list((output/'video').glob('frame-*')))
            self.assertTrue(all(len(c['rounds'])==2 for c in report['candidates']))


if __name__=='__main__': unittest.main()
