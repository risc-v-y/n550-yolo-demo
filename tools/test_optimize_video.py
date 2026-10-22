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
                for i in range(30):
                    prefix='frame-{:06d}'.format(i)
                    (directory/(prefix+'-input.bin')).write_bytes(raw)
                    (directory/(prefix+'-output.bin')).write_bytes(bytes(7200))
                    video.cv2.imencode('.png',image)[1].tofile(str(directory/(prefix+'-source.png')))
                    rows.append({'frame_id':i,'input_sha256':hashlib.sha256(raw).hexdigest(),
                        'output_sha256':hashlib.sha256(bytes(7200)).hexdigest(),'transform':mapping,
                        'completed_interval_seconds':.1})
                return {'status':'completed','results':rows}
            args=argparse.Namespace(source=Path('video'),reset_dir='',pbcopy='',pbload='',
                                   infer_timeout=10,tool_timeout=2)
            with patch.object(app,'ROOT',root),patch.object(app,'samples',return_value=frames), \
                 patch.object(app,'baseline_image',return_value=(Path('old.bin'),'hash')), \
                 patch.object(app,'load'),patch.object(app,'baseline_video',side_effect=baseline), \
                 patch.object(app,'pipeline',side_effect=pipeline),patch.object(app,'verify_movie',return_value=30), \
                 patch('pcie_stream.StreamBackend',Board),patch('video_golden.VideoGolden',Golden):
                report=app.run(args,output)
            self.assertEqual(report['status'],'passed',report.get('error'))
            self.assertEqual(len(report['frames']),30)
            selected=json.loads((root/'yolo26_pc/outputs/optimization/selected.json').read_text())
            self.assertEqual(selected['status'],'passed')
            self.assertFalse(list(output.glob('baseline-*')))
            self.assertFalse(list((output/'video').glob('frame-*')))
            self.assertTrue(all(len(c['rounds'])==2 for c in report['candidates']))


if __name__=='__main__': unittest.main()
