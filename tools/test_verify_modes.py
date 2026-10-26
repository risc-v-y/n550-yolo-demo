"""Both release images use the same inputs; failures never promote a config."""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verify_modes as app
import video_pipeline
import host_video
from optimization_report import report_pages


class Golden:
    def __init__(self,*args): pass
    def predict(self,data): return bytes(7200), []
    def close(self): pass


class ModesTests(unittest.TestCase):
    def execute(self,root,failure=None):
        video=host_video.dependencies()
        (root/'firmware').mkdir()
        for name in ('demo-fast.bin','demo-pcie.bin'): (root/'firmware'/name).write_bytes(name.encode())
        source=root/'source.mp4'; image=video.np.zeros((16,16,3),dtype=video.np.uint8)
        movie=host_video.writer(video,source,25,image)
        for i in range(30): image.fill(i); movie.write(image)
        movie.release()
        args=argparse.Namespace(source=source,no_display=True,playback_fps=0,reset_dir='',pbcopy='',pbload='',
                                infer_timeout=2,tool_timeout=1)
        calls=[]
        class Board:
            def __init__(self,*args,**kw):
                self.mode=kw['mode']; self.stream={}; self.last_profile=None; self.frame=0
            def configure(self,cfg):
                if cfg!=app.CONFIG: raise AssertionError(cfg)
            def upload(self,raw,frame): self.frame=frame
            def begin(self): pass
            def close(self): pass
            def finish(self):
                if failure==self.mode: raise RuntimeError('injected write failure')
                self.last_timing={'cycles':None,'frame_seconds':.02,'profile_read_seconds':None}
                if self.mode=='diagnostic':
                    self.last_profile={'nodes_detail':[],'phases':{}}
                    self.last_timing['cycles']={'model_cycles':100}
                return bytes(7200)
        def pipeline(options,directory,**kwargs):
            calls.append((options.mode,options.max_frames,options.load,kwargs))
            options.load=False  # Only the actual transport is replaced in this fixture.
            return video_pipeline.run(options,directory,backend_factory=Board,**kwargs)
        target=root/'yolo26_pc/outputs/optimization/selected.json'; target.parent.mkdir(parents=True)
        target.write_text('{"previous":"accepted"}')
        output=root/'acceptance'; output.mkdir()
        with patch.object(app,'ROOT',root),patch.object(video_pipeline,'ROOT',root), \
             patch.object(video_pipeline,'check_host',return_value={'source_fps':25}), \
             patch.object(video_pipeline,'saved_configuration',return_value={'selected':app.CONFIG,'upload_mode':'serial'}), \
             patch('sys.stdout',new_callable=io.StringIO):
            report=app.run(args,output,pipeline_fn=pipeline,golden_factory=Golden)
        return report,calls,target,output

    def test_two_images_exact_outputs_same_inputs_and_no_retuning(self):
        with tempfile.TemporaryDirectory() as name:
            report,calls,target,output=self.execute(Path(name))
            self.assertEqual(report['status'],'passed',report.get('error'))
            self.assertEqual([(a,b,c) for a,b,c,_ in calls],[('diagnostic',3,True),('performance',30,True)])
            self.assertTrue(all(c[3]['upload_mode']=='serial' for c in calls))
            selected=json.loads(target.read_text())
            self.assertEqual(selected['selected'],app.CONFIG)
            self.assertNotEqual(selected['firmware_sha256'],selected['diagnostic_sha256'])
            self.assertEqual(len(report['frames']),30)
            self.assertEqual(list(output.glob('*/frame-*')),[])
            self.assertTrue((output/'performance/demo.mp4').is_file())
            pages=report_pages(report)
            self.assertIn('NOT COLLECTED',pages[1])
            for page in pages:
                self.assertLessEqual(len(page.splitlines()),22)
                self.assertTrue(all(len(line)<=80 for line in page.splitlines()))

    def test_failed_mode_stops_and_preserves_configuration_and_evidence(self):
        for mode,count in (('diagnostic',1),('performance',2)):
            with tempfile.TemporaryDirectory() as name:
                report,calls,target,output=self.execute(Path(name),mode)
                self.assertEqual(report['status'],'failed')
                self.assertEqual(len(calls),count)
                self.assertEqual(json.loads(target.read_text()),{'previous':'accepted'})
                self.assertTrue((output/mode/'failure-input.bin').is_file())


if __name__=='__main__': unittest.main()
