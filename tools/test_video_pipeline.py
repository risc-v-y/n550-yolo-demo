import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import video_pipeline as app
import host_video
video=host_video.dependencies()


class FakeBoard:
    def __init__(self,*args,**kwargs):
        self.stream={'test':True}; self.active=None; self.pending=None; self.frame=0
    def configure(self,config): pass
    def upload(self,raw,frame):
        assert len(raw)==519168 and frame==self.frame+1 and self.pending is None
        self.pending=frame
    def begin(self):
        assert self.active is None and self.pending is not None
        self.active=self.pending; self.pending=None; self.frame=self.active
    def finish(self):
        time.sleep(.02)
        self.last_timing={'cycles':{'model_cycles':100,'service_cycles':200},'upload_seconds':.001,
                          'download_seconds':.001,'run_roundtrip_seconds':.02,'profile_read_seconds':None}
        self.active=None
        return bytes(7200)
    def close(self): pass


class PipelineTests(unittest.TestCase):
    def test_preprocessing_reuses_storage_and_matches_old_logical_input(self):
        prep=app.Preprocessor(video)
        for h,w in ((13,17),(481,640),(1080,1920),(416,416),(17,13)):
            image=video.np.arange(h*w*3,dtype=video.np.uint8).reshape(h,w,3)
            raw,mapping=prep.prepare(image)
            old,original=video.prepare_input(image)
            rebuilt=video.np.frombuffer(raw,dtype=video.np.uint8).astype('<f4')/video.np.float32(255)
            self.assertEqual(old.tobytes(),rebuilt.tobytes())
            for key in ('scale','padding_ltrb','original_hw'): self.assertEqual(mapping[key],original[key])

    def fixture(self,root):
        source=root/'source.mp4'
        image=video.np.zeros((48,64,3),dtype=video.np.uint8)
        movie=host_video.writer(video,source,5,image)
        for i in range(5): image.fill(i*30); movie.write(image)
        movie.release()
        args=argparse.Namespace(source=source,no_display=True,load=False,max_frames=5,playback_fps=0,
            reset_dir='',pbcopy='',pbload='',infer_timeout=2,tool_timeout=1)
        output=root/'run'; output.mkdir()
        return args,output

    def test_all_frames_recorded_in_order_without_per_frame_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,output=self.fixture(Path(tmp))
            with patch.object(app,'check_host',return_value={'source_fps':5}):
                result=app.run(args,output,backend_factory=FakeBoard,selected={})
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual([r['frame_id'] for r in result['results']],list(range(5)))
            self.assertEqual(result['recorded_frames'],5)
            self.assertEqual({p.name for p in output.iterdir()},{'demo.mp4','report.json'})
            self.assertEqual(host_video.verify_movie(video,output/'demo.mp4',5,(48,64,3)),5)

    def test_failure_saves_input_and_does_not_hang(self):
        class Broken(FakeBoard):
            def finish(self): raise RuntimeError('injected board failure')
        with tempfile.TemporaryDirectory() as tmp:
            args,output=self.fixture(Path(tmp))
            with patch.object(app,'check_host',return_value={'source_fps':5}):
                result=app.run(args,output,backend_factory=Broken,selected={})
            self.assertEqual(result['status'],'failed')
            self.assertTrue((output/'failure-input.bin').is_file())
            self.assertTrue((output/'failure-source.png').is_file())


if __name__=='__main__': unittest.main()
