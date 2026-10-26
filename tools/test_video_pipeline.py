import argparse
import hashlib
import io
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
        self.stream={'test':True}; self.active=None; self.pending=None; self.frame=0; self.uploads=[]
    def configure(self,config): pass
    def upload(self,raw,frame):
        assert len(raw)==519168 and frame==self.frame+1 and self.pending is None
        self.uploads.append((frame,self.active))
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
    def test_performance_uses_quiet_backend_and_defers_evidence_until_finish(self):
        for retain in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                args,output=self.fixture(Path(tmp)); args.mode='performance'
                states=[]
                class Fast(FakeBoard):
                    def __init__(self,*a,**kw):
                        self_kwargs=kw.copy(); states.append(self_kwargs)
                        super().__init__()
                    def finish(self):
                        self_outer.assertEqual(list(output.glob('frame-*')),[])
                        raw=super().finish()
                        self.last_timing={'cycles':None,'frame_seconds':.02,'profile_read_seconds':None}
                        return raw
                self_outer=self
                with patch.object(app,'ROOT',Path(tmp)),patch.object(app,'saved_configuration',return_value={'selected':{},'upload_mode':'serial'}), \
                     patch('sys.stdout',new_callable=io.StringIO) as text:
                    (Path(tmp)/'firmware').mkdir(); (Path(tmp)/'firmware/demo-fast.bin').write_bytes(b'fast')
                    result=app.run(args,output,backend_factory=Fast,selected={},retain=retain)
                self.assertEqual(result['status'],'completed',result.get('error'))
                self.assertEqual((states[0]['mode'],states[0]['profile'],states[0]['diagnostic']),('performance',False,None))
                self.assertNotIn('VIDEO frame=',text.getvalue())
                self.assertEqual(len(list(output.glob('frame-*-input.bin'))),5 if retain else 0)
                for row in result['results']:
                    self.assertIsNone(row['cycles'])
                    self.assertEqual('input_sha256' in row,retain)

    def test_gui_failure_stops_before_loading_or_opening_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,output=self.fixture(Path(tmp)); args.mode='performance'; args.load=True; args.no_display=False
            with patch.object(app,'check_display',side_effect=RuntimeError('broken GUI')), \
                 patch.object(app.subprocess,'run') as load,patch.object(app,'saved_configuration') as config:
                result=app.run(args,output,backend_factory=FakeBoard)
            self.assertEqual(result['status'],'failed')
            load.assert_not_called(); config.assert_not_called()

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

    def test_serial_and_overlap_have_the_same_order_but_distinct_upload_windows(self):
        for mode in ('serial','overlap'):
            boards=[]
            def factory(*args,**kwargs):
                board=FakeBoard(); boards.append(board); return board
            with tempfile.TemporaryDirectory() as tmp:
                args,output=self.fixture(Path(tmp))
                with patch.object(app,'check_host',return_value={'source_fps':5}):
                    result=app.run(args,output,backend_factory=factory,selected={},upload_mode=mode)
                self.assertEqual(result['status'],'completed',result.get('error'))
                self.assertEqual(result['upload_mode'],mode)
                self.assertEqual(result['submitted_frames'],list(range(5)))
                self.assertEqual(result['inference_completed_frames'],5)
                expected=[(i+1,i if mode=='overlap' and i else None) for i in range(5)]
                self.assertEqual(boards[0].uploads,expected)

    def test_pending_input_failure_is_not_misattributed_to_active_frame(self):
        from pcie_stream import RequestReadbackError
        class Broken(FakeBoard):
            def upload(self,raw,frame):
                if self.active:
                    raise RequestReadbackError({'frame_id':frame-1,'active_frame_id':self.active-1,
                        'slot':2,'expected_hex':'11'*64,'actual_hex':'00'*64,'different_offsets':list(range(64))})
                super().upload(raw,frame)
        with tempfile.TemporaryDirectory() as tmp:
            args,output=self.fixture(Path(tmp))
            with patch.object(app,'check_host',return_value={'source_fps':5}):
                result=app.run(args,output,backend_factory=Broken,selected={},upload_mode='overlap')
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['submitted_frames'],[0])
            self.assertEqual((result['failure']['frame_id'],result['failure']['active_frame_id']),(1,0))
            self.assertEqual(result['inference_completed_frames'],0)
            self.assertTrue((output/'request-readback.json').is_file())
            import json
            self.assertEqual(json.loads((output/'failure-frame.json').read_text())['frame_id'],1)


if __name__=='__main__': unittest.main()
