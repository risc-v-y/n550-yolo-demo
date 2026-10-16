"""Real video codecs/preprocessing with a fake inference transport; no board."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import host_video as app
import video_demo as video


class VideoTests(unittest.TestCase):
    def test_isolated_process_uses_project_video_libraries(self):
        with tempfile.TemporaryDirectory() as name:
            libraries = Path(name)
            (libraries / 'cv2.py').write_text('PROBE="offline-project-library"\n')
            (libraries / 'video_demo.py').write_text('from cv2 import PROBE\n')
            code = ('import runpy,pathlib; m=runpy.run_path({!r},run_name="video_startup"); '
                    'f=m["dependencies"]; f.__globals__["VIDEO_LIBS"]=pathlib.Path({!r}); '
                    'assert f().PROBE=="offline-project-library"').format(app.__file__, name)
            child = subprocess.run([sys.executable, '-I', '-c', code], cwd=name,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_offline_pip_starts_directly_from_its_wheel(self):
        wheel = app.ROOT / 'host_deps/py38-linux-x86_64/pip-24.1.2-py3-none-any.whl'
        code = ('import sys,runpy; sys.path.insert(0,{!r}); '
                'import pip; assert ".whl" in pip.__file__; '
                'sys.argv=["pip","--version"]; runpy.run_module("pip",run_name="__main__")').format(str(wheel))
        child = subprocess.run([sys.executable, '-I', '-c', code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn('pip 24.1.2', child.stdout)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.mp4'
        frame = video.np.zeros((48, 64, 3), dtype=video.np.uint8)
        out = app.writer(video, self.source, 5, frame)
        for color in (30, 100, 220):
            out.write(frame + color)
        out.release()
        self.output = self.root / 'output'
        self.output.mkdir()
        self.args = argparse.Namespace(source=self.source, check=False, no_display=True, load=False,
                                       playback_fps=5, max_frames=0, pbcopy='unused', pbload='unused',
                                       infer_timeout=1, tool_timeout=1)
        self.hashes = []
        self.closed = False
        self.backend_calls = 0
        self.fail_frame = None
        owner = self
        class FakeBackend:
            info = {'fake': True}
            last_timing = {'fake': True}
            def __init__(self, *args, **kwargs):
                owner.backend_calls += 1
            def infer(self, tensor):
                self_id = len(owner.hashes)
                if self_id == owner.fail_frame:
                    raise RuntimeError('injected inference failure')
                time.sleep(.025)
                owner.hashes.append(hashlib.sha256(tensor.tobytes()).hexdigest())
                return video.np.zeros((300, 6), dtype=video.np.float32)
            def close(self):
                owner.closed = True
        self.fake = FakeBackend

    def run_video(self):
        with patch('pcie_backend.PCIeBackend', self.fake), \
             patch.object(app, 'check_host', return_value={'source_fps': 5}):
            return app.run(self.args, self.output)

    def test_sequential_eof_no_skip_repeat_and_movie_matches_result_count(self):
        report = self.run_video()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['completion_reason'], 'end of video')
        self.assertEqual([r['source_frame'] for r in report['results']], [0, 1, 2])
        self.assertEqual(report['encoded_frames'], 3)
        self.assertEqual(self.backend_calls, 1)
        self.assertTrue(self.closed)
        self.assertEqual(report['functional'], 'PASSED')
        self.assertTrue((self.output / 'preview.png').is_file())
        cap = video.cv2.VideoCapture(str(self.source))
        for i, record in enumerate(report['results']):
            ok, original = cap.read()
            self.assertTrue(ok)
            tensor, _ = video.prepare_input(original)
            self.assertEqual(self.hashes[i], hashlib.sha256(tensor.tobytes()).hexdigest())
            self.assertEqual(record['input_sha256'], self.hashes[i])
            self.assertTrue(video.np.array_equal(original, video.read_image(self.output / 'source-{:06d}.png'.format(i))))
            self.assertTrue(video.np.array_equal(original, video.read_image(self.output / 'detected-{:06d}.png'.format(i))))
            self.assertEqual((self.output / 'candidates-{:06d}.bin'.format(i)).stat().st_size, 7200)
        cap.release()

    def test_result_limit_and_failure_keep_already_completed_frames(self):
        self.args.max_frames = 1
        report = self.run_video()
        self.assertEqual(report['encoded_frames'], 1)
        self.assertEqual(report['completion_reason'], 'requested frame limit')
        self.output = self.root / 'failed'; self.output.mkdir()
        self.hashes.clear(); self.args.max_frames = 0; self.fail_frame = 1
        report = self.run_video()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['encoded_frames'], 1)
        self.assertIn('injected', report['error'])
        self.assertEqual(report['active_frame'], 1)
        self.assertEqual(report['stage'], 'inference_pcie')
        self.assertEqual(report['functional'], 'FAILED')
        self.assertTrue((self.output / 'detected-000000.png').is_file())
        self.assertFalse((self.output / 'detected-000001.png').exists())

    def test_every_completed_frame_reaches_display_and_wait_loop_pumps_gui(self):
        self.args.no_display = False
        with patch.object(video.cv2, 'namedWindow'), patch.object(video.cv2, 'setWindowTitle'), \
             patch.object(video.cv2, 'imshow') as show, patch.object(video.cv2, 'waitKey', return_value=-1) as wait, \
             patch.object(video.cv2, 'getWindowProperty', return_value=1), patch.object(video.cv2, 'destroyAllWindows'):
            report = self.run_video()
        self.assertEqual(report['encoded_frames'], 3)
        self.assertGreater(wait.call_count, 3)
        displayed = [call.args[1] for call in show.call_args_list if call.args[1].shape == (48, 64, 3)]
        for i in range(3):
            expected = video.read_image(self.output / 'detected-{:06d}.png'.format(i))
            self.assertTrue(any(video.np.array_equal(image, expected) for image in displayed))

    def test_keyboard_interrupt_reports_stopped_and_closes_transport(self):
        with patch.object(self.fake, 'infer', side_effect=KeyboardInterrupt):
            report = self.run_video()
        self.assertEqual(report['status'], 'stopped')
        self.assertEqual(report['results'], [])
        self.assertTrue(self.closed)

    def test_b_key_marks_visible_frame_not_inflight_frame(self):
        self.args.no_display = False
        marked = []
        def key(delay):
            # During frame 1 inference, the visible image is completed frame 0.
            if (delay == 20 and len(self.hashes) == 1 and not marked and
                    (self.output / 'result-000000.json').is_file()):
                marked.append(True)
                return ord('b')
            return -1
        with patch.object(video.cv2, 'namedWindow'), patch.object(video.cv2, 'setWindowTitle'), \
             patch.object(video.cv2, 'imshow') as show, patch.object(video.cv2, 'waitKey', side_effect=key), \
             patch.object(video.cv2, 'getWindowProperty', return_value=1), patch.object(video.cv2, 'destroyAllWindows'):
            report = self.run_video()
        flags = json.loads((self.output / 'flags.json').read_text())['frames']
        self.assertEqual([f['frame_id'] for f in flags], [0])
        self.assertEqual(report['functional'], 'PASSED')
        self.assertEqual(show.call_count, 4)  # Placeholder + one update per completed frame.

    def test_q_key_stops_and_preserves_report_instead_of_passing_frame_limit(self):
        self.args.no_display = False
        self.args.max_frames = 1
        with patch.object(video.cv2, 'namedWindow'), patch.object(video.cv2, 'setWindowTitle'), \
             patch.object(video.cv2, 'imshow'), \
             patch.object(video.cv2, 'waitKey', side_effect=lambda delay: ord('q') if delay == 30 else -1), \
             patch.object(video.cv2, 'getWindowProperty', return_value=1), patch.object(video.cv2, 'destroyAllWindows'):
            report = self.run_video()
        self.assertEqual(report['status'], 'stopped')
        self.assertEqual(report['encoded_frames'], 1)
        self.assertEqual(report['functional'], 'INCOMPLETE')

    def test_thirty_consecutive_frames_one_session_and_decodable_mp4(self):
        image = video.np.zeros((48, 64, 3), dtype=video.np.uint8)
        out = app.writer(video, self.source, 5, image)
        for frame_id in range(35):
            out.write(image + frame_id * 6)
        out.release()
        self.args.max_frames = 30
        report = self.run_video()
        self.assertEqual(report['functional'], 'PASSED')
        self.assertEqual(report['encoded_frames'], 30)
        self.assertEqual([r['frame_id'] for r in report['results']], list(range(30)))
        self.assertEqual(self.backend_calls, 1)

    def test_timeout_and_invalid_output_keep_failure_context_and_evidence(self):
        with patch.object(self.fake, 'infer', side_effect=TimeoutError('PCIe timeout')):
            report = self.run_video()
        self.assertEqual(report['functional'], 'FAILED')
        self.assertEqual(report['active_frame'], 0)
        self.assertEqual(report['stage'], 'inference_pcie')
        self.assertTrue((self.output / 'diagnostics.jsonl').is_file())
        self.output = self.root / 'invalid'; self.output.mkdir()
        bad = video.np.zeros((300, 6), video.np.float32); bad[0, 5] = 80
        with patch.object(self.fake, 'infer', return_value=bad):
            report = self.run_video()
        self.assertEqual(report['stage'], 'validate_postprocess')
        self.assertEqual(report['functional'], 'FAILED')
        self.assertEqual((self.output / 'candidates-000000.bin').read_bytes(), bad.tobytes())
        self.assertEqual(report['results'], [])

    def test_real_host_check_never_constructs_board_transport(self):
        with patch('pcie_backend.PCIeBackend', side_effect=AssertionError('board opened')):
            report = app.check_host(video, self.source, self.output)
        self.assertTrue(report['bus_preprocess_exact'])
        self.assertEqual(report['mp4_frames'], 3)
        self.assertGreater(report['bus_boxes'], 0)
        self.assertFalse(report['display_tested'])

    def test_display_probe_inherits_project_library_path(self):
        libraries = self.root / 'libraries'; libraries.mkdir()
        with patch.object(app, 'VIDEO_LIBS', libraries), \
             patch.dict('os.environ', {'DISPLAY': ':test', 'PYTHONPATH': '/existing'}), \
             patch.object(app.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as child:
            report = app.check_host(video, self.source, self.output, display=True)
        self.assertTrue(report['display_tested'])
        expected = str(libraries) + __import__('os').pathsep + '/existing'
        self.assertEqual(child.call_args.kwargs['env']['PYTHONPATH'], expected)

    def test_host_failure_prevents_board_access_and_saves_error(self):
        with patch.object(app, 'check_host', side_effect=ValueError('host codec unavailable')), \
             patch('pcie_backend.PCIeBackend') as board:
            report = app.run(self.args, self.output)
        board.assert_not_called()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('codec', json.loads((self.output / 'report.json').read_text())['error'])

    def test_missing_dependency_still_emits_failure_marker_and_never_opens_board(self):
        import io
        with patch.object(app, 'dependencies', side_effect=RuntimeError('missing cv2')), \
             patch('pcie_backend.PCIeBackend') as board, patch('sys.stdout', new_callable=io.StringIO) as terminal:
            report = app.run(self.args, self.output)
        board.assert_not_called()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('VIDEO CHECK FAILED', terminal.getvalue())
        self.assertIn('missing cv2', (self.output / 'report.json').read_text())

    def test_load_runs_once_after_host_check_and_failure_never_starts_inference(self):
        self.args.load = True
        self.args.reset_dir = '/platform'
        with patch.object(app.subprocess, 'run') as load:
            report = self.run_video()
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0][-4:], ['--reset-dir', '/platform', '--pbcopy', 'unused'])
        self.assertEqual(report['encoded_frames'], 3)
        self.output = self.root / 'load-failed'; self.output.mkdir()
        with patch.object(app.subprocess, 'run', side_effect=RuntimeError('load failed')):
            report = self.run_video()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(report['results']), 0)
        self.assertEqual(self.backend_calls, 1)

    def test_candidate_values_and_empty_detections(self):
        np = video.np
        good = np.zeros((300, 6), dtype=np.float32)
        app.validate_candidates(video, good)
        for index, value in ((0, float('nan')), (0, 2), (4, 1.1), (5, .5), (5, 80)):
            bad = good.copy(); bad[0, index] = value
            with self.assertRaises(ValueError): app.validate_candidates(video, bad)
        with self.assertRaises(ValueError): app.validate_candidates(video, good[:299])


if __name__ == '__main__':
    unittest.main()
