"""Replay saved video evidence without loading or communicating with a board."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import host_video as app
import video_demo as video
import video_report as review
import video_golden


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = video.np.zeros((48, 64, 3), dtype=video.np.uint8)
        movie = app.writer(video, self.root / 'demo.mp4', 5, image)
        self.records = []
        self.expected = {}
        for frame_id in range(3):
            source = image + frame_id * 70
            tensor, transform = video.prepare_input(source)
            candidates = video.np.zeros((300, 6), dtype=video.np.float32)
            # Input-space boxes beyond 416 are legal before inverse mapping/clipping.
            if frame_id:
                candidates[0] = [-10, -20, 500, 600, .8, 0]
            boxes = video.postprocess(candidates, transform, .25)
            drawn = video.annotate(source, boxes)
            for name, pixels in (('source', source), ('detected', drawn)):
                app.save_png(video, self.root / '{}-{:06d}.png'.format(name, frame_id), pixels)
            raw = candidates.tobytes()
            self.expected[frame_id] = raw
            (self.root / 'candidates-{:06d}.bin'.format(frame_id)).write_bytes(raw)
            record = {'frame_id': frame_id, 'source_frame': frame_id,
                      'input_sha256': hashlib.sha256(tensor.tobytes()).hexdigest(),
                      'output_sha256': hashlib.sha256(raw).hexdigest(), 'transform': transform,
                      'detections': boxes.tolist(), 'pcie_timing': {'run_roundtrip_seconds': 2}}
            review.save_json(self.root / 'result-{:06d}.json'.format(frame_id), record)
            self.records.append(record)
            movie.write(drawn)
        movie.release()
        self.report = {'status': 'completed', 'results': self.records, 'encoded_frames': 3,
                       'source': '/old/location/input.mp4', 'completion_reason': 'end of video'}
        self.save_report()
        self.golden_calls = []
        owner = self
        class FixtureGolden:
            identity = {'fixture': 'independent expected bytes'}
            def check(self, frame_id, data, actual):
                owner.golden_calls.append(frame_id)
                return dict(video_golden.compare(actual, owner.expected[frame_id]), frame_id=frame_id)
            def close(self): pass
        self.fake_golden = FixtureGolden()
        context = patch('video_golden.VideoGolden', return_value=self.fake_golden)
        context.start(); self.addCleanup(context.stop)

    def save_report(self):
        review.save_json(self.root / 'report.json', self.report)

    def analyze(self):
        with patch('pcie_backend.PCIeBackend', side_effect=AssertionError('offline opened board')):
            return review.analyze(self.root, video)

    def test_old_three_frames_replay_without_source_video_or_new_inference(self):
        before = (self.root / 'report.json').read_bytes()
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'PASSED')
        self.assertEqual(summary['numeric']['status'], 'PASSED')
        self.assertEqual(summary['numeric']['exact_frames'], 3)
        self.assertEqual(summary['overall'], 'PASSED')
        self.assertEqual(summary['empty_detection_frames'], 1)
        self.assertEqual(summary['semantic_review'], 'NOT_CONFIRMED')
        self.assertEqual(summary['encoded'], 3)
        self.assertIn('Legacy', summary['warnings'][0])
        self.assertEqual((self.root / 'report.json').read_bytes(), before)
        self.assertIsNotNone(video.read_image(self.root / 'preview.png'))

    def test_mark_saved_frame_is_bound_and_does_not_change_original_report(self):
        before = (self.root / 'report.json').read_bytes()
        review.flag_frame(self.root, 1, 'wrong box at street corner')
        review.flag_frame(self.root, 1, 'corrected note')
        with self.assertRaises(ValueError):
            review.flag_frame(self.root, 3)
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'PASSED')
        self.assertEqual(summary['semantic_review'], 'FLAGGED')
        self.assertEqual(len(summary['flags']), 1)
        self.assertEqual(summary['flags'][0]['note'], 'corrected note')
        self.assertEqual((self.root / 'report.json').read_bytes(), before)

    def test_single_byte_raw_corruption_and_missing_images_are_errors(self):
        path = self.root / 'candidates-000001.bin'
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
        (self.root / 'detected-000002.png').unlink()
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'FAILED')
        self.assertEqual([i['frame'] for i in summary['issues'] if i['code'] == 'frame_evidence'], [1, 2])
        self.assertIn('SHA256', summary['issues'][0]['detail'])

    def test_empty_png_is_collected_as_issue_instead_of_aborting_report(self):
        (self.root / 'detected-000001.png').write_bytes(b'')
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'FAILED')
        self.assertEqual(summary['issues'][0]['frame'], 1)
        self.assertTrue((self.root / 'preview.png').is_file())

    def test_new_png_hashes_detect_corruption_and_boot_failure_uses_log_snapshot(self):
        self.records[0]['detected_png_sha256'] = '0' * 64
        review.save_json(self.root / 'result-000000.json', self.records[0])
        self.report.update(status='failed', error='startup selftest failed')
        self.save_report()
        board = {'stage_name': 'selftest', 'node': None, 'error': 4}
        (self.root / 'diagnostics.jsonl').write_text(json.dumps({'type':'board','diagnostic':board}) + '\n')
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'FAILED')
        self.assertIn('PNG SHA256', summary['issues'][0]['detail'])
        self.assertEqual(summary['board_diagnostic'], board)

    def test_input_hash_mapping_drawing_and_invalid_candidate_checks(self):
        scenarios = ('input', 'mapping', 'drawing', 'candidate')
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                record = self.records[0]
                saved = json.loads(json.dumps(record))
                image_path = self.root / 'detected-000000.png'
                pixels = image_path.read_bytes()
                raw_path = self.root / 'candidates-000000.bin'
                raw = raw_path.read_bytes()
                if scenario == 'input': record['input_sha256'] = '0' * 64
                elif scenario == 'mapping': record['transform']['scale'] += 1
                elif scenario == 'drawing': app.save_png(video, image_path, video.np.ones((48, 64, 3), video.np.uint8))
                else:
                    broken = video.np.frombuffer(raw, dtype='<f4').copy(); broken[0] = float('nan')
                    raw_path.write_bytes(broken.tobytes()); record['output_sha256'] = review.digest(raw_path)
                review.save_json(self.root / 'result-000000.json', record); self.save_report()
                self.assertEqual(self.analyze()['functional'], 'FAILED')
                self.records[0] = saved
                review.save_json(self.root / 'result-000000.json', saved)
                image_path.write_bytes(pixels); raw_path.write_bytes(raw)
        self.save_report()

    def test_sequence_movie_count_and_incomplete_run_never_pass(self):
        self.report.update(status='stopped', active_frame=3, stage='inference')
        self.save_report()
        self.assertEqual(self.analyze()['functional'], 'INCOMPLETE')
        self.report.update(status='completed', requested_frames=30, completion_reason='requested frame limit')
        self.records[1]['source_frame'] = 0
        review.save_json(self.root / 'result-000001.json', self.records[1]); self.save_report()
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'FAILED')
        self.assertTrue({'sequence', 'frame_limit'} <= {i['code'] for i in summary['issues']})
        self.report['encoded_frames'] = 2; self.save_report()
        self.assertIn('movie', [i['code'] for i in self.analyze()['issues']])

    def test_runtime_failure_and_page_bounds_keep_detailed_error_on_host(self):
        self.report.update(status='failed', error='Board trap ' + 'x' * 2000, active_frame=3, stage='inference')
        self.report['last_board_status'] = {'state': 4, 'frame': 4, 'error': 3}
        self.report['last_board_diagnostic'] = {'stage_name': 'trap', 'node': 96, 'op_name': 'conv',
                                               'phase_name': 'compute', 'trap_pc': 0x80001234}
        self.save_report()
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'FAILED')
        self.assertIn('x' * 2000, summary['issues'][0]['detail'])
        for page in (1, 2, 3):
            text = review.render(summary, page)
            self.assertLessEqual(len(text.splitlines()), 22)
            self.assertTrue(all(len(line) <= 80 for line in text.splitlines()))
            self.assertIn('PAGE={}/3'.format(page), text)
        self.assertIn('node=96', review.render(summary, 2))
        self.assertIn('pc=0x80001234', review.render(summary, 2))

    def test_chinese_notes_fit_terminal_columns(self):
        import unicodedata
        review.flag_frame(self.root, 0, '\u9519\u6846' * 80)
        summary = self.analyze()
        for page in (1,2,3):
            text = review.render(summary, page)
            for line in text.splitlines():
                columns = sum(2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in line)
                self.assertLessEqual(columns, 80)

    def test_single_bit_numerical_difference_is_not_hidden_by_functional_pass(self):
        path = self.root / 'candidates-000000.bin'
        raw = bytearray(path.read_bytes()); raw[3] = 0x80  # -0 instead of +0: same drawn box.
        path.write_bytes(raw)
        self.records[0]['output_sha256'] = review.digest(path)
        review.save_json(self.root / 'result-000000.json', self.records[0]); self.save_report()
        summary = self.analyze()
        self.assertEqual(summary['functional'], 'PASSED')
        self.assertEqual(summary['numeric']['status'], 'DIFFERENT')
        self.assertEqual(summary['overall'], 'FAILED')
        self.assertEqual(summary['numeric']['exact_frames'], 2)
        self.assertEqual(self.golden_calls, [0, 1, 2])
        text = review.render(summary, 2)
        self.assertIn('got=0x80000000', text)
        self.assertIn('golden=0x00000000', text)

    def test_reference_error_blocks_numeric_pass_without_board_access(self):
        with patch('video_golden.VideoGolden', side_effect=ValueError('reference SHA mismatch')):
            summary = self.analyze()
        self.assertEqual(summary['functional'], 'PASSED')
        self.assertEqual(summary['numeric']['status'], 'ERROR')
        self.assertEqual(summary['overall'], 'FAILED')
        self.assertIn('reference SHA', review.render(summary, 2))

    def test_interrupted_numeric_check_saves_incomplete_report(self):
        original = self.fake_golden.check
        def interrupt(frame_id, data, actual):
            if frame_id == 1: raise KeyboardInterrupt()
            return original(frame_id, data, actual)
        with patch.object(self.fake_golden, 'check', side_effect=interrupt):
            summary = self.analyze()
        self.assertEqual(summary['functional'], 'PASSED')
        self.assertEqual(summary['numeric']['exact_frames'], 1)
        self.assertEqual(summary['numeric']['status'], 'INCOMPLETE')
        self.assertEqual(summary['overall'], 'INCOMPLETE')
        self.assertTrue(summary['numeric']['interrupted'])
        # Reopening the report performs software checking only, with no board rerun.
        self.assertEqual(self.analyze()['numeric']['status'], 'PASSED')

    def test_bad_frame_evidence_prevents_full_numeric_pass_but_keeps_other_frames(self):
        (self.root / 'source-000001.png').unlink()
        summary = self.analyze()
        self.assertEqual(summary['numeric']['status'], 'INCOMPLETE')
        self.assertEqual(self.golden_calls, [0, 2])
        self.assertEqual(summary['overall'], 'FAILED')

    def test_one_frame_reference_error_continues_remaining_frames(self):
        original = self.fake_golden.check
        def failure(frame_id, data, actual):
            if frame_id == 1: raise TimeoutError('reference process timeout')
            return original(frame_id, data, actual)
        with patch.object(self.fake_golden, 'check', side_effect=failure):
            summary = self.analyze()
        self.assertEqual(self.golden_calls, [0, 2])
        self.assertEqual(summary['numeric']['exact_frames'], 2)
        self.assertEqual(summary['numeric']['error_count'], 1)
        self.assertEqual(summary['overall'], 'FAILED')

    def test_reference_cleanup_failure_does_not_produce_false_pass(self):
        with patch.object(self.fake_golden, 'close', side_effect=RuntimeError('cannot close reference')):
            summary = self.analyze()
        self.assertEqual(summary['numeric']['exact_frames'], 3)
        self.assertEqual(summary['numeric']['status'], 'ERROR')
        self.assertEqual(summary['overall'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
