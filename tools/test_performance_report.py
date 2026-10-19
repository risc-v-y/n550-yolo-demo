"""Cycle wire integrity, accounting and offline compatibility; no hardware."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import performance_report as app
from board_profile import decode_profile, validate_saved_profile, PROFILE_MAGIC, PROFILE_BYTES


def payload(frame=3, session=7, options=0):
    header = [PROFILE_MAGIC, 1, PROFILE_BYTES, frame, 334, 334, options, 1,
              5000, 4500, 400, 7000, 800, 600, 3340, session]
    raw = struct.pack('<16Q', *header)+struct.pack('<3Q', 2, 7, 1)*334
    return raw+struct.pack('<Q', zlib.crc32(raw))


class ProfileTests(unittest.TestCase):
    def test_integrity_identity_and_accounting(self):
        raw = payload()
        profile = decode_profile(raw, 3, 7)
        self.assertEqual(len(profile['nodes_detail']), 334)
        self.assertEqual(profile['graph_overhead_cycles'], 1160)
        self.assertEqual(validate_saved_profile(profile, 3), profile)
        for damaged in (raw[:-1], raw[:100]+bytes([raw[100]^1])+raw[101:]):
            with self.assertRaises(ValueError): decode_profile(damaged, 3, 7)
        for frame, session in ((4, 7), (3, 9)):
            with self.assertRaisesRegex(ValueError, 'frame/session'): decode_profile(raw, frame, session)
        for index, value in ((5, 333), (14, 3341), (9, 1), (11, 1)):
            changed = bytearray(raw)
            struct.pack_into('<Q', changed, index*8, value)
            struct.pack_into('<Q', changed, len(changed)-8, zlib.crc32(changed[:-8]))
            with self.assertRaises(ValueError): decode_profile(bytes(changed), 3, 7)
        profile['nodes_detail'][0]['cache'] += 1
        with self.assertRaisesRegex(ValueError, 'SHA256'): validate_saved_profile(profile, 3)

    def test_offline_new_legacy_corrupt_and_nonrepresentative_profiles(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            record = dict(frame_id=0, prepare_seconds=.01, elapsed_seconds=10,
                          pcie_timing=dict(pcie_frame=3, run_roundtrip_seconds=9,
                                           board_profile=decode_profile(payload(), 3, 7)))
            report = dict(id='test-run', results=[record])
            (directory/'diagnostics.jsonl').write_text(json.dumps(dict(
                type='pcie_tool', message='pbload: exit 0', elapsed_seconds=.5, returncode=0))+'\n{')
            def run():
                (directory/'report.json').write_text(json.dumps(report), encoding='utf-8')
                return app.analyze(directory, 50)
            result = run()
            self.assertEqual(result['profiled'], 1)
            self.assertEqual(result['model_seconds'], .0001)
            self.assertEqual(len(result['nodes']), 334)
            self.assertEqual(result['tools']['pbload'], dict(calls=1, seconds=.5, failed=0))
            for page in (1, 2, 3):
                text = app.render(result, page)
                self.assertLessEqual(len(text.splitlines()), 22)
                self.assertTrue(all(len(line) <= 80 for line in text.splitlines()))
            record['pcie_timing']['board_profile'] = decode_profile(payload(options=8), 3, 7)
            self.assertEqual(run()['excluded'], [0])
            record['pcie_timing']['board_profile']['model_cycles'] += 1
            self.assertEqual(len(run()['errors']), 1)
            del record['pcie_timing']['board_profile']
            self.assertEqual(run()['profiled'], 0)
            self.assertEqual(run()['timings']['run_roundtrip_seconds']['mean'], 9)


if __name__ == '__main__': unittest.main()
