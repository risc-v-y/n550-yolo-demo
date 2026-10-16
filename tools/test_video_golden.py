"""Independent graph propagation, strict bit comparison and bound golden caches."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import video_golden as app


def candidates(value):
    return struct.pack('<6f', 0, 0, value, value, .5, 1) * 300


class GoldenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('model', 'reference/nodes', 'firmware', 'out'):
            (self.root / name).mkdir(parents=True)
        tensors = [{'id': i, 'constant': int(i == 1), 'offset': 0, 'bytes': 7200 if i == 4 else 4}
                   for i in range(5)]
        self.manifest = {'tensors': tensors, 'nodes': [
            {'id': 0, 'kind': 'ADD', 'inputs': [0, 1], 'outputs': [2]},
            {'id': 1, 'kind': 'COPY', 'inputs': [2], 'outputs': [3]},
            {'id': 2, 'kind': 'CONCAT', 'inputs': [2, 3], 'outputs': [-1, 4]}]}
        body = json.dumps(self.manifest).encode('utf-8')
        (self.root / 'reference/nodes/manifest.json').write_bytes(body)
        (self.root / 'model/constants.bin').write_bytes(struct.pack('<f', 2))
        (self.root / 'firmware/host-verify').write_bytes(b'stub')
        app.save_json(self.root / 'firmware/host-verify.json',
                      {'sha256': app.sha(b'stub'), 'reference_manifest_sha256': app.sha(body),
                       'sources': {'model/constants.bin': app.sha(struct.pack('<f', 2))}})
        self.calls, self.closed, self.started = [], 0, 0
        owner = self
        class FakeReplay:
            def __init__(self, mode, **kwargs):
                owner.started += 1
                assert mode == 16
            def run(self, node, tensors, inputs):
                owner.calls.append((node['id'], inputs))
                if node['id'] == 0:
                    return {2: struct.pack('<f', sum(struct.unpack('<f', inputs[t])[0] for t in (0, 1)))}
                if node['id'] == 1: return {3: inputs[2]}
                value = sum(struct.unpack('<f', inputs[t])[0] for t in (2, 3))
                return {4: candidates(value)}
            def close(self): owner.closed += 1
        self.factory = FakeReplay
        for target, value in (('platform.system', 'Linux'), ('platform.machine', 'x86_64'),
                              ('verified_binary', self.root / 'firmware/host-verify')):
            context = patch('video_golden.' + target, return_value=value)
            context.start(); self.addCleanup(context.stop)

    def engine(self):
        engine = app.VideoGolden(self.root / 'out', self.root, self.factory)
        self.addCleanup(engine.close)
        return engine

    def test_predictions_feed_branches_and_never_use_actual_candidates(self):
        engine = self.engine()
        result = engine.check(0, struct.pack('<f', 3), candidates(999))
        self.assertEqual(result['status'], 'DIFFERENT')
        self.assertEqual(result['different_elements'], 600)
        self.assertEqual(self.calls[1][1][2], struct.pack('<f', 5))
        self.assertEqual(self.calls[2][1][3], struct.pack('<f', 5))
        self.assertEqual((self.root / 'out/golden-000000.bin').read_bytes(), candidates(10))
        metadata = app.load(self.root / 'out/golden-000000.json')
        self.assertEqual(len(metadata['node_hashes']), 3)

    def test_exact_cache_is_reused_but_actual_output_is_compared_again(self):
        engine = self.engine(); data = struct.pack('<f', 3)
        self.assertEqual(engine.check(0, data, candidates(10))['status'], 'EXACT')
        engine.close()
        before = len(self.calls)
        fresh = self.engine()
        result = fresh.check(0, data, candidates(12))
        self.assertTrue(result['cached'])
        self.assertEqual(result['status'], 'DIFFERENT')
        self.assertEqual(len(self.calls), before)
        self.assertEqual(self.started, 1)

    def test_input_or_identity_changes_force_recomputation(self):
        engine = self.engine()
        engine.check(0, struct.pack('<f', 3), candidates(10))
        result = engine.check(0, struct.pack('<f', 4), candidates(12))
        self.assertFalse(result['cached']); self.assertEqual(result['status'], 'EXACT')
        engine.fingerprint = 'changed-reference'
        self.assertFalse(engine.check(0, struct.pack('<f', 4), candidates(12))['cached'])
        self.assertEqual(len(self.calls), 9)

    def test_corrupted_golden_cache_is_reported_and_not_silently_regenerated(self):
        engine = self.engine(); data = struct.pack('<f', 3)
        engine.check(0, data, candidates(10))
        path = self.root / 'out/golden-000000.bin'
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'cache corrupted'): engine.check(0, data, candidates(10))
        self.assertEqual(len(self.calls), 3)

    def test_partial_frame_without_commit_metadata_is_recomputed(self):
        (self.root / 'out/golden-000000.bin').write_bytes(b'partial')
        result = self.engine().check(0, struct.pack('<f', 3), candidates(10))
        self.assertEqual(result['status'], 'EXACT'); self.assertFalse(result['cached'])

    def test_failure_closes_reference_process_and_does_not_commit_false_golden(self):
        engine = self.engine()
        with patch.object(self.factory, 'run', side_effect=RuntimeError('bad node')):
            with self.assertRaisesRegex(RuntimeError, 'Golden node 0'): engine.check(0, struct.pack('<f', 3), candidates(10))
        self.assertEqual(self.closed, 1)
        self.assertFalse((self.root / 'out/golden-000000.json').exists())

    def test_weights_are_rechecked_after_asset_verification(self):
        (self.root / 'model/constants.bin').write_bytes(b'bad!')
        with self.assertRaisesRegex(ValueError, 'weights changed'): self.engine()

    def test_one_bit_and_signed_zero_are_differences_even_with_zero_abs_error(self):
        raw = candidates(10)
        changed = bytearray(raw); changed[3] = 0x80
        result = app.compare(bytes(changed), raw)
        self.assertEqual(result['different_elements'], 1)
        self.assertEqual(result['max_coordinate_difference'], 0)
        self.assertEqual(result['first_difference']['actual_bits'], '0x80000000')
        self.assertEqual(result['first_difference']['golden_bits'], '0x00000000')
        with self.assertRaises(ValueError): app.compare(raw[:-1], raw)
        with self.assertRaises(ValueError): app.compare(struct.pack('<f', float('nan')) + raw[4:], raw)


@unittest.skipUnless(sys.platform == 'linux', 'Packaged native auditor runs on Linux')
class NativeVideoGoldenTests(unittest.TestCase):
    def test_full_bus_forward_matches_saved_board_output(self):
        with tempfile.TemporaryDirectory() as name:
            engine = app.VideoGolden(Path(name))
            try:
                raw, hashes = engine.predict((app.ROOT / 'reference/prepared/bus-input-fp32.bin').read_bytes(),
                                             lambda node: print('BUS independent forward node={}'.format(node), flush=True))
            finally:
                engine.close()
        self.assertEqual(len(hashes), 335)
        # Final result of the complete verified BUS board capture, not a QEMU result.
        self.assertEqual(app.sha(raw), '5440862fc0d41b09eb426a548ef7afa5ff0e3bec4b36a538f420a7de1705a369')


if __name__ == '__main__':
    unittest.main()
