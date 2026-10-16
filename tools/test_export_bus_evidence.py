"""Export saved evidence, relocate it and reject missing/corrupt data; no board."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

import export_bus_evidence as app
import full_evidence as evidence
from node_capture import HEADER, TENSOR, parse_snapshot
from pcie_backend import MAGIC, PBTools


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ExportTests(unittest.TestCase):
    def test_clean_process_loads_runtime_modules_without_pythonpath(self):
        entry = Path(app.__file__).resolve()
        code = ('import sys,runpy; sys.path.insert(0,{!r}); '
                'runpy.run_path({!r},run_name="export_startup"); '
                'import node_capture,full_evidence,tile_probe').format(str(entry.parent), str(entry))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-c', code], cwd=directory,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / 'yolo26_pc/outputs'
        self.check = self.outputs / 'full-check/latest'
        self.bus = self.outputs / 'node-capture/reused-bus'
        self.probe = self.outputs / 'host-menu/old-probe/capture'
        for directory in (self.check / 'bus', self.check / 'zidane', self.bus / 'tensors',
                          self.bus / 'inputs', self.bus / 'snapshots', self.probe,
                          self.root / 'firmware', self.root / 'model', self.root / 'reference/nodes',
                          self.root / 'reference/prepared', self.outputs / 'host-menu'):
            directory.mkdir(parents=True, exist_ok=True)
        self.raw = bytes(7200)
        self.tensors = [{'dtype': 1, 'shape': [300, 6], 'bytes': 7200, 'constant': 0, 'count': 1800}
                        for _ in range(549)]
        self.node = {'id': 0, 'op': 5, 'kind': 'COPY', 'inputs': [0], 'outputs': [548]}
        self.manifest = {'tensors': self.tensors, 'nodes': [self.node]}
        self.save(self.root / 'reference/nodes/manifest.json', self.manifest)
        self.prepared = json.loads((app.ROOT / 'reference/prepared/manifest.json').read_text())['samples']['bus']
        self.prepared['input_sha256'] = sha(self.raw)
        self.save(self.root / 'reference/prepared/manifest.json', {'samples': {'bus': self.prepared}})
        self.save(self.root / 'model/manifest.json', {})
        self.save(self.root / 'firmware/host-verify.json', {})
        (self.root / 'firmware/SHA256SUMS').write_text('fixture\n')
        (self.root / 'firmware/demo-pcie.bin').write_bytes(b'firmware')
        self.capture_summary = {'status': 'completed', 'sample': 'bus', 'evidence_version': 1,
                                'nodes_captured': 1, 'tensors_captured': 1, 'input_sha256': sha(self.raw),
                                'firmware_sha256': sha(b'firmware'),
                                'reference_manifest_sha256': sha((self.root / 'reference/nodes/manifest.json').read_bytes()),
                                'normal_repeat': {'sha256': sha(self.raw), 'byte_identical_to_capture': True}}
        self.save(self.bus / 'summary.json', self.capture_summary)
        (self.bus / 'input.bin').write_bytes(self.raw)
        (self.bus / 'normal-candidates.bin').write_bytes(self.raw)
        (self.bus / 'tensors/0548.bin').write_bytes(self.raw)
        (self.bus / 'inputs/extra-weight.bin').write_bytes(b'physical weight')
        (self.bus / 'return.zip').write_bytes(b'old redundant archive')
        (self.check / 'zidane/secret.bin').write_bytes(b'excluded')
        blob = bytearray(512)
        HEADER.pack_into(blob, 0, MAGIC, 1, 512, 0, 123, 1, 2, 5, 1, 1, 1, 12, 0, 0)
        for slot in range(6):
            struct.pack_into('<i', blob, 64 + slot * 64, -1)
        for slot, tid in ((0, 0), (4, 548)):
            TENSOR.pack_into(blob, 64 + slot * 64, tid, 1, 2, 7200, 0xf0000000, 0, 1800,
                             300, 6, 0, 0, zlib.crc32(self.raw), zlib.crc32(self.raw), 0, 0)
        struct.pack_into('<I', blob, 508, zlib.crc32(blob[:508]))
        backend = type('B', (), {'session': 123, 'frame': 1, 'sequence': 2, 'options': 12,
                                'link': type('L', (), {'_range': staticmethod(PBTools._range)})()})()
        snapshot = parse_snapshot(blob, {'detail': 0, 'output_bytes': 512, 'output_crc': zlib.crc32(blob)}, backend)
        (self.bus / 'snapshots/0000.bin').write_bytes(blob)
        self.save(self.bus / 'snapshots/0000.json', {
            'version': 1, 'snapshot': snapshot, 'snapshot_sha256': sha(blob),
            'inputs': [{'id': 0, 'verified': True, 'crc32_before': zlib.crc32(self.raw),
                        'path': 'input.bin', 'sha256': sha(self.raw)}],
            'outputs': [{'id': 548, 'sha256': sha(self.raw)}]})
        probe_raw = bytes(25024)
        for i in (1, 2):
            (self.probe / 'probe-{}.bin'.format(i)).write_bytes(probe_raw)
        self.save(self.probe / 'summary.json', {'id': 'probe', 'status': 'completed', 'mode': 'physical board',
                                               'runs': [{'sha256': sha(probe_raw)}] * 2})
        self.gate = {'status': 'passed', 'directory': str(self.probe),
                     'source_report_sha256': sha((self.probe / 'summary.json').read_bytes()),
                     'raw_sha256': [sha(probe_raw)] * 2}
        self.save(self.check / 'tile-gate.json', self.gate)
        self.save(self.check / 'summary.json', {'id': 'parent', 'status': 'not_passed',
                  'samples': {'bus': {'directory': str(self.bus), 'passed': False}}, 'tile_gate': self.gate})
        self.save(self.outputs / 'host-menu/last-full.json', {'directory': str(self.check)})
        for module in (app, evidence):
            change = patch.object(module, 'ROOT', self.root)
            change.start(); self.addCleanup(change.stop)

    def save(self, path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def export(self):
        # The tiny graph is a fixture, while real snapshot/CRC/final validation runs.
        with patch('node_capture.verify_reference_manifest'), patch.object(app, 'git_metadata', return_value={'commit': 'fixture'}), \
             patch('pcie_backend.PCIeBackend', side_effect=AssertionError('board opened')), \
             patch('full_replay.Replay', side_effect=AssertionError('numerical replay started')):
            return app.export(output_dir=self.outputs / 'export')

    def test_complete_reused_bus_archive_and_hashes_survive_relocation(self):
        original = (self.bus / 'summary.json').read_bytes()
        archive = self.export()
        with zipfile.ZipFile(archive) as bundle:
            record = json.loads(bundle.read('manifest.json'))
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['capture']['nodes'], 1)
            self.assertTrue(record['capture']['normal_repeat_exact'])
            self.assertEqual(record['origins']['bus'], str(self.bus))
            self.assertIn('bus/inputs/extra-weight.bin', bundle.namelist())
            self.assertFalse(any('zidane' in p or p.endswith('return.zip') for p in bundle.namelist()))
            for entry in record['files']:
                self.assertEqual(sha(bundle.read(entry['path'])), entry['sha256'])
            moved = self.root / 'relocated'
            bundle.extractall(moved)
        checked = app.validate_capture(moved / 'bus', self.manifest, self.prepared)
        self.assertEqual(checked['capture_sha256'], sha(self.raw))
        self.assertEqual(original, (self.bus / 'summary.json').read_bytes())
        expected = archive.with_suffix('.zip.sha256').read_text().split()[0]
        self.assertEqual(expected, app.digest_file(archive))

    def test_missing_tensor_and_corrupt_input_do_not_publish_complete_archive(self):
        for name in ('tensors/0548.bin', 'input.bin'):
            with self.subTest(name=name):
                path = self.bus / name
                saved = path.read_bytes()
                path.unlink()
                with self.assertRaises((FileNotFoundError, ValueError)):
                    self.export()
                path.write_bytes(saved)
        (self.bus / 'input.bin').write_bytes(b'x' * 7200)
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.export()
        self.assertEqual(list(self.outputs.glob('export/*.zip')), [])

    def test_missing_probe_and_changed_probe_hash_are_explicit(self):
        path = self.probe / 'probe-2.bin'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing 1 required'): self.export()
        path.write_bytes(b'x' * 25024)
        with self.assertRaises(ValueError): self.export()
        self.assertEqual(list(self.outputs.glob('export/*.zip')), [])

    def test_report_pointer_is_used_instead_of_an_unrelated_new_capture(self):
        other = self.check / 'bus/attempt-2'; other.mkdir()
        self.save(other / 'summary.json', {'sample': 'bus', 'status': 'completed'})
        self.assertEqual(app.locate()[1], self.bus)
        self.gate['directory'] = str(self.root / 'outside')
        self.save(self.check / 'tile-gate.json', self.gate)
        with self.assertRaisesRegex(ValueError, 'allowed directory'): app.locate()

    def test_source_change_during_zip_and_size_limit_leave_no_finished_output(self):
        source = self.root / 'source.bin'; source.write_bytes(b'original')
        original_write = zipfile.ZipFile.write
        def changed(archive, path, *args, **kwargs):
            source.write_bytes(b'changed')
            return original_write(archive, path, *args, **kwargs)
        destination = self.root / 'bad'
        with patch.object(zipfile.ZipFile, 'write', changed):
            with self.assertRaisesRegex(ValueError, 'changed during export'):
                app.make_archive({'bus/test.bin': source}, {}, destination)
        self.assertEqual(list(destination.iterdir()), [])
        with patch.object(app, 'MAX_ARCHIVE_BYTES', 1):
            with self.assertRaisesRegex(ValueError, '3 GB'):
                app.make_archive({'bus/test.bin': source}, {}, destination)
        self.assertEqual(list(destination.iterdir()), [])

    def test_incomplete_capture_and_changed_snapshot_are_rejected(self):
        self.capture_summary['nodes_captured'] = 0
        self.save(self.bus / 'summary.json', self.capture_summary)
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.export()
        self.capture_summary['nodes_captured'] = 1
        self.save(self.bus / 'summary.json', self.capture_summary)
        path = self.bus / 'snapshots/0000.bin'
        path.write_bytes(bytes(512))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'): self.export()

    def test_all_missing_dependencies_are_reported_together(self):
        for path in (self.bus / 'snapshots/0000.bin', self.bus / 'tensors/0548.bin', self.probe / 'probe-1.bin'):
            path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing 3 required') as caught:
            app.require_files(self.bus, self.probe, self.manifest)
        for name in ('snapshots', '0548.bin', 'probe-1.bin'):
            self.assertIn(name, str(caught.exception))

    @unittest.skipUnless(__import__('sys').platform == 'linux', 'Uses Linux flock')
    def test_active_controller_refuses_export_before_reading_evidence(self):
        import host_control
        with patch.object(host_control, 'ROOT', self.root), \
             patch('sys.argv', ['export_bus_evidence.py']), patch.object(app, 'export') as export:
            with host_control.guard(host_control.workspace() / 'control.lock'):
                self.assertEqual(app.main(), 1)
        export.assert_not_called()

    @unittest.skipUnless(__import__('sys').platform == 'linux', 'Uses Linux symlinks')
    def test_symlink_in_capture_is_rejected(self):
        outside = self.root / 'outside.bin'; outside.write_bytes(b'do not export')
        (self.bus / 'linked.bin').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'): self.export()

    def test_output_inside_source_is_refused(self):
        with patch('node_capture.verify_reference_manifest'), patch.object(app, 'git_metadata', return_value={}):
            with self.assertRaisesRegex(ValueError, 'outside source'):
                app.export(output_dir=self.bus / 'exports')
        self.assertFalse((self.bus / 'exports').exists())


if __name__ == '__main__':
    unittest.main()
