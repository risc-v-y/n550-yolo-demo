"""Offline checks for the dependency-free prepared-image PCIe workflow."""

import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zlib

import pcie_prepared
from pcie_backend import DONE, INPUT_BYTES, OUTPUT_BYTES, PCIeBackend, READY


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(pcie_prepared.MANIFEST.read_text(encoding='utf-8'))

    def test_assets_and_reference(self):
        self.assertEqual(len(pcie_prepared.CLASS_NAMES), 80)
        source = ast.parse((pcie_prepared.ROOT / 'yolo26_pc' / 'video_demo.py').read_text(encoding='utf-8'))
        names = next(ast.literal_eval(node.value) for node in source.body
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'COCO80'
                                                              for target in node.targets))
        self.assertEqual(pcie_prepared.CLASS_NAMES, names)
        for name, expected_count in (('bus', 6), ('zidane', 2)):
            with self.subTest(name=name):
                record = self.manifest['samples'][name]
                prepared = pcie_prepared.checked_file(record['input'], record['input_sha256'], INPUT_BYTES)
                self.assertEqual(hashlib.sha256(prepared).hexdigest(), record['input_sha256'])
                reference = pcie_prepared.checked_file(record['reference'], record['reference_sha256'],
                                                       pcie_prepared.REFERENCE_HEADER.size + OUTPUT_BYTES)
                raw = reference[pcie_prepared.REFERENCE_HEADER.size:]
                self.assertTrue(pcie_prepared.reference_result(raw, reference)['byte_identical'])
                found = pcie_prepared.detections(raw, record['transform'])
                self.assertEqual(len(found), expected_count)
                source = pcie_prepared.checked_file(record['source'], record['source_sha256'])
                svg = pcie_prepared.annotated_svg(source, record['transform'], found)
                root = ET.fromstring(svg)
                self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}image')), 1)
                self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}text')), expected_count)
                self.assertIn('data:image/jpeg;base64,', svg)

    def test_raw_transport_reuses_protocol(self):
        raw = pcie_prepared.ROW.pack(0, 0, 1, 1, 0.9, 0) * 300
        self.assertEqual(len(raw), OUTPUT_BYTES)
        backend = PCIeBackend.__new__(PCIeBackend)
        backend.failed = False
        backend.infer_timeout = 5
        backend.session = 123
        backend.sequence = 7
        backend.frame = 0
        backend.input_address = 0x80800000
        backend.output_address = 0x80a00000
        backend.last_timing = {}
        backend.progress = lambda message: None
        backend._snapshot = lambda deadline: None
        ready = {'session': 123, 'sequence': 7, 'state': READY}
        done = {'session': 123, 'sequence': 8, 'state': DONE, 'output_bytes': OUTPUT_BYTES,
                'output_crc': zlib.crc32(raw)}
        status_calls = iter((ready, done))
        backend._status = lambda deadline: next(status_calls)
        backend._submit = lambda command, size, crc, deadline: setattr(backend, 'sequence', 8)
        backend._wait = lambda target, deadline: done

        class Link:
            def __init__(self):
                self.writes = []

            def write(self, address, data, deadline):
                self.writes.append((address, data))

            def read(self, address, size, deadline):
                return raw

        backend.link = Link()
        prepared = bytes(INPUT_BYTES)
        self.assertEqual(backend.infer_bytes(prepared), raw)
        self.assertEqual(backend.link.writes, [(backend.input_address, prepared)])
        self.assertEqual(backend.last_timing['pcie_frame'], 1)

    def test_invalid_prepared_input_is_rejected_before_transfer(self):
        backend = PCIeBackend.__new__(PCIeBackend)
        backend.failed = False
        with self.assertRaisesRegex(ValueError, 'Expected'):
            backend.infer_bytes(b'wrong length')
        with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
            backend.infer_bytes(struct.pack('<f', float('nan')) + bytes(INPUT_BYTES - 4))

    def test_reference_difference_is_visible(self):
        record = self.manifest['samples']['bus']
        reference = (pcie_prepared.ROOT / record['reference']).read_bytes()
        raw = bytearray(reference[pcie_prepared.REFERENCE_HEADER.size:])
        original = struct.unpack_from('<f', raw)[0]
        struct.pack_into('<f', raw, 0, original + 1)
        result = pcie_prepared.reference_result(bytes(raw), reference)
        self.assertFalse(result['byte_identical'])
        self.assertGreater(result['max_coordinate_difference'], 0)

    def test_cli_writes_report_and_annotated_svg(self):
        record = self.manifest['samples']['bus']
        reference = (pcie_prepared.ROOT / record['reference']).read_bytes()
        raw = reference[pcie_prepared.REFERENCE_HEADER.size:]

        class FakeBackend:
            def __init__(self, *args, **kwargs):
                self.info = {'version': 1}
                self.last_timing = {'pcie_frame': 1}
                self.last_diagnostic = None

            def infer_bytes(self, data):
                if len(data) != INPUT_BYTES:
                    raise ValueError('Bad prepared input')
                return raw

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'result'
            argv = ['pcie_prepared.py', '--sample', 'bus', '--output', str(output)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(pcie_prepared, 'PCIeBackend', FakeBackend):
                pcie_prepared.main()
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'completed')
            self.assertTrue(report['reference']['byte_identical'])
            self.assertEqual(len(report['detections']), 6)
            self.assertEqual((output / 'candidates.bin').read_bytes(), raw)
            self.assertTrue((output / 'annotated.svg').is_file())
            self.assertTrue((output / 'diagnostics.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
