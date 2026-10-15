"""Offline PNG export preserves old evidence and supports reused captures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import render_results as render


@unittest.skipUnless(sys.platform=='linux','Uses Linux image renderer')
class ExportTests(unittest.TestCase):
    def test_full_report_with_reused_capture_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); report=root/'full'; report.mkdir(); capture=root/'old'; capture.mkdir()
            svg=b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="6"><rect width="8" height="6" fill="blue"/></svg>'
            (capture/'annotated.svg').write_bytes(svg)
            original=json.dumps({'status':'not_passed','samples':{'bus':{'directory':str(capture)}}}).encode()
            (report/'summary.json').write_bytes(original)
            entries=render.export(report)
            self.assertEqual(len(entries),1); self.assertEqual(entries[0]['size'],[8,6])
            self.assertEqual((capture/'annotated.svg').read_bytes(),svg)
            self.assertEqual((report/'summary.json').read_bytes(),original)
            self.assertEqual(render.png_size((capture/'annotated.png').read_bytes()),(8,6))
            self.assertEqual(len(render.export(report)),1)

    def test_failed_latest_run_falls_back_to_existing_image(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); base=root/'yolo26_pc/outputs'; menu=base/'host-menu'; menu.mkdir(parents=True)
            failed=base/'full-check/failed'; failed.mkdir(parents=True)
            older=base/'board-pcie/old'; older.mkdir(parents=True); (older/'annotated.svg').write_text('saved')
            (menu/'last-full.json').write_text(json.dumps({'directory':str(failed)}))
            with patch.object(render,'ROOT',root): self.assertEqual(render.latest(),older)

    def test_no_data_is_not_replaced_by_reference_image(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError,'No saved drawings'): render.export(Path(name))
            self.assertEqual(list(Path(name).iterdir()),[])


if __name__=='__main__': unittest.main()
