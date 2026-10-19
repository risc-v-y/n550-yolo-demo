"""Confidence/NMS behavior and replay evidence, with real PNGs and preprocessing."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import accuracy_review as app


class AccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.video = app.dependencies()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.video.np.zeros((64, 64, 3), dtype=self.video.np.uint8)
        self.tensor, self.transform = self.video.prepare_input(self.image)
        self.candidates = self.video.np.zeros((300, 6), dtype=self.video.np.float32)
        self.candidates[:4] = [[10,10,70,70,.9,0], [10,10,70,70,.7,0],
                              [10,10,70,70,.8,1], [180,180,240,240,.3,2]]

    def test_class_aware_nms_threshold_strictness_and_original_order(self):
        source = self.candidates.copy()
        baseline = app.select(self.video, source, self.transform, .25, None)
        result = app.select(self.video, source, self.transform, .25, .5)
        self.assertEqual(baseline['candidate_ids'], [0,1,2,3])
        self.assertEqual(result['candidate_ids'], [0,2,3])
        self.assertEqual(result['nms_removed'], [dict(candidate=1, kept_candidate=0, iou=1.)])
        self.assertEqual(app.select(self.video, source, self.transform, .35, .5)['threshold_removed'], [3])
        self.assertTrue(self.video.np.array_equal(source, self.candidates))
        source[:,4] = .25
        self.assertEqual(app.select(self.video, source, self.transform, .25, .5)['boxes'], [])
        self.assertEqual(app.iou([0,0,0,0], [0,0,0,0]), 0)
        source[:2,4] = .9
        self.assertEqual(app.select(self.video, source, self.transform, .25, .5)['candidate_ids'], [0])

    def test_metrics_are_one_to_one_and_count_removed_true_positives(self):
        boxes = [[0,0,10,10,.9,0], [0,0,10,10,.8,0], [20,20,30,30,.3,1]]
        labels = [dict(class_id=0, xyxy=[0,0,10,10]), dict(class_id=1, xyxy=[20,20,30,30])]
        self.assertEqual(app.metrics(boxes, labels), dict(tp=2, fp=1, fn=0))
        self.assertEqual(app.metrics(boxes[:1], labels), dict(tp=1, fp=0, fn=1))
        self.assertEqual(app.metrics(boxes, []), dict(tp=0, fp=3, fn=0))

    def test_png_comparisons_manual_labels_and_corrupt_evidence(self):
        source = self.root/'source-000000.png'
        app.save_png(self.video, source, self.image)
        raw = self.candidates.astype('<f4').tobytes()
        (self.root/'candidates-000000.bin').write_bytes(raw)
        record = dict(frame_id=0, transform=self.transform, output_sha256=hashlib.sha256(raw).hexdigest(),
                      input_sha256=hashlib.sha256(self.tensor.astype('<f4').tobytes()).hexdigest())
        (self.root/'result-000000.json').write_text(json.dumps(record))
        result = app.analyze(self.root, self.root/'review', video=self.video)
        self.assertIsNone(result['metrics'])
        sheet = self.video.read_image(self.root/'review/comparison-000000.png')
        self.assertEqual(sheet.shape[1], 1600)
        labels_path = self.root/'review/labels-template.json'
        labels = json.loads(labels_path.read_text()); labels['frames'][0]['reviewed'] = True
        labels['frames'][0]['objects'] = [dict(class_id=0, xyxy=[64*10/416,64*10/416,64*70/416,64*70/416])]
        labels_path.write_text(json.dumps(labels))
        result = app.analyze(self.root, self.root/'labeled', labels_path=labels_path, video=self.video)
        self.assertEqual(result['labeled_frames'], 1)
        self.assertEqual(result['metrics']['baseline']['tp'], 1)
        self.assertEqual(result['metrics']['nms050']['fp'], 2)
        self.assertEqual(result['metrics']['confidence035_nms050']['fp'], 1)
        labels['frames'][0]['source_png_sha256'] = 'bad'; labels_path.write_text(json.dumps(labels))
        with self.assertRaisesRegex(ValueError, 'labels do not match'):
            app.analyze(self.root, self.root/'bad-label', labels_path=labels_path, video=self.video)
        (self.root/'candidates-000000.bin').write_bytes(raw[:-4])
        with self.assertRaisesRegex(ValueError, 'size/SHA'):
            app.analyze(self.root, self.root/'bad', video=self.video)


if __name__ == '__main__': unittest.main()
