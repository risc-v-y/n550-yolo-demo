import copy
import struct
import unittest
from optimization_report import compare_display, repeatable_improvement, report_pages


def raw(score=.501, x=10.01, category=0):
    rows = [(x, 20., 40., 60., score, category)] + [(0., 0., 0., 0., 0., 0.)]*299
    return b''.join(struct.pack('<6f', *r) for r in rows)


class AcceptanceTests(unittest.TestCase):
    transform = {'original_hw': [416, 416], 'padding_ltrb': [0, 0, 0, 0], 'scale': 1.}

    def test_internal_changes_can_preserve_drawing(self):
        report = compare_display(raw(x=10.02), raw(), self.transform)
        self.assertTrue(report['display_equal'])
        self.assertEqual(report['different_words'], 1)

    def test_changes_to_any_visible_field_fail(self):
        for actual in (raw(x=10.6), raw(score=.52), raw(category=1), raw(score=.25)):
            self.assertFalse(compare_display(actual, raw(), self.transform)['display_equal'])

    def test_nonfinite_and_bad_length_fail(self):
        for actual in (raw(x=float('nan')), raw()[:-1], raw(category=80)):
            with self.assertRaises(ValueError):
                compare_display(actual, raw(), self.transform)

    def test_matched_repeated_gain(self):
        old = [[{'input_sha256': 'a', 'display_equal': True, 'model_cycles': 100}] for _ in range(2)]
        new = copy.deepcopy(old)
        for run in new:
            run[0]['model_cycles'] = 90
        self.assertTrue(repeatable_improvement(old, new))
        new[1][0]['model_cycles'] = 101
        self.assertFalse(repeatable_improvement(old, new))
        self.assertFalse(repeatable_improvement(old[:1], new[:1]))

    def test_report_bounds(self):
        pages = report_pages({})
        self.assertEqual(len(pages), 3)
        for page in pages:
            self.assertLessEqual(len(page.splitlines()), 22)
            self.assertTrue(all(len(x) <= 80 for x in page.splitlines()))

    def test_report_accepts_real_profile_schema(self):
        from test_performance_report import payload as profile_payload
        from board_profile import decode_profile
        profile=decode_profile(profile_payload(1,7,version=9),1,7)
        sample={'model_cycles':profile['model_cycles'],'elapsed_seconds':1,'profile':profile}
        pages=report_pages({'baseline':[[sample],[sample]],'selected_measurements':[[sample],[sample]]})
        self.assertIn('input_pack:',pages[1])
        self.assertIn('Slow node',pages[1])


if __name__ == '__main__':
    unittest.main()
