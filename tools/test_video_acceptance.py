import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import video_acceptance  # Installs the packaged model-module search path.
from board_profile import decode_profile
from test_performance_report import payload
from optimization_report import compare_display
from video_acceptance import checked_selection, checked_baseline, interval_records


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        frames=[{'input_sha256':str(i),'transform':{'original_hw':[416,416],
                 'padding_ltrb':[0,0,0,0],'scale':1.}} for i in range(3)]
        refs=[bytes(7200)]*3
        profile=decode_profile(payload(1,7,version=9),1,7)
        rows=[dict(compare_display(refs[i],refs[i],frames[i]['transform']),
                   input_sha256=frames[i]['input_sha256'],model_cycles=profile['model_cycles'],
                   elapsed_seconds=1,profile=profile) for i in range(3)]
        report={'firmware_sha256':'firmware','selected':{'m':128,'n':32,'pack':1,'compute':0,'graph':11},
                'baseline':copy.deepcopy([rows,rows]),'selected_measurements':copy.deepcopy([rows,rows])}
        return report,frames,refs

    def test_complete_old_selection_can_be_reused_without_board_or_tuning(self):
        record,frames,refs=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'; path.write_text(json.dumps(record))
            actual,identity=checked_selection(path,'firmware',frames,refs)
            self.assertEqual(actual['selected'],record['selected'])
            self.assertEqual(identity['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_changed_firmware_inputs_profiles_or_incomplete_evidence_are_rejected(self):
        record,frames,refs=self.fixture()
        for variant in ('firmware','input','profile','missing','signature','configuration'):
            bad=copy.deepcopy(record)
            if variant=='firmware': bad['firmware_sha256']='changed'
            elif variant=='input': bad['baseline'][0][0]['input_sha256']='changed'
            elif variant=='profile': bad['selected_measurements'][0][0]['profile']['model_cycles']+=1
            elif variant=='missing': bad['selected_measurements'].pop()
            elif variant=='signature': bad['selected_measurements'][0][0]['actual_signature']=[[1]]
            else: bad['selected']['pack']=-1
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'report.json'; path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): checked_selection(path,'firmware',frames,refs)

    def test_interval_comparison_excludes_first_frame_and_rejects_missing_data(self):
        checks=[{'input_sha256':str(i),'display_equal':True} for i in range(3)]
        result={'results':[{'completed_interval_seconds':x} for x in (None,2.,1.)]}
        rows=interval_records(result,checks)
        self.assertEqual([r['input_sha256'] for r in rows],['1','2'])
        self.assertEqual([r['frame_seconds'] for r in rows],[2.,1.])
        result['results'][2]['completed_interval_seconds']=float('nan')
        with self.assertRaises(ValueError): interval_records(result,checks)

    def test_baseline_requires_accepted_30_exact_frames_and_matching_configuration(self):
        record,frames,refs=self.fixture()
        record.update(status='passed',recorded_frames=30,
                      frames=[{'frame_id':i,'display_equal':True,'different_words':0} for i in range(30)])
        config=dict(record['selected'])
        for variant in ('valid','incomplete','repeated','numeric','candidate','config','failed'):
            bad=copy.deepcopy(record)
            if variant=='incomplete': bad['frames'].pop()
            elif variant=='repeated': bad['frames'][29]['frame_id']=0
            elif variant=='numeric': bad['frames'][0]['different_words']=1
            elif variant=='candidate': bad['selected_measurements'][0][0]['different_words']=1
            elif variant=='config': bad['selected']['n']=64
            elif variant=='failed': bad['status']='failed'
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'report.json'; path.write_text(json.dumps(bad))
                if variant=='valid':
                    result,_=checked_baseline(path,'firmware',config,frames,refs)
                    self.assertEqual(result['recorded_frames'],30)
                else:
                    with self.assertRaises(ValueError): checked_baseline(path,'firmware',config,frames,refs)


if __name__=='__main__': unittest.main()
