"""Protocol rejection and numerical fault-localization tests, no physical board."""
import contextlib
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch
import tile_probe as probe
from pcie_backend import INFO, MAGIC, INPUT_BYTES, PCIeBackend


def blank():
    raw=bytearray(probe.SIZE)
    struct.pack_into('<16I',raw,0,0x31505459,1,probe.SIZE,3,0,0,0,7,*([0]*8))
    for slot,first in enumerate(probe.STARTS):
        struct.pack_into('<7I',raw,64+slot*8320,first,16,32,27,27,32,43264)
    return raw


def source_report(raws):
    """Pre-v2 report structure, without the new numerical comparison fields."""
    return {'id':'original1234','status':'completed','mode':'physical board',
            'firmware_info':{'capabilities':2,'output_bytes':probe.SIZE},
            'old_capture':None,'elapsed_seconds':12.5,
            'stable':raws[0]==raws[1],
            'runs':[{'sha256':hashlib.sha256(raw).hexdigest()} for raw in raws]}


class RoundingTests(unittest.TestCase):
    def test_midpoints_round_to_even_for_both_signs(self):
        for sign in (1,-1):
            lower_even=sign*(Fraction(1)+Fraction(1,2**24))
            upper_even=sign*(Fraction(1)+Fraction(3,2**24))
            self.assertEqual(probe.bits(probe.fp32_rne(lower_even)),
                             '3f800000' if sign==1 else 'bf800000')
            self.assertEqual(probe.bits(probe.fp32_rne(upper_even)),
                             '3f800002' if sign==1 else 'bf800002')

    def test_fraction_rounding_avoids_binary64_double_rounding(self):
        halfway=Fraction(1)+Fraction(1,2**24)
        for sign in (1,-1):
            exact=sign*(halfway+Fraction(1,2**80))
            # Converting to binary64 first loses the tiny, decisive tail.
            self.assertNotEqual(probe.bits(probe.fp32_rne(exact)),probe.bits(float(exact)))
            self.assertEqual(probe.bits(probe.fp32_rne(exact)),
                             '3f800001' if sign==1 else 'bf800001')

    def test_subnormal_and_normal_boundary_rounding(self):
        cases=((Fraction(1,2**150),'00000000'),
               (-Fraction(1,2**150),'80000000'),
               (Fraction(1,2**150)+Fraction(1,2**180),'00000001'),
               (Fraction(3,2**150),'00000002'),
               (Fraction(1,2**126)-Fraction(1,2**150),'00800000'))
        for exact,expected in cases:
            with self.subTest(exact=exact):
                self.assertEqual(probe.bits(probe.fp32_rne(exact)),expected)

    def test_split16_differs_from_once_rounded_dot(self):
        products=[1.0,2.0**-24]+[0.0]*14+[2.0**-24]+[0.0]*10
        pre,post=probe.split16(products,0.0)
        self.assertEqual(probe.bits(pre),'3f800000')
        self.assertEqual(probe.bits(post),'3f800000')
        self.assertEqual(probe.bits(probe.fp32_rne(sum(map(Fraction,products)))),'3f800001')

    def test_second_slice_must_not_be_rounded_before_adding_accumulator(self):
        products=[1.0]+[0.0]*15+[2.0**-24,2.0**-48]+[0.0]*9
        pre,post=probe.split16(products,0.0)
        self.assertEqual(probe.bits(pre),'3f800001')
        self.assertEqual(probe.bits(post),'3f800001')
        independently_rounded_tail=probe.fp32_rne(sum(map(Fraction,products[16:])))
        wrong=probe.fp32_rne(Fraction(1)+Fraction(independently_rounded_tail))
        self.assertEqual(probe.bits(wrong),'3f800000')

    def test_bias_is_applied_after_fp32_accumulation(self):
        products=[1.0,2.0**-24]+[0.0]*25
        pre,post=probe.split16(products,-1.0)
        self.assertEqual(probe.bits(pre),'3f800000')
        self.assertEqual(probe.bits(post),'00000000')


class FormatTests(unittest.TestCase):
    def test_partial_wrong_version_and_stride_rejected(self):
        for at,value in ((0,0),(4,2),(28,3),(64+24,32),(64+28,1)):
            raw=blank(); struct.pack_into('<I',raw,at,value)
            with self.assertRaises(ValueError): probe.decode(raw)
        with self.assertRaises(ValueError): probe.decode(blank()[:-1])

    def test_nonfinite_active_lane_rejected_but_padding_observed(self):
        raw=blank(); struct.pack_into('<H',raw,64+64+27*2,0x7e00)
        probe.decode(raw)
        struct.pack_into('<H',raw,64+64,0x7e00)
        with self.assertRaises(ValueError): probe.decode(raw)

    def test_normal_and_probe_firmware_cannot_be_confused(self):
        class Link:
            closed=False
            def __init__(self,size,nodes,caps):
                b=INFO.pack(MAGIC,1,256,caps,0xf0000000,0x80a00000,0x80b00000,
                            INPUT_BYTES,size,816,nodes,0,0)
                self.blob=b[:60]+struct.pack('<I',zlib.crc32(b[:60]))
            def read(self,*args): return self.blob
            def close(self): self.closed=True
        for backend,size,nodes,caps in ((PCIeBackend,probe.SIZE,1,2),
                                        (probe.ProbeBackend,7200,334,1),
                                        (probe.ProbeBackend,probe.SIZE,1,0)):
            link=Link(size,nodes,caps)
            with self.assertRaises(ValueError): backend(transport=link)
            self.assertTrue(link.closed)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture=({'nodes':[{}],'tensors':[]},{0:b'input'},b'weights',bytes(692224*4))
        self.raw=blank()
        struct.pack_into('<2I',self.raw,16,zlib.crc32(b'input'),zlib.crc32(b'weights'))
        self.terms=[{'weight_half':0,'input_half':0} for _ in range(27)]
        self.mock=patch.object(probe,'dot_terms',return_value=(self.terms,0.0))
        self.mock.start(); self.addCleanup(self.mock.stop)

    def test_clean_and_changed_bias_stage(self):
        clean=probe.analyze(self.raw,self.fixture)
        self.assertTrue(all(t['post_vs_actual_pre_plus_bias']==0 for t in clean['tiles']))
        self.assertTrue(all(t['pre_vs_split16']==t['post_vs_split16']==0 for t in clean['tiles']))
        struct.pack_into('<f',self.raw,64+6208+28*4,1.0)
        result=probe.analyze(self.raw,self.fixture)
        self.assertEqual(result['tiles'][0]['post_vs_actual_pre_plus_bias'],1)
        self.assertEqual(result['tiles'][0]['pre_vs_exact_RNE'],0)
        self.assertEqual(result['points'][0]['post'],'3f800000')

    def test_changed_amu_stage_and_packing_are_separate(self):
        struct.pack_into('<f',self.raw,64+4160+28*4,1.0)
        result=probe.analyze(self.raw,self.fixture)
        self.assertEqual(result['tiles'][0]['pre_vs_exact_RNE'],1)
        self.assertEqual(result['tiles'][0]['weight_bad'],0)
        struct.pack_into('<H',self.raw,64+64,0x3c00)
        result=probe.analyze(self.raw,self.fixture)
        self.assertEqual(result['tiles'][0]['weight_bad'],1)
        self.assertEqual(result['packing_mismatches'][0][:4],['weight',0,0,0])

    def test_each_single_bit_change_is_detected_at_its_stage(self):
        for field,offset in (('pre',4160),('post',6208)):
            with self.subTest(field=field):
                raw=bytearray(self.raw)
                struct.pack_into('<I',raw,64+offset+28*4,1)
                result=probe.analyze(raw,self.fixture)
                self.assertEqual(result['tiles'][0]['pre_vs_split16'],int(field=='pre'))
                self.assertEqual(result['tiles'][0]['post_vs_split16'],int(field=='post'))
                point=result['points'][0]
                self.assertEqual(point[field],'00000001')
                self.assertEqual(point['pre_split16'],'00000000')
                self.assertEqual(point['post_split16'],'00000000')

    def test_all_1536_values_use_actual_packed_lanes_and_split16(self):
        lanes=[0x3c00,0x0c00]+[0]*14+[0x0c00]+[0]*10
        self.terms[:]=[{'weight_half':x,'input_half':x} for x in lanes]
        raw=bytearray(self.raw)
        for slot in range(3):
            start=64+slot*8320
            for row in range(16):
                struct.pack_into('<27H',raw,start+64+row*64,*lanes)
            for row in range(32):
                struct.pack_into('<27H',raw,start+2112+row*64,*lanes)
            for offset in (4160,6208):
                struct.pack_into('<512f',raw,start+offset,*([1.0]*512))
        result=probe.analyze(raw,self.fixture)
        self.assertEqual(sum(t['pre_vs_exact_RNE'] for t in result['tiles']),1536)
        self.assertEqual(sum(t['pre_vs_split16'] for t in result['tiles']),0)
        self.assertEqual(sum(t['post_vs_split16'] for t in result['tiles']),0)
        self.assertEqual(result['packing_mismatches'],[])

    def test_wrong_assets_crc_rejected(self):
        self.raw[16]^=1
        with self.assertRaises(ValueError): probe.analyze(self.raw,self.fixture)

    def test_pages_fit_screenshots(self):
        result=probe.analyze(self.raw,self.fixture)
        result['packing_mismatches']=[['weight',18848,15,26,65535,65535]]*3
        for stable in (True,False):
            report={'id':'123456789abc','status':'completed','stable':stable,
                    'analysis_version':2,'runs':[result,result]}
            for page in (1,2,3):
                stream=io.StringIO()
                with contextlib.redirect_stdout(stream): probe.pages(report,page)
                lines=stream.getvalue().splitlines()
                self.assertLessEqual(len(lines),22)
                self.assertLessEqual(max(map(len,lines)),80)

    def test_long_failure_fits_screenshot(self):
        for page in (1,2,3):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                probe.pages({'id':'123456789abc','status':'failed','error':'bad tool output '*1000},page)
            lines=stream.getvalue().splitlines()
            self.assertLessEqual(len(lines),22)
            self.assertLessEqual(max(map(len,lines)),80)

    def test_legacy_pages_do_not_require_split16_fields(self):
        result=probe.analyze(self.raw,self.fixture)
        for tile in result['tiles']:
            for name in ('pre_vs_split16','post_vs_split16'): tile.pop(name)
        for point in result['points']:
            for name in ('pre_split16','post_split16'): point.pop(name)
        report={'id':'123456789abc','status':'completed','stable':True,'runs':[result,result]}
        for page in (1,2,3):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): probe.pages(report,page)
            lines=stream.getvalue().splitlines()
            self.assertLessEqual(len(lines),22)
            self.assertLessEqual(max(map(len,lines)),80)


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='tile-probe-test-')
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.fixture=({'nodes':[{}],'tensors':[]},{0:b'input'},b'weights',bytes(692224*4))
        self.raw=blank()
        struct.pack_into('<2I',self.raw,16,zlib.crc32(b'input'),zlib.crc32(b'weights'))
        terms=[{'weight_half':0,'input_half':0} for _ in range(27)]
        self.asset_mock=patch.object(probe,'assets',return_value=self.fixture)
        self.terms_mock=patch.object(probe,'dot_terms',return_value=(terms,0.0))
        self.asset_mock.start(); self.addCleanup(self.asset_mock.stop)
        self.terms_mock.start(); self.addCleanup(self.terms_mock.stop)
        self.report=self.save_capture((self.raw,self.raw))

    def save_capture(self,raws):
        for i,raw in enumerate(raws,1):
            (self.directory/('probe-{}.bin'.format(i))).write_bytes(raw)
        report=source_report(raws)
        (self.directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report

    def originals(self):
        return {name:(self.directory/name).read_bytes()
                for name in ('summary.json','probe-1.bin','probe-2.bin')}

    def test_reanalysis_keeps_sources_and_preserves_acquisition_provenance(self):
        original=self.originals()
        report=probe.reanalyze(self.directory)
        self.assertEqual(self.originals(),original)
        self.assertEqual(report['analysis_version'],2)
        for field in ('mode','status','firmware_info','old_capture','elapsed_seconds'):
            self.assertEqual(report[field],self.report[field])
        self.assertEqual(report['source_report_id'],self.report['id'])
        self.assertEqual(report['source_report_sha256'],
                         hashlib.sha256(original['summary.json']).hexdigest())
        self.assertEqual(report['source_runs_sha256'],[r['sha256'] for r in self.report['runs']])
        self.assertEqual(report['source_runs_sha256'],[r['sha256'] for r in report['runs']])
        saved_report=json.loads(json.dumps(report))
        self.assertEqual(json.loads((self.directory/'analysis-v2.json').read_text()),saved_report)
        self.assertTrue(report['stable'])
        self.assertEqual(len(report['runs']),2)
        self.assertEqual(probe.load_report(self.directory),saved_report)

    def test_legacy_report_loads_without_reanalysis_or_hardware(self):
        with patch.object(probe,'reanalyze',side_effect=AssertionError('unexpected analysis')):
            self.assertEqual(probe.load_report(self.directory),self.report)
        self.assertFalse((self.directory/'analysis-v2.json').exists())

    def test_prior_capture_path_is_reused(self):
        self.report['old_capture']=str(self.directory/'old-full-capture')
        (self.directory/'summary.json').write_text(json.dumps(self.report),encoding='utf-8')
        with patch.object(probe,'old_capture',return_value=None) as old:
            report=probe.reanalyze(self.directory)
        old.assert_called_once_with(Path(self.report['old_capture']),self.fixture)
        self.assertEqual(report['old_capture'],self.report['old_capture'])

    def test_both_unstable_runs_are_compared_independently(self):
        changed=bytearray(self.raw)
        struct.pack_into('<I',changed,64+4160+28*4,1)
        self.save_capture((self.raw,changed))
        report=probe.reanalyze(self.directory)
        self.assertFalse(report['stable'])
        self.assertEqual(report['runs'][0]['tiles'][0]['pre_vs_split16'],0)
        self.assertEqual(report['runs'][1]['tiles'][0]['pre_vs_split16'],1)
        self.assertEqual(report['runs'][0]['points'][0]['pre'],'00000000')
        self.assertEqual(report['runs'][1]['points'][0]['pre'],'00000001')

    def test_modified_or_missing_raw_rejected_without_replacing_valid_sidecar(self):
        probe.reanalyze(self.directory)
        sidecar=(self.directory/'analysis-v2.json').read_bytes()
        path=self.directory/'probe-2.bin'
        changed=bytearray(path.read_bytes()); changed[100]^=1
        path.write_bytes(changed)
        with self.assertRaises((ValueError,OSError)): probe.reanalyze(self.directory)
        with self.assertRaises((ValueError,OSError)): probe.load_report(self.directory)
        self.assertEqual((self.directory/'analysis-v2.json').read_bytes(),sidecar)
        path.unlink()
        with self.assertRaises((ValueError,OSError)): probe.reanalyze(self.directory)
        with self.assertRaises((ValueError,OSError)): probe.load_report(self.directory)
        self.assertEqual((self.directory/'analysis-v2.json').read_bytes(),sidecar)

    def test_modified_original_report_invalidates_sidecar(self):
        probe.reanalyze(self.directory)
        original=self.directory/'summary.json'
        original.write_bytes(original.read_bytes()+b'\n')
        with self.assertRaises(ValueError): probe.load_report(self.directory)

    def test_cached_analysis_must_bind_each_run_to_the_original_raw(self):
        probe.reanalyze(self.directory)
        sidecar=self.directory/'analysis-v2.json'
        report=json.loads(sidecar.read_text())
        report['runs'][1]['sha256']='0'*64
        sidecar.write_text(json.dumps(report),encoding='utf-8')
        with self.assertRaises(ValueError): probe.load_report(self.directory)

    def test_late_comparison_failure_preserves_existing_sidecar(self):
        probe.reanalyze(self.directory)
        sidecar=(self.directory/'analysis-v2.json').read_bytes()
        analyzed=probe.analyze(self.raw,self.fixture)
        with patch.object(probe,'analyze',side_effect=[analyzed,ValueError('second run invalid')]):
            with self.assertRaises(ValueError): probe.reanalyze(self.directory)
        self.assertEqual((self.directory/'analysis-v2.json').read_bytes(),sidecar)

    def test_cli_reanalysis_and_report_do_not_load_or_access_board(self):
        argv=['tile_probe.py','--reanalyze','--report',str(self.directory),'--page','1']
        stream=io.StringIO()
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(stream), \
             patch.object(probe,'ProbeBackend',side_effect=AssertionError('unexpected PCIe')), \
             patch.object(probe.subprocess,'run',side_effect=AssertionError('unexpected loader')):
            self.assertEqual(probe.main(),0)
        self.assertIn('TILE PROBE',stream.getvalue())
        self.assertTrue((self.directory/'analysis-v2.json').exists())


class ScreenshotGoldenTests(unittest.TestCase):
    def test_five_screenshot_points_match_split16_replay_of_verified_bus_assets(self):
        # These five points are independent screenshot goldens, not a claim that
        # all 1,536 real-board values have already been supplied or matched.
        manifest,inputs,_,_=probe.assets()
        expected=((28,'bed2f0a4','3fa9d738'),(29,'bf88bde7','3f2baaf4'),
                  (30,'bc771180','3fdca53e'),(15357,'bfde91b3','38570000'),
                  (624567,'42811790','4280f2e9'))
        for element,pre_bits,post_bits in expected:
            terms,bias=probe.dot_terms(manifest['nodes'][0],manifest['tensors'],inputs,element)
            pre,post=probe.split16([t['product'] for t in terms],bias)
            with self.subTest(element=element):
                self.assertEqual(probe.bits(pre),pre_bits)
                self.assertEqual(probe.bits(post),post_bits)


if __name__=='__main__': unittest.main()
