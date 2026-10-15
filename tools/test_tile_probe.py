"""Protocol rejection and numerical fault-localization tests, no physical board."""
import contextlib
import io
import struct
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

    def test_wrong_assets_crc_rejected(self):
        self.raw[16]^=1
        with self.assertRaises(ValueError): probe.analyze(self.raw,self.fixture)

    def test_pages_fit_screenshots(self):
        result=probe.analyze(self.raw,self.fixture)
        result['packing_mismatches']=[['weight',18848,15,26,65535,65535]]*3
        for stable in (True,False):
            report={'id':'123456789abc','status':'completed','stable':stable,'runs':[result,result]}
            for page in (1,2,3):
                stream=io.StringIO()
                with contextlib.redirect_stdout(stream): probe.pages(report,page)
                lines=stream.getvalue().splitlines()
                self.assertLessEqual(len(lines),22)
                self.assertLessEqual(max(map(len,lines)),80)

    def test_long_failure_fits_screenshot(self):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            probe.pages({'id':'123456789abc','status':'failed','error':'bad tool output '*1000},1)
        lines=stream.getvalue().splitlines()
        self.assertLessEqual(len(lines),22)
        self.assertLessEqual(max(map(len,lines)),80)


if __name__=='__main__': unittest.main()
