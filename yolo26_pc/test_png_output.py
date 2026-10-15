"""Real headless PNG rendering, original JPEG, box pixels and atomic failures."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib
from png_output import write_png,png_size
from pcie_prepared import ROOT,annotated_svg,checked_file,detections


def pixels(path):
    data=path.read_bytes(); width,height=png_size(data); at=8; blocks=[]
    while at<len(data):
        size=struct.unpack_from('>I',data,at)[0]; kind=data[at+4:at+8]; block=data[at+8:at+8+size]
        if zlib.crc32(kind+block)!=struct.unpack_from('>I',data,at+8+size)[0]: raise ValueError('PNG chunk CRC mismatch')
        if kind==b'IHDR': depth,color,_,_,interlace=struct.unpack('>2I5B',block)[2:]
        if kind==b'IDAT': blocks.append(block)
        at+=size+12
    if depth!=8 or color not in (2,6) or interlace: raise ValueError('Unexpected test PNG format')
    channels=3 if color==2 else 4; stride=width*channels
    raw=zlib.decompress(b''.join(blocks)); rows=[]; previous=bytearray(stride)
    if len(raw)!=height*(stride+1): raise ValueError('Wrong PNG data length')
    for y in range(height):
        start=y*(stride+1); mode=raw[start]; row=bytearray(raw[start+1:start+1+stride])
        for x in range(stride):
            left=row[x-channels] if x>=channels else 0; above=previous[x]
            corner=previous[x-channels] if x>=channels else 0
            p=left+above-corner; distances=(abs(p-left),abs(p-above),abs(p-corner))
            paeth=(left,above,corner)[distances.index(min(distances))]
            row[x]=(row[x]+(0,left,above,(left+above)//2,paeth)[mode])&255
        rows.append(bytes(row)); previous=row
    return width,height,channels,rows


@unittest.skipUnless(sys.platform=='linux','Uses the Linux host image-viewer libraries')
class PNGTests(unittest.TestCase):
    def test_known_pixels_and_full_png_integrity_without_display(self):
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12"><rect width="16" height="12" fill="red"/></svg>'
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'out.png'; self.assertEqual(write_png(svg,path),(16,12))
            w,h,c,rows=pixels(path)
            self.assertEqual((w,h),(16,12)); self.assertEqual(rows[0][:3],bytes([255,0,0]))
            self.assertEqual(rows[-1][-c:][:3],bytes([255,0,0]))

    def test_real_jpeg_and_both_detection_drawings(self):
        samples=json.loads((ROOT/'reference/prepared/manifest.json').read_text())['samples']
        with tempfile.TemporaryDirectory() as name:
            for key,sample in samples.items():
                source=checked_file(sample['source'],sample['source_sha256'])
                raw=checked_file(sample['reference'],sample['reference_sha256'])[32:]
                found=detections(raw,sample['transform']); paths=[Path(name)/(key+s+'.png') for s in ('-original','-detected')]
                for path,boxes in zip(paths,([],found)): write_png(annotated_svg(source,sample['transform'],boxes),path)
                original=pixels(paths[0]); detected=pixels(paths[1])
                self.assertEqual(original[:2],tuple(reversed(sample['transform']['original_hw'])))
                self.assertGreater(len(set(original[3][len(original[3])//2])),50)
                self.assertEqual(original[:3],detected[:3]); self.assertNotEqual(original[3],detected[3])
                self.assertEqual(original[3][0],detected[3][0])

    def test_invalid_input_does_not_replace_existing_image(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'out.png'; path.write_bytes(b'old')
            for svg in ('broken','<svg xmlns="http://www.w3.org/2000/svg" width="0" height="1"/>'):
                with self.assertRaises(ValueError): write_png(svg,path)
                self.assertEqual(path.read_bytes(),b'old')
            self.assertEqual([p.name for p in Path(name).iterdir()],['out.png'])


if __name__=='__main__': unittest.main()
