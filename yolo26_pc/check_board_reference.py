"""Verify packaged fixtures and the shared host postprocess byte-for-byte."""
import struct
from video_demo import ROOT, np, read_image, prepare_input, postprocess


def main():
    for name in ('bus','zidane'):
        image=read_image(ROOT/'samples'/f'{name}.jpg')
        _,transform=prepare_input(image)
        blob=(ROOT.parent/'reference'/f'{name}-qemu-fp16.bin').read_bytes()
        assert len(blob)==7232 and blob[:8]==b'Y26VDET\0'
        candidates=np.frombuffer(blob,dtype='<f4',offset=32).reshape(300,6)
        boxes=postprocess(candidates,transform,.25)
        actual=struct.pack('<8sII',b'Y26DETS1',1,len(boxes))+boxes.astype('<f4').tobytes()
        expected=(ROOT.parent/'reference'/f'{name}-detections.bin').read_bytes()
        if actual!=expected:
            raise AssertionError(f'{name}: host postprocess differs from packaged QEMU baseline')
        print(f'{name}: {len(boxes)} detections, byte-identical postprocess')


if __name__=='__main__':
    main()
