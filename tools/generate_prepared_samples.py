"""Generate the fixed PCIe board-validation inputs with the normal host preprocessing."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'yolo26_pc'))
from video_demo import cv2, np, prepare_input, read_image  # noqa: E402


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    if np.__version__ != '2.2.6' or cv2.__version__ != '4.11.0':
        raise RuntimeError('Use the pinned NumPy 2.2.6 and OpenCV 4.11.0.86 host dependencies')
    asset_lock = json.loads((ROOT / 'yolo26_pc' / 'assets.lock.json').read_text(encoding='utf-8'))
    source_hashes = {entry['path']: entry['sha256'] for entry in asset_lock['samples']}
    target = ROOT / 'reference' / 'prepared'
    target.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'format': 'RGB CHW [3,416,416] little-endian FP32',
                'preprocessing': 'yolo26_pc/video_demo.py:prepare_input',
                'numpy_version': np.__version__, 'opencv_version': cv2.__version__,
                'samples': {}}
    for name in ('bus', 'zidane'):
        source_path = ROOT / 'yolo26_pc' / 'samples' / f'{name}.jpg'
        reference_path = ROOT / 'reference' / f'{name}-qemu-fp16.bin'
        input_path = target / f'{name}-input-fp32.bin'
        source_blob = source_path.read_bytes()
        if sha256(source_blob) != source_hashes[f'samples/{name}.jpg']:
            raise ValueError(f'Source image hash differs from assets.lock.json: {name}')
        image = read_image(source_path)
        tensor, transform = prepare_input(image)
        data = tensor.astype('<f4', copy=False).tobytes()
        if len(data) != 3 * 416 * 416 * 4:
            raise ValueError(f'Unexpected prepared input size for {name}')
        input_path.write_bytes(data)
        manifest['samples'][name] = {
            'input': input_path.relative_to(ROOT).as_posix(),
            'input_sha256': sha256(data),
            'source': source_path.relative_to(ROOT).as_posix(),
            'source_sha256': sha256(source_blob),
            'reference': reference_path.relative_to(ROOT).as_posix(),
            'reference_sha256': sha256(reference_path.read_bytes()),
            'transform': transform,
        }
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                           encoding='utf-8')


if __name__ == '__main__':
    main()
