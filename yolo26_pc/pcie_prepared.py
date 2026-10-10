"""Verify prepared PCIe samples without third-party Python packages."""

import argparse
import base64
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import struct

from board_diagnostics import DiagnosticLog
from pcie_backend import INPUT_BYTES, OUTPUT_BYTES, PCIeBackend


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'reference' / 'prepared' / 'manifest.json'
REFERENCE_HEADER = struct.Struct('<8sIIQII')
ROW = struct.Struct('<6f')
CLASS_NAMES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
)


def checked_file(relative, expected_sha256, expected_size=None):
    path = ROOT / relative
    blob = path.read_bytes()
    if expected_size is not None and len(blob) != expected_size:
        raise ValueError(f'{relative}: expected {expected_size} bytes, got {len(blob)}')
    if hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError(f'{relative}: SHA-256 mismatch')
    return blob


def reference_result(actual, expected_blob):
    if len(expected_blob) != REFERENCE_HEADER.size + OUTPUT_BYTES:
        raise ValueError('Reference candidate length mismatch')
    header = REFERENCE_HEADER.unpack_from(expected_blob)
    if header[0] != b'Y26VDET\0' or header[1] != 1 or header[4:] != (300, 6):
        raise ValueError('Invalid Y26VDET reference header')
    expected = expected_blob[REFERENCE_HEADER.size:]
    actual_rows = list(ROW.iter_unpack(actual))
    expected_rows = list(ROW.iter_unpack(expected))
    if len(actual_rows) != 300 or len(expected_rows) != 300:
        raise ValueError('Candidate row count mismatch')
    return {
        'byte_identical': actual == expected,
        'class_sequence_match': all(a[5] == b[5] for a, b in zip(actual_rows, expected_rows)),
        'max_coordinate_difference': max(abs(a[i] - b[i]) for a, b in zip(actual_rows, expected_rows)
                                         for i in range(4)),
        'max_score_difference': max(abs(a[4] - b[4]) for a, b in zip(actual_rows, expected_rows)),
    }


def detections(raw, transform, threshold=0.25):
    height, width = transform['original_hw']
    left, top = transform['padding_ltrb'][:2]
    scale = transform['scale']
    if height <= 0 or width <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid sample transform')
    found = []
    for x1, y1, x2, y2, score, category in ROW.iter_unpack(raw):
        if score <= threshold:
            continue
        category = int(category)
        coords = [max(0.0, min(bound, (value - pad) / scale)) for value, pad, bound in
                  zip((x1, y1, x2, y2), (left, top, left, top), (width, height, width, height))]
        found.append({'xyxy': coords, 'confidence': score, 'class_id': category,
                      'class_name': CLASS_NAMES[category]})
    return found


def annotated_svg(source, transform, found):
    height, width = transform['original_hw']
    image = base64.b64encode(source).decode('ascii')
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{image}"/>']
    for item in found:
        x1, y1, x2, y2 = item['xyxy']
        category = item['class_id']
        color = f'rgb({29 * (category + 7) % 255},{17 * (category + 11) % 255},'
        color += f'{37 * (category + 3) % 255})'
        label = html.escape(f"{item['class_name']} {item['confidence']:.2f}")
        label_width = min(width - x1, max(70, len(label) * 9 + 10))
        label_top = max(0, y1 - 25)
        lines.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{max(0,x2-x1):.2f}" '
                     f'height="{max(0,y2-y1):.2f}" fill="none" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<rect x="{x1:.2f}" y="{label_top:.2f}" width="{label_width:.2f}" '
                     f'height="25" fill="{color}"/>')
        lines.append(f'<text x="{x1+4:.2f}" y="{label_top+18:.2f}" fill="white" '
                     f'font-size="17" font-family="sans-serif">{label}</text>')
    lines.append('</svg>')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', choices=('bus', 'zidane'), required=True)
    parser.add_argument('--pbcopy', default='pbcopy')
    parser.add_argument('--pbload', default='pbload')
    parser.add_argument('--infer-timeout', type=float, default=900)
    parser.add_argument('--tool-timeout', type=float, default=30)
    parser.add_argument('--board-trace', action='store_true', help='Print every node on the debug UART')
    parser.add_argument('--check-intermediates', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'yolo26_pc' / 'outputs' / 'board-pcie' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('version') != 1:
        raise ValueError('Incompatible prepared sample manifest')
    sample = manifest['samples'][args.sample]
    prepared = checked_file(sample['input'], sample['input_sha256'], INPUT_BYTES)
    source = checked_file(sample['source'], sample['source_sha256'])
    reference = checked_file(sample['reference'], sample['reference_sha256'],
                             REFERENCE_HEADER.size + OUTPUT_BYTES)
    output.mkdir(parents=True, exist_ok=False)
    log = DiagnosticLog(output)
    backend = None
    report = {'status': 'starting', 'sample': args.sample, 'input_sha256': sample['input_sha256']}
    try:
        backend = PCIeBackend(args.pbcopy, args.pbload, infer_timeout=args.infer_timeout,
                              tool_timeout=args.tool_timeout, progress=print, diagnostic=log.emit,
                              trace=args.board_trace, check_finite=args.check_intermediates)
        report['board'] = backend.info
        raw = backend.infer_bytes(prepared)
        found = detections(raw, sample['transform'])
        comparison = reference_result(raw, reference)
        (output / 'candidates.bin').write_bytes(raw)
        (output / 'annotated.svg').write_text(annotated_svg(source, sample['transform'], found),
                                              encoding='utf-8')
        report.update(status='completed', timing=backend.last_timing, reference=comparison,
                      detections=found, annotated='annotated.svg')
        print(f"{args.sample}: {len(found)} detections; reference byte-identical={comparison['byte_identical']}")
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        log.exception(exc)
        raise
    finally:
        if backend is not None:
            report['last_diagnostic'] = backend.last_diagnostic
            backend.close()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8')
        log.close()
        print(output, flush=True)


if __name__ == '__main__':
    main()
