"""Offline confidence/NMS comparisons; existing candidates, never board inference.

Optional complete manual labels give precision/recall at IoU 0.5, not mAP.
No parameter is applied to the live demo by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from host_video import dependencies, save_json, save_png, validate_candidates

VARIANTS = (('baseline', .25, None), ('confidence035', .35, None),
            ('nms050', .25, .5), ('confidence035_nms050', .35, .5))


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = max(0., a[2]-a[0])*max(0., a[3]-a[1]) + max(0., b[2]-b[0])*max(0., b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def select(video, candidates, transform, confidence, nms):
    boxes = video.postprocess(candidates, transform, confidence)
    ids = video.np.flatnonzero(candidates[:, 4] > video.np.float32(confidence)).tolist()
    survivors, removed = list(range(len(boxes))), []
    if nms is not None:
        survivors = []
        for index in sorted(range(len(boxes)), key=lambda i: (-float(boxes[i, 4]), ids[i])):
            match = next((j for j in survivors if boxes[j, 5] == boxes[index, 5] and
                          iou(boxes[j, :4].tolist(), boxes[index, :4].tolist()) > nms), None)
            if match is None: survivors.append(index)
            else:
                removed.append(dict(candidate=ids[index], kept_candidate=ids[match],
                                    iou=iou(boxes[match, :4].tolist(), boxes[index, :4].tolist())))
        survivors.sort()  # Preserve original model order for drawing/review.
    baseline_ids = video.np.flatnonzero(candidates[:, 4] > video.np.float32(.25)).tolist()
    return dict(boxes=boxes[survivors].tolist(), candidate_ids=[ids[i] for i in survivors],
                threshold_removed=[i for i in baseline_ids if i not in ids], nms_removed=removed)


def metrics(boxes, objects):
    matched = set()
    tp = 0
    for box in sorted(boxes, key=lambda b: -b[4]):
        possible = [(iou(box[:4], obj['xyxy']), i) for i, obj in enumerate(objects)
                    if i not in matched and obj['class_id'] == int(box[5])]
        overlap, index = max(possible, default=(0., -1))
        if overlap >= .5:
            matched.add(index); tp += 1
    fp, fn = len(boxes)-tp, len(objects)-tp
    return dict(tp=tp, fp=fp, fn=fn)


def summarize_counts(counts):
    tp, fp, fn = (sum(item[k] for item in counts) for k in ('tp', 'fp', 'fn'))
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def labels_for(labels, frame_id, digest, shape):
    items = [item for item in labels.get('frames', []) if item['frame_id'] == frame_id]
    if not items: return None
    if len(items) != 1: raise ValueError('Duplicate manual label frame')
    item = items[0]
    if item.get('reviewed') is not True: return None
    if item.get('source_png_sha256') != digest:
        raise ValueError('Manual labels do not match saved source PNG')
    height, width = shape[:2]
    for obj in item['objects']:
        coordinates = obj['xyxy']
        if (type(obj['class_id']) is not int or not 0 <= obj['class_id'] < 80 or len(coordinates) != 4 or
                not 0 <= coordinates[0] < coordinates[2] <= width or
                not 0 <= coordinates[1] < coordinates[3] <= height):
            raise ValueError('Invalid manual class/box in original pixel coordinates')
    return item['objects']


def analyze(directory, output, frame_ids=None, labels_path=None, video=None):
    video = video or dependencies()
    if output.exists(): raise ValueError('Use a new accuracy output directory')
    if (directory/'report.json').is_file():
        records = json.loads((directory/'report.json').read_text(encoding='utf-8'))['results']
    else:
        records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('result-*.json'))]
        # The previously exported original-PyTorch first frame keeps hashes
        # in provenance.json rather than its coordinate/detection descriptor.
        if len(records) == 1 and (directory/'provenance.json').is_file():
            provenance = json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
            if provenance.get('frame_id') == records[0]['frame_id']:
                for key in ('input_sha256', 'output_sha256'):
                    if key not in records[0] and key in provenance: records[0][key] = provenance[key]
        # The previously exported original-PyTorch first frame keeps hashes
        # in provenance.json rather than its coordinate/detection descriptor.
        if len(records) == 1 and (directory/'provenance.json').is_file():
            provenance = json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
            if provenance.get('frame_id') == records[0]['frame_id']:
                for key in ('input_sha256', 'output_sha256'):
                    if key not in records[0] and key in provenance: records[0][key] = provenance[key]
    if not records: raise ValueError('No completed saved frames')
    by_id = {r['frame_id']: r for r in records}
    if len(by_id) != len(records): raise ValueError('Duplicate saved frame IDs')
    ids = frame_ids if frame_ids is not None else sorted({records[i]['frame_id'] for i in (0, len(records)//2, len(records)-1)})
    if not ids or len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
        raise ValueError('Choose unique completed frame IDs')
    labels = json.loads(labels_path.read_text(encoding='utf-8')) if labels_path else {}
    if labels and labels.get('coordinates') != 'original_pixels_xyxy':
        raise ValueError('Manual labels must use original_pixels_xyxy')
    output.mkdir(parents=True)
    results, templates = [], []
    for frame_id in ids:
        record = by_id[frame_id]
        source = directory/'source-{:06d}.png'.format(frame_id)
        raw = (directory/'candidates-{:06d}.bin'.format(frame_id)).read_bytes()
        if len(raw) != 7200 or hashlib.sha256(raw).hexdigest() != record['output_sha256']:
            raise ValueError('Candidate size/SHA mismatch at frame {}'.format(frame_id))
        image = video.read_image(source)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if record.get('source_png_sha256') and source_hash != record['source_png_sha256']:
            raise ValueError('Source PNG SHA mismatch')
        tensor, transform = video.prepare_input(image)
        if (transform != record['transform'] or
                hashlib.sha256(tensor.astype('<f4', copy=False).tobytes()).hexdigest() != record['input_sha256']):
            raise ValueError('Source/preprocessing differs from saved submitted input')
        candidates = video.np.frombuffer(raw, dtype='<f4').reshape(300, 6)
        validate_candidates(video, candidates)
        objects = labels_for(labels, frame_id, source_hash, image.shape)
        templates.append(dict(frame_id=frame_id, source_png_sha256=source_hash, reviewed=False, objects=[]))
        result = dict(frame_id=frame_id, input_sha256=record['input_sha256'], output_sha256=record['output_sha256'],
                      labeled=objects is not None, variants={})
        panels = []
        for name, confidence, nms in VARIANTS:
            variant = select(video, candidates, transform, confidence, nms)
            variant.update(confidence=confidence, nms_iou=nms)
            if objects is not None: variant['metrics'] = metrics(variant['boxes'], objects)
            result['variants'][name] = variant
            boxes = video.np.asarray(variant['boxes'], dtype=video.np.float32).reshape(-1, 6)
            drawn = video.annotate(image, boxes)
            save_png(video, output/'frame-{:06d}-{}.png'.format(frame_id, name), drawn)
            panel = video.cv2.resize(drawn, (800, max(1, round(image.shape[0]*800/image.shape[1]))))
            panel = video.cv2.copyMakeBorder(panel, 40, 0, 0, 0, video.cv2.BORDER_CONSTANT)
            text = '{} conf>{:.2f} NMS={} boxes={}'.format(name, confidence, nms, len(boxes))
            video.cv2.putText(panel, text, (8, 26), video.cv2.FONT_HERSHEY_SIMPLEX, .55,
                              (255, 255, 255), 1, video.cv2.LINE_AA)
            panels.append(panel)
        sheet = video.np.concatenate([video.np.concatenate(panels[:2], axis=1),
                                      video.np.concatenate(panels[2:], axis=1)], axis=0)
        save_png(video, output/'comparison-{:06d}.png'.format(frame_id), sheet)
        results.append(result)
    aggregate = {name: summarize_counts([r['variants'][name]['metrics'] for r in results if r['labeled']])
                 for name, _, _ in VARIANTS} if any(r['labeled'] for r in results) else None
    report = dict(version=1, source_directory=str(directory.resolve()), frames=results,
                  labeled_frames=sum(r['labeled'] for r in results), metrics=aggregate,
                  note='Complete manual labels required; metrics are IoU50 precision/recall, not mAP or proof of general accuracy.')
    save_json(output/'accuracy-summary.json', report)
    save_json(output/'labels-template.json', dict(coordinates='original_pixels_xyxy', frames=templates))
    lines = ['ACCURACY REVIEW: {} frames; labeled={}'.format(len(results), report['labeled_frames'])]
    for name, confidence, nms in VARIANTS:
        kept = sum(len(r['variants'][name]['boxes']) for r in results)
        removed = sum(len(r['variants'][name]['nms_removed']) for r in results)
        lines.append('{}: boxes={} NMS-removed={}'.format(name, kept, removed))
        if aggregate: lines.append('  TP={tp} FP={fp} FN={fn} P={precision} R={recall}'.format(**aggregate[name]))
    lines += ['No deployment parameters changed. Inspect missed real targets as well.',
              'PNG comparisons and exact removed-candidate IDs are saved.',
              'Without complete manual labels, no accuracy improvement is claimed.']
    text = '\n'.join(line[:80] for line in lines)+'\n'
    (output/'accuracy-report.txt').write_text(text, encoding='utf-8')
    print(text, end=''); print('Output:', output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', help='Comma-separated completed frame IDs; default first/middle/last')
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    analyze(args.directory, args.output, [int(i) for i in args.frames.split(',')] if args.frames else None, args.labels)


if __name__ == '__main__': main()
