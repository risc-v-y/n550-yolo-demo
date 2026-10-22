"""Acceptance rules shared by tuning, video validation and offline reports.

The original finite-RTL golden remains the reference. A changed FP32 result
is accepted only when the existing drawing inputs (including order) agree.
No numerical tolerance is used to turn a changed detection into a pass.
"""
import math
import struct
import textwrap


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def candidates(raw):
    if len(raw) != 7200:
        raise ValueError('Expected 300 x 6 FP32 candidates')
    result = list(struct.iter_unpack('<6f', raw))
    for x1, y1, x2, y2, score, category in result:
        if (not all(map(math.isfinite, (x1, y1, x2, y2, score, category))) or
                not 0 <= score <= 1 or category != int(category) or
                not 0 <= category < 80 or x2 < x1 or y2 < y1):
            raise ValueError('Invalid detection values')
    return result


def drawing_signature(raw, transform):
    height, width = transform['original_hw']
    left, top = transform['padding_ltrb'][:2]
    scale = f32(transform['scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid letterbox scale')
    result = []
    for box in candidates(raw):
        if box[4] <= f32(.25):
            continue
        xy = [int(round(min(float(limit), max(0., f32(f32(value - pad) / scale)))))
              for value, pad, limit in zip(box[:4], (left, top, left, top),
                                          (width, height, width, height))]
        result.append([int(box[5]), *xy, format(box[4], '.2f')])
    return result


def compare_display(actual, golden, transform):
    a, b = candidates(actual), candidates(golden)
    aw, bw = struct.unpack('<1800I', actual), struct.unpack('<1800I', golden)
    sa, sb = drawing_signature(actual, transform), drawing_signature(golden, transform)
    changed = sum(x != y for x, y in zip(aw, bw))
    return {'display_equal': sa == sb, 'different_words': changed,
            'numeric_status': 'EXACT' if not changed else 'DIFFERENT',
            'actual_detections': len(sa), 'golden_detections': len(sb),
            'actual_signature': sa, 'golden_signature': sb,
            'max_coordinate_difference': max(abs(x[i]-y[i]) for x, y in zip(a, b) for i in range(4)),
            'max_score_difference': max(abs(x[4]-y[4]) for x, y in zip(a, b))}


def repeatable_improvement(baseline, candidate, metric='model_cycles'):
    """Two or more matched complete passes must each improve their total.

    Samples are identified by original input hashes, not frame numbers alone.
    An interrupted run, numerical failure or missing measurement cannot win.
    """
    if len(baseline) < 2 or len(candidate) != len(baseline):
        return False
    for old, new in zip(baseline, candidate):
        if not old or len(old) != len(new):
            return False
        for a, b in zip(old, new):
            if (a.get('input_sha256') != b.get('input_sha256') or
                    not a.get('display_equal') or not b.get('display_equal')):
                return False
            for record in (a, b):
                value = record.get(metric)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    return False
        if sum(x[metric] for x in new) >= sum(x[metric] for x in old):
            return False
    return True


def report_pages(report):
    stages = report.get('stages', {})
    frames = report.get('frames', [])
    failure=report.get('failure',{})
    error_lines=textwrap.wrap('Error: '+str(report.get('error','none')),width=80)[:3]
    failure_lines=[]
    if failure:
        failure_lines=['Failure stage: '+str(failure.get('stage','unknown')),
            'Pending frame_id={} active_frame_id={} slot={}'.format(failure.get('frame_id'),
                failure.get('active_frame_id'),failure.get('slot'))]
        if 'different_offsets' in failure:
            failure_lines.append('Different header offsets: '+str(failure['different_offsets']))
        for label,key in (('EXP','expected_hex'),('GOT','actual_hex')):
            raw=failure.get(key,'')
            failure_lines+=['{}[{}:{}]: {}'.format(label,i//2,(i+64)//2,raw[i:i+64]) for i in range(0,len(raw),64)]
    old=[r for group in report.get('baseline',[]) for r in group]
    new=[r for group in report.get('selected_measurements',[]) for r in group]
    ratio=(sum(r['model_cycles'] for r in old)/len(old))/(sum(r['model_cycles'] for r in new)/len(new)) if old and new else None
    profiles=[r.get('profile',{}) for r in new]
    measured=[p for p in profiles if p.get('phases')]
    phase_lines=[]
    if measured:
        phase_lines=['{}: {:.0f} cycles'.format(k,sum(p['phases'][k]['ticks'] for p in measured)/len(measured))
                     for k in measured[0]['phases']]
        details=measured[0].get('nodes_detail',[])
        nodes=[]
        for row in details:
            ident=row['node']
            total=sum(next(x['total'] for x in p['nodes_detail'] if x['node']==ident) for p in measured)/len(measured)
            nodes.append((total,ident))
        phase_lines+=['Slow node {}: {:.0f} cycles'.format(n,t) for t,n in sorted(nodes,reverse=True)[:5]]
    pages = [
        ['OPTIMIZATION PAGE=1/3', 'Status: ' + report.get('status', 'unknown'),
         'Stage: {} | upload={}'.format(report.get('stage','unknown'),report.get('upload_mode','unknown')),
         'Completed={} | golden-checked={}'.format(report.get('completed_frames',len(frames)),len(frames)),
         'Display agreement: '+('{}/{}'.format(sum(bool(x.get('display_equal')) for x in frames),len(frames)) if frames else 'NOT CHECKED'),
         'Numerically exact: '+('{}/{}'.format(sum(x.get('different_words')==0 for x in frames),len(frames)) if frames else 'NOT CHECKED'),
         'Firmware: ' + report.get('firmware_sha256', 'unknown')[:16],
         'Selected: ' + str(report.get('selected', {}))[:65],
         'Model baseline/current: '+('{:.3f}x'.format(ratio) if ratio else 'pending board measurements'),
         'Accepted serial interval seconds: '+('{:.4f}'.format(report['baseline_interval_seconds'])
                                               if report.get('baseline_interval_seconds') else 'not recorded'),
         'Stage seconds below: mean / max / count; startup excluded.'] +
        ['{}: {:.4f} / {:.4f} / {}'.format(name, sum(values)/len(values), max(values), len(values))
         for name, values in stages.items() if values][:11],
        ['OPTIMIZATION PAGE=2/3','Operator/data-movement timing from matched full-profile passes.',
         'Load / host preflight / OPEN seconds: {} / {} / {}'.format(
             *['{:.3f}'.format(report[k]) if isinstance(report.get(k),(int,float)) else 'not recorded'
               for k in ('firmware_load_seconds','cold_start_seconds','open_configure_seconds')])]+
        (phase_lines or ['Not collected yet.'])+
        ['Pipeline {}: {:.0f} cycles/frame'.format(k,sum(v)/len(v)) for k,v in report.get('cycles',{}).items() if v]+
        ['Candidate measurements: '+str(len(report.get('candidates',[]))),
         'Full node timing: '+report.get('full_profile','pending'),
         'Fusion retains logical node IDs; skipped work has near-zero cycles.'],
        ['OPTIMIZATION PAGE=3/3', 'Original golden retained; no numerical tolerance.',
         'Compared: count, class, integer box, 2-decimal score, draw order.',
         'Final changed FP32 words: '+(str(sum(r.get('different_words',0) for r in frames)) if frames else 'NOT CHECKED'),
         'MP4 frames: '+str(report.get('recorded_frames','pending'))]+error_lines+failure_lines+
        ['Full errors/headers in report.json and video/request-readback.json.',
         'Intervals include scheduling; overlapping stages must not be added.',
         'Display submission does not measure remote screen presentation.',
         'This validates the sampled video, not all possible inputs.']]
    return ['\n'.join(line[:80] for line in page[:22]) for page in pages]
