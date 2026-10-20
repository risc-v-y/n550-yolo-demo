"""Offline video evidence checks, manual flags and three screenshot-sized pages.

Never accesses the board. Software golden inference runs on the host CPU.
Original report/results stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def flag_frame(directory, frame_id, note='Marked for visual review'):
    """Bind a human flag to a completed frame, never the frame in flight."""
    report = read_json(directory / 'report.json')
    matches = [r for r in report['results'] if r['frame_id'] == frame_id]
    if len(matches) != 1:
        raise ValueError('Frame {} is not uniquely completed; cannot flag it'.format(frame_id))
    path = directory / 'flags.json'
    flags = read_json(path) if path.exists() else {'version': 1, 'frames': []}
    flags['frames'] = [r for r in flags['frames'] if r['frame_id'] != frame_id]
    flags['frames'].append({'frame_id': frame_id, 'note': note,
                            'input_sha256': matches[0]['input_sha256'],
                            'output_sha256': matches[0]['output_sha256']})
    save_json(path, flags)


def preview(video, directory, report, flags):
    records = report['results']
    if not records:
        return None
    ids = [r['frame_id'] for r in records]
    selected = []
    for item in [{'frame_id': ids[i]} for i in (0, len(ids) // 2, len(ids) - 1)] + flags:
        frame_id = item['frame_id']
        if frame_id in ids and frame_id not in selected:
            selected.append(frame_id)
    # All flagged frames remain as individual PNGs; the overview stays readable.
    selected = selected[:6]
    cv, np = video.cv2, video.np
    sheet = np.zeros((34 + 266 * ((len(selected) + 1) // 2), 800, 3), dtype=np.uint8)
    cv.putText(sheet, 'Run {} | first/middle/last + flags'.format(report.get('id', directory.name))[:95],
               (8, 24), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
    flagged = {r['frame_id'] for r in flags}
    for i, frame_id in enumerate(selected):
        x, y = (i % 2) * 400, 34 + (i // 2) * 266
        label = 'frame {}{}'.format(frame_id, ' FLAGGED' if frame_id in flagged else '')
        cv.putText(sheet, label, (x + 8, y + 23), cv.FONT_HERSHEY_SIMPLEX, .6,
                   (0, 180, 255) if frame_id in flagged else (255, 255, 255), 1, cv.LINE_AA)
        try:
            image = video.read_image(directory / 'detected-{:06d}.png'.format(frame_id))
            scale = min(384 / image.shape[1], 228 / image.shape[0])
            image = cv.resize(image, (max(1, round(image.shape[1] * scale)),
                                      max(1, round(image.shape[0] * scale))))
            sheet[y + 32:y + 32 + image.shape[0], x + 8:x + 8 + image.shape[1]] = image
        except (ValueError, OSError, cv.error):
            cv.putText(sheet, 'PNG missing/unreadable', (x + 8, y + 70),
                       cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    ok, encoded = cv.imencode('.png', sheet)
    if not ok:
        raise ValueError('Preview PNG encoding failed')
    encoded.tofile(str(directory / 'preview.png'))
    return 'preview.png'


def analyze(directory, video):
    """Verify saved evidence, including old runs lacking PNG hash fields."""
    from host_video import even_frame, validate_candidates, verify_movie
    directory = directory.resolve()
    source = directory / 'report.json'
    report = read_json(source)
    records = report['results']
    issues, warnings = [], []
    def issue(code, detail, frame=None):
        issues.append({'code': code, 'frame': frame, 'detail': str(detail)})
    ids = [r.get('frame_id') for r in records]
    if ids != list(range(len(records))) or any(r.get('source_frame') != i for i, r in enumerate(records)):
        issue('sequence', 'Expected unique, consecutive source/frame IDs from zero')
    shapes, eligible = [], set()
    png_hashes_missing = False
    for record in records:
        frame_id = record.get('frame_id')
        try:
            if type(frame_id) is not int or frame_id < 0:
                raise ValueError('Invalid frame ID')
            if read_json(directory / 'result-{:06d}.json'.format(frame_id)) != record:
                raise ValueError('Per-frame JSON differs from report.json')
            raw = (directory / 'candidates-{:06d}.bin'.format(frame_id)).read_bytes()
            if len(raw) != 7200 or hashlib.sha256(raw).hexdigest() != record['output_sha256']:
                raise ValueError('Candidate length/SHA256 mismatch')
            candidates = video.np.frombuffer(raw, dtype='<f4').reshape(300, 6)
            validate_candidates(video, candidates)
            images = {}
            for kind in ('source', 'detected'):
                path = directory / '{}-{:06d}.png'.format(kind, frame_id)
                expected_hash = record.get(kind + '_png_sha256')
                if expected_hash and digest(path) != expected_hash:
                    raise ValueError(kind + ' PNG SHA256 mismatch')
                if not expected_hash:
                    png_hashes_missing = True
                images[kind] = video.read_image(path)
            tensor, transform = video.prepare_input(images['source'])
            if hashlib.sha256(tensor.astype('<f4', copy=False).tobytes()).hexdigest() != record['input_sha256']:
                raise ValueError('Saved source does not reproduce submitted input SHA256')
            if transform != record['transform']:
                raise ValueError('Coordinate mapping differs from saved source')
            boxes = video.postprocess(candidates, transform, .25)
            saved_boxes = video.np.asarray(record['detections'], dtype=video.np.float32).reshape(-1, 6)
            if not video.np.array_equal(boxes, saved_boxes):
                raise ValueError('Saved postprocessing differs from replay')
            drawn = video.annotate(images['source'], boxes)
            if not video.np.array_equal(drawn, images['detected']):
                raise ValueError('Detection PNG differs from postprocessing/drawing replay')
            shapes.append(even_frame(video, drawn).shape)
            eligible.add(frame_id)
        except (OSError, ValueError, TypeError, KeyError, OverflowError, video.cv2.error) as exc:
            issue('frame_evidence', exc, frame_id)
    if png_hashes_missing:
        warnings.append('Legacy PNG hashes absent; decoded images verified by replay')
    encoded = None
    if records:
        try:
            if len(shapes) != len(records) or len(set(shapes)) != 1:
                raise ValueError('Not all frame dimensions verified')
            encoded = verify_movie(video, directory / 'demo.mp4', len(records), shapes[0])
            if report.get('encoded_frames') != encoded:
                raise ValueError('Recorded MP4 count differs from report')
        except (OSError, ValueError, video.cv2.error) as exc:
            issue('movie', exc)
    if report.get('status') == 'completed' and not records and not report.get('host_check'):
        issue('empty_run', 'Completed inference run has no frames')
    if report.get('status') == 'completed' and records:
        limit = report.get('requested_frames')
        if report.get('completion_reason') == 'requested frame limit' and limit and len(records) != limit:
            issue('frame_limit', 'Completed count differs from requested frame limit')
    for field in ('error', 'cleanup_error', 'recording_error', 'report_error'):
        if report.get(field) and report.get('status') != 'stopped':
            issue('runtime_' + field, report[field], report.get('active_frame'))
    marks = directory / 'flags.json'
    flags = read_json(marks).get('frames', []) if marks.exists() else []
    by_id = {r['frame_id']: r for r in records}
    for flag in flags:
        item = by_id.get(flag['frame_id'], {})
        if any(flag.get(k) != item.get(k) for k in ('input_sha256', 'output_sha256')):
            issue('flag_binding', 'Flag no longer matches frame evidence', flag['frame_id'])
    state = report.get('status')
    board = report.get('last_board_diagnostic')
    board_status = report.get('last_board_status')
    log = directory / 'diagnostics.jsonl'
    if log.is_file() and (board is None or board_status is None):
        # A constructor/selftest failure can happen before the backend is assigned.
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                    if event.get('type') == 'board': board = event.get('diagnostic', board)
                    if event.get('type') == 'pcie_status': board_status = event.get('status', board_status)
                except ValueError:
                    warnings.append('An incomplete diagnostic log line was ignored')
    status = ('FAILED' if issues or state == 'failed' else
              'HOST_CHECK_ONLY' if state == 'completed' and not records else
              'PASSED' if state == 'completed' else 'INCOMPLETE')
    timings = {}
    for key, getter in (
            ('prepare', lambda r: r.get('prepare_seconds')),
            ('pcie_roundtrip', lambda r: r.get('pcie_timing', {}).get('run_roundtrip_seconds')),
            ('upload', lambda r: r.get('pcie_timing', {}).get('upload_seconds')),
            ('download', lambda r: r.get('pcie_timing', {}).get('download_seconds')),
            ('postprocess', lambda r: r.get('postprocess_seconds')),
            ('frame_total', lambda r: r.get('elapsed_seconds'))):
        values = [getter(r) for r in records if isinstance(getter(r), (int, float))]
        timings[key] = {'mean': sum(values) / len(values), 'max': max(values)} if values else None
    from video_golden import verify_frames
    numeric = verify_frames(directory, records, eligible, video)
    overall = ('FAILED' if status == 'FAILED' or numeric['status'] in ('DIFFERENT', 'ERROR') else
               'PASSED' if status == 'PASSED' and numeric['status'] == 'PASSED' else 'INCOMPLETE')
    summary = {'version': 2, 'id': report.get('id', directory.name), 'directory': str(directory),
               'report_sha256': digest(source), 'functional': status, 'run_status': state,
               'overall': overall, 'numeric': numeric,
               'completion_reason': report.get('completion_reason'), 'stage': report.get('stage', 'legacy'),
               'active_frame': report.get('active_frame'), 'completed': len(records), 'encoded': encoded,
               'empty_detection_frames': sum(not r['detections'] for r in records),
               'source': report.get('source'), 'wall_seconds': report.get('wall_seconds'),
               'playback_fps': report.get('playback_fps'), 'timings': timings,
               'flags': flags, 'issues': issues, 'warnings': warnings,
               'board_status': board_status, 'board_diagnostic': board,
               'semantic_review': 'FLAGGED' if flags else 'NOT_CONFIRMED',
               'preview': preview(video, directory, report, flags)}
    save_json(directory / 'video-summary.json', summary)
    for page in (1, 2, 3):
        (directory / 'video-page-{}.txt'.format(page)).write_text(render(summary, page), encoding='utf-8')
    return summary


def render(summary, page):
    numeric = summary['numeric']
    lines = ['VIDEO REPORT {} PAGE={}/3'.format(summary['id'], page),
             'CHECK: {} | FUNCTIONAL: {} | run={}'.format(summary['overall'],summary['functional'],summary['run_status']),
             'NUMERICAL: {} exact={}/{} checked={} diff={} error={}'.format(
                 numeric['status'],numeric['exact_frames'],numeric['expected_frames'],
                 numeric['checked_frames'],numeric['different_frames'],numeric['error_count'])]
    if page == 1:
        lines += ['Completed={} MP4={} empty-detection={}'.format(summary['completed'], summary['encoded'],
                                                                summary['empty_detection_frames']),
                  'Reason: ' + str(summary['completion_reason']),
                  'Last stage={} active-frame={}'.format(summary['stage'], summary['active_frame']),
                  'Source: ' + Path(summary.get('source') or '?').name,
                  'Wall seconds: {}'.format(summary['wall_seconds']),
                  'Golden check seconds: {:.2f}'.format(numeric['elapsed_seconds']),
                  'MP4 playback FPS: {} (not inference FPS)'.format(summary['playback_fps']),
                  'Visual review: {} | flagged={}'.format(summary['semantic_review'], len(summary['flags'])),
                  'Evidence checks: IDs, input/output hashes, mapping, drawing, MP4.',
                  'Transport: runtime PCIe session/frame/CRC validation.',
                  'Golden: independent 334-node forward, all 1800 FP32 words/frame.',
                  'Rule: ' + numeric['numeric_rule'],
                  'No tolerance; numeric match does not prove detection quality.']
        lines += summary['warnings'][:2]
    elif page == 2:
        lines += ['Saved diagnostics: diagnostics.jsonl; full details: video-summary.json',
                  'Issues={} | manually flagged={}'.format(len(summary['issues']), len(summary['flags']))]
        board = summary.get('board_diagnostic') or {}
        status = summary.get('board_status') or {}
        if status:
            lines.append('PCIe state={} frame={} error={}'.format(status.get('state'),status.get('frame'),status.get('error')))
        if board:
            lines += ['Last board frame={} stage={} node={} op={} phase={}'.format(
                board.get('frame'),board.get('stage_name'),board.get('node'),board.get('op_name'),board.get('phase_name')),
                      'Board error={} detail={} trap={} pc={} value={}'.format(
                board.get('error'),board.get('detail'),board.get('trap_cause'),
                hex(board.get('trap_pc',0)),hex(board.get('trap_value',0)))]
        for item in summary['issues'][:1]:
            lines += ['ERROR frame={} {}'.format(item['frame'], item['code']), item['detail'][:100]]
        exceptions = [r for r in numeric['frames'] if r['status'] != 'EXACT']
        for item in exceptions[:2]:
            lines.append('NUM frame={} {} different={}/1800'.format(
                item['frame_id'],item['status'],item.get('different_elements','?')))
            first = item.get('first_difference')
            if first:
                lines += ['candidate={} field={} got={} golden={}'.format(
                    first['candidate'],first['field'],first['actual_bits'],first['golden_bits']),
                          'max coord={:.8g} score={:.8g} class-diff={}'.format(
                    item['max_coordinate_difference'],item['max_score_difference'],item['field_differences']['class'])]
            elif item.get('error'): lines.append(item['error'][:100])
        if not summary['issues'] and not exceptions:
            lines.append('No reported errors; completed-frame golden comparison is bit-exact.'
                         if numeric['status']=='PASSED' else 'Numerical check not complete; no numeric pass claimed.')
        for item in summary['flags'][:1]:
            lines.append('FLAG frame={}: {}'.format(item['frame_id'], item['note'][:60]))
        lines += ['All issue/flag details stay on host; summary may show only a subset.',
                  'Source frame IDs start at 0; PCIe/board frame IDs start at 1.',
                  'UART text is not captured automatically.']
    elif page == 3:
        lines += ['Timing in seconds: mean / max (completed frames only)']
        for key, values in summary['timings'].items():
            lines.append('{}: {}'.format(key, '{:.3f} / {:.3f}'.format(values['mean'], values['max'])
                                         if values else 'not recorded'))
        lines += ['PCIe roundtrip includes polling/tool overhead; not pure AMU time.',
                  'SSH/X11 can increase frame/display time.',
                  'Screenshot these 3 pages and preview.png.',
                  'For wrong boxes: send flagged detected-NNNNNN.png + frame ID.',
                  'Originals: source-NNNNNN.png; raw output: candidates-NNNNNN.bin.',
                  'Golden cache is verified; menu 5 resumes unfinished frames.',
                  'Offline recheck does not access/reset/re-run the board.']
    else:
        raise ValueError('Page must be 1, 2 or 3')
    wrapped = []
    for line in lines:
        part, columns = '', 0
        for char in str(line):
            if not char.isprintable(): char = ' '
            width = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ('W','F') else 1
            if columns + width > 80:
                wrapped.append(part)
                part, columns = '', 0
            part += char
            columns += width
        wrapped.append(part)
    # Error text can be arbitrarily long; full details remain in JSON, never hidden.
    if len(wrapped) > 22:
        wrapped = wrapped[:21] + ['More details: video-summary.json']
    return '\n'.join(wrapped) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--flag', type=int, action='append', default=[])
    parser.add_argument('--note', default='Marked for visual review')
    parser.add_argument('--page', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    from host_video import dependencies
    try:
        for frame_id in args.flag:
            flag_frame(args.directory, frame_id, args.note)
        summary = analyze(args.directory, dependencies())
        for page in (args.page,) if args.page else (1, 2, 3):
            print(render(summary, page), end='')
        return 0 if summary['overall'] == 'PASSED' else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('VIDEO REPORT ERROR: {}: {}'.format(type(exc).__name__, exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
