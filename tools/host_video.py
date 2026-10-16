"""Python 3.8+ sequential video -> existing PCIe model -> PNG/display/MP4.

--check tests host video processing only; it never opens the board transport.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolo26_pc'))


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def dependencies():
    try:
        import video_demo as video
        return video
    except (ImportError, OSError) as exc:
        raise RuntimeError('Video dependencies unavailable: {}. Use the same Python with '
                           'yolo26_pc/requirements-board.txt; no board action was taken.'.format(exc)) from exc


def save_png(video, path, image):
    ok, encoded = video.cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG encode failed: ' + str(path))
    encoded.tofile(str(path))


def validate_candidates(video, candidates):
    np = video.np
    if candidates.shape != (300, 6) or candidates.dtype != np.float32:
        raise ValueError('Expected [300,6] FP32 candidates')
    classes = candidates[:, 5]
    if (not np.isfinite(candidates).all() or np.any(classes != np.rint(classes)) or
            np.any((classes < 0) | (classes >= 80)) or
            np.any((candidates[:, 4] < 0) | (candidates[:, 4] > 1)) or
            np.any(candidates[:, 2:4] < candidates[:, :2])):
        raise ValueError('Invalid candidate values, classes, scores or coordinate order')


def writer(video, path, fps, image):
    h, w = image.shape[:2]
    if h % 2 or w % 2:
        raise ValueError('MP4 frame dimensions must be even')
    out = video.cv2.VideoWriter(str(path), video.cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not out.isOpened():
        out.release()
        raise RuntimeError('Cannot encode MP4: ' + str(path))
    return out


def even_frame(video, image):
    h, w = image.shape[:2]
    return video.cv2.copyMakeBorder(image, 0, h % 2, 0, w % 2, video.cv2.BORDER_CONSTANT)


def verify_movie(video, path, expected, shape):
    cap = video.cv2.VideoCapture(str(path))
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError('Cannot reopen recorded MP4: ' + str(path))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != shape:
                raise ValueError('Recorded MP4 dimensions changed')
            count += 1
    finally:
        cap.release()
    if count != expected:
        raise ValueError('Recorded MP4 frame count {} != {}'.format(count, expected))
    return count


def check_host(video, source, output, display=False):
    """Exercise real decode, preprocessing, postprocessing and encoders locally."""
    cv, np = video.cv2, video.np
    started = time.monotonic()
    cap = cv.VideoCapture(str(source))
    try:
        ok, frame = cap.read()
        fps = cap.get(cv.CAP_PROP_FPS)
    finally:
        cap.release()
    if not ok:
        raise ValueError('Cannot decode the first video frame: ' + str(source))
    tensor, transform = video.prepare_input(frame)
    if (tensor.shape != (3, 416, 416) or tensor.dtype != np.float32 or
            tensor.nbytes != 2076672 or not np.isfinite(tensor).all() or
            tensor.min() < 0 or tensor.max() > 1):
        raise ValueError('Unexpected preprocessed video input')
    # Check the existing fixed sample, not synthetic detections, through drawing.
    manifest = json.loads((ROOT / 'reference/prepared/manifest.json').read_text(encoding='utf-8'))
    sample = manifest['samples']['bus']
    bus = video.read_image(ROOT / sample['source'])
    bus_input, bus_transform = video.prepare_input(bus)
    actual_hash = hashlib.sha256(bus_input.astype('<f4', copy=False).tobytes()).hexdigest()
    if actual_hash != sample['input_sha256']:
        raise ValueError('Host BUS preprocessing differs from frozen input; SHA256=' + actual_hash)
    raw = (ROOT / sample['reference']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sample['reference_sha256']:
        raise ValueError('Frozen candidate SHA256 mismatch')
    candidates = np.frombuffer(raw, dtype='<f4', offset=32).reshape(300, 6)
    validate_candidates(video, candidates)
    boxes = video.postprocess(candidates, bus_transform, .25)
    drawn = video.annotate(bus, boxes)
    save_png(video, output / 'check-bus.png', drawn)
    if not np.array_equal(video.read_image(output / 'check-bus.png'), drawn):
        raise ValueError('PNG round trip differs')
    frame = even_frame(video, frame)
    movie = writer(video, output / 'check.mp4', 5, frame)
    try:
        for _ in range(3):
            movie.write(frame)
    finally:
        movie.release()
    verify_movie(video, output / 'check.mp4', 3, frame.shape)
    if display:
        if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
            raise RuntimeError('DISPLAY is empty. Use a graphical SSH/desktop session or --no-display.')
        # A broken Qt/X11 setup may abort the process; isolate that GUI probe.
        code = ('import cv2,numpy as np; cv2.namedWindow("YOLO display check"); '
                'cv2.imshow("YOLO display check",np.zeros((100,200,3),np.uint8)); '
                'cv2.waitKey(100); cv2.destroyAllWindows()')
        probe = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
        if probe.returncode:
            raise RuntimeError('GUI check failed before board access: ' + probe.stderr[-800:])
    report = {'status': 'passed', 'python': sys.version.split()[0], 'numpy': np.__version__,
              'opencv': cv.__version__, 'source': str(source),
              'source_fps': fps if math.isfinite(fps) and fps > 0 else None,
              'bus_preprocess_exact': True, 'bus_boxes': len(boxes), 'mp4_frames': 3,
              'display_tested': display, 'elapsed_seconds': time.monotonic() - started}
    save_json(output / 'host-check.json', report)
    print('HOST VIDEO CHECK: passed (display {})'.format('tested' if display else 'not tested'), flush=True)
    print('Python {} NumPy {} OpenCV {}'.format(report['python'], report['numpy'], report['opencv']), flush=True)
    return report


def run(args, output):
    report = {'status': 'starting', 'source': str(args.source), 'results': [],
              'transport': 'PCIe DDR', 'policy': 'sequential; no skipped/repeated frames'}
    started = time.monotonic()
    stop = threading.Event()
    backend = [None]
    progress = ['Preparing host']
    pool = None
    cap = None
    movie = None
    diagnostic = None
    video = None
    window = 'YOLO PCIe: completed frame (Q/Esc exits)'
    window_open = False
    movie_shape = None
    try:
        video = dependencies()
        check = check_host(video, args.source, output, display=not args.no_display)
        if args.check:
            report.update(status='completed', host_check=check)
            return report
        if args.load:
            subprocess.run(['bash', str(ROOT / 'tools/load_board.sh'), '--reset-dir', args.reset_dir,
                            '--pbcopy', args.pbcopy], cwd=ROOT, check=True)
        cv, np = video.cv2, video.np
        fps = args.playback_fps or check['source_fps'] or 25
        report.update(playback_fps=fps, host_check=check)
        from board_diagnostics import DiagnosticLog
        from pcie_backend import PCIeBackend
        diagnostic = DiagnosticLog(output)
        def infer(tensor):
            if backend[0] is None:
                backend[0] = PCIeBackend(args.pbcopy, args.pbload, stop, args.infer_timeout,
                                          lambda message: progress.__setitem__(0, message), diagnostic.emit,
                                          tool_timeout=args.tool_timeout)
                report['board'] = backend[0].info
            return backend[0].infer(tensor)
        cap = cv.VideoCapture(str(args.source))
        if not cap.isOpened():
            raise ValueError('Video cannot be reopened')
        if not args.no_display:
            cv.namedWindow(window, cv.WINDOW_NORMAL)
            window_open = True
        pool = ThreadPoolExecutor(max_workers=1)
        frame_id = 0
        last_display = np.zeros((360, 640, 3), dtype=np.uint8)
        while not stop.is_set():
            decode_start = time.monotonic()
            ok, image = cap.read()
            if not ok:
                declared = cap.get(cv.CAP_PROP_FRAME_COUNT)
                if frame_id == 0 or (math.isfinite(declared) and declared > frame_id):
                    raise ValueError('Video ended before its declared frame count at frame {}'.format(frame_id))
                report.update(status='completed', completion_reason='end of video')
                break
            tensor, transform = video.prepare_input(image)
            prepared = time.monotonic()
            input_hash = hashlib.sha256(tensor.astype('<f4', copy=False).tobytes()).hexdigest()
            save_png(video, output / 'source-{:06d}.png'.format(frame_id), image)
            progress[0] = 'Processing frame {}'.format(frame_id)
            future = pool.submit(infer, tensor)
            while not future.done():
                if not args.no_display:
                    cv.setWindowTitle(window, 'YOLO PCIe: ' + progress[0])
                    cv.imshow(window, last_display)
                    if cv.waitKey(20) & 255 in (27, ord('q'), ord('Q')) or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                        stop.set()
                        break
                else:
                    stop.wait(.02)
            if stop.is_set():
                break
            candidates = future.result()
            received = time.monotonic()
            validate_candidates(video, candidates)
            raw = candidates.astype('<f4', copy=False).tobytes()
            boxes = video.postprocess(candidates, transform, .25)
            last_display = video.annotate(image, boxes)
            post_done = time.monotonic()
            save_png(video, output / 'detected-{:06d}.png'.format(frame_id), last_display)
            (output / 'candidates-{:06d}.bin'.format(frame_id)).write_bytes(raw)
            encoded_frame = even_frame(video, last_display)
            if movie is None:
                movie = writer(video, output / 'demo.mp4', fps, encoded_frame)
                movie_shape = encoded_frame.shape
            if encoded_frame.shape != movie_shape:
                raise ValueError('Video dimensions changed during decoding')
            movie.write(encoded_frame)
            record = {'frame_id': frame_id, 'source_frame': frame_id,
                      'source_seconds': frame_id / check['source_fps'] if check['source_fps'] else None,
                      'input_sha256': input_hash, 'output_sha256': hashlib.sha256(raw).hexdigest(),
                      'transform': transform, 'detections': boxes.tolist(),
                      'prepare_seconds': prepared - decode_start, 'infer_seconds': received - prepared,
                      'postprocess_seconds': post_done - received,
                      'pcie_timing': backend[0].last_timing,
                      'elapsed_seconds': time.monotonic() - decode_start}
            save_json(output / 'result-{:06d}.json'.format(frame_id), record)
            report['results'].append(record)
            save_json(output / 'report.json', dict(report, status='running'))
            print('FRAME {} DONE: {} boxes, {:.2f}s; detected-{:06d}.png'.format(
                frame_id, len(boxes), record['elapsed_seconds'], frame_id), flush=True)
            if not args.no_display:
                cv.setWindowTitle(window, 'YOLO PCIe: frame {} complete'.format(frame_id))
                cv.imshow(window, last_display)
                if cv.waitKey(30) & 255 in (27, ord('q'), ord('Q')) or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                    stop.set()
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                report.update(status='completed', completion_reason='requested frame limit')
                break
        if report['status'] == 'starting':
            report['status'] = 'stopped'
    except KeyboardInterrupt:
        report.update(status='stopped', error='Interrupted by user')
    except Exception as exc:
        report.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
        if diagnostic is not None:
            diagnostic.exception(exc)
    finally:
        stop.set()
        if pool is not None:
            pool.shutdown(wait=True)
        try:
            if backend[0] is not None:
                backend[0].close()
        except Exception as exc:
            report.update(status='failed', cleanup_error=str(exc))
        if cap is not None:
            cap.release()
        if movie is not None:
            movie.release()
            try:
                report['encoded_frames'] = verify_movie(video, output / 'demo.mp4', len(report['results']), movie_shape)
            except Exception as exc:
                report.update(status='failed', recording_error=str(exc))
        if window_open:
            video.cv2.destroyAllWindows()
        if diagnostic is not None:
            diagnostic.close()
        report['wall_seconds'] = time.monotonic() - started
        save_json(output / 'report.json', report)
        print('{} {}: {} frames'.format('HOST CHECK' if args.check else 'VIDEO',
                                       report['status'].upper(), len(report['results'])), flush=True)
        for field in ('error', 'cleanup_error', 'recording_error'):
            if report.get(field):
                print(field + ': ' + report[field], flush=True)
        print('Output: ' + str(output), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'yolo26_pc/samples/pexels-3796613.mp4')
    parser.add_argument('--check', action='store_true', help='Host video check only; no board access')
    parser.add_argument('--load', action='store_true', help='After host checks, reset/load normal model firmware once')
    parser.add_argument('--reset-dir', default='/rv/fpga/bin')
    parser.add_argument('--no-display', action='store_true', help='Save PNG/MP4 without opening a window')
    parser.add_argument('--max-frames', type=int, default=3, help='Default 3-frame smoke run; 0 = entire video')
    parser.add_argument('--playback-fps', type=float, default=0, help='MP4 playback FPS; 0 = source FPS')
    parser.add_argument('--pbcopy', default='/rv/fpga/bin/pbcopy')
    parser.add_argument('--pbload', default='/rv/fpga/bin/pbload')
    parser.add_argument('--infer-timeout', type=float, default=900)
    parser.add_argument('--tool-timeout', type=float, default=30)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.max_frames < 0 or not math.isfinite(args.playback_fps) or args.playback_fps < 0 or min(args.infer_timeout, args.tool_timeout) <= 0:
        parser.error('Invalid frame count, FPS or timeout')
    base = ROOT / 'yolo26_pc/outputs/host-video'
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        base.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='run-', dir=str(base)))
    report = run(args, output)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
