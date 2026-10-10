"""Live preview and asynchronous YOLO detection: PyTorch, QEMU TCP, board PCIe."""
import argparse
from datetime import datetime
import queue
import secrets
import socket
import struct
import subprocess
import threading
import time
import sys
from board_diagnostics import DiagnosticLog

from video_demo import (ROOT, RISCV, WORKSPACE, cv2, np, prepare_input, postprocess,
                        annotate, find_wsl, windows_to_wsl, stop_process, write_json)

HEADER = struct.Struct('<8sIIQII')


def receive(sock, size):
    data = bytearray()
    while len(data) < size:
        block = sock.recv(size - len(data))
        if not block:
            raise ConnectionError('QEMU disconnected before completing the message')
        data.extend(block)
    return bytes(data)


def panel(image, text):
    result = np.zeros((400, 640, 3), dtype=np.uint8)
    if image is not None:
        h, w = image.shape[:2]
        scale = min(640 / w, 360 / h)
        resized = cv2.resize(image, (round(w * scale), round(h * scale)))
        rh, rw = resized.shape[:2]
        result[:rh, :rw] = resized
    cv2.putText(result, text, (8, 385), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    return result


def save_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Cannot encode {path}')
    encoded.tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=('pc', 'qemu', 'pcie'), default='pc')
    parser.add_argument('--pbcopy', default='pbcopy', help='PCIe write executable')
    parser.add_argument('--pbload', default='pbload', help='PCIe read executable')
    parser.add_argument('--tool-timeout', type=float, default=30)
    parser.add_argument('--poll-interval', type=float, default=.1)
    parser.add_argument('--infer-timeout', type=float, default=900)
    parser.add_argument('--save-frames', action='store_true', help='Save diagnostic PNG/NPY files for every result')
    parser.add_argument('--board-trace', action='store_true', help='Log every board node start/end (adds UART traffic)')
    parser.add_argument('--check-intermediates', action='store_true', help='Check each board FP32 node output for NaN/Inf')
    parser.add_argument('--source', help='Loop a local video for repeatable validation')
    parser.add_argument('--port', type=int, default=5557)
    parser.add_argument('--distro', default='Ubuntu-24.04')
    parser.add_argument('--max-results', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    if args.backend == 'pcie' and min(args.infer_timeout,args.tool_timeout,args.poll_interval)<=0:
        parser.error('PCIe needs positive timeouts and poll interval')
    if not 1024 <= args.port <= 65535 or args.seconds < 0 or args.max_results < 0:
        parser.error('Invalid port, duration or result count')
    output = ROOT / 'outputs' / 'live-demo' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    diagnostics = DiagnosticLog(output)
    stop = threading.Event()
    lock = threading.Lock()
    state = {'latest': None, 'detected': None, 'status': f'Starting {args.backend}', 'error': None,
             'results': [], 'record_dropped': 0}
    recording = queue.Queue(maxsize=60)
    report = {'transport': 'in-process memory' if args.backend == 'pc' else 'TCP to QEMU UART',
              'backend': 'PyTorch CPU FP32' if args.backend == 'pc' else 'RVV+AMU FP16',
              'source': args.source or f'camera:{args.camera}'}
    if args.backend == 'pcie':
        report.update(transport='PCIe DDR via pbcopy/pbload',backend='N550 RVV+AMU FP16')
    connection = [None]
    process = None
    log = None

    def fail(exc):
        with lock:
            if state['error'] is None:
                state['error'] = str(exc)
        try:
            diagnostics.exception(exc)
        except OSError as log_error:
            print(f'Cannot save diagnostic: {log_error}; original error: {exc}',file=sys.stderr)
        finally:
            stop.set()

    def capture():
        cap = None
        try:
            cap = (cv2.VideoCapture(args.source) if args.source else
                   cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY))
            if not cap.isOpened() and not args.source and sys.platform == 'win32':
                cap.release()
                cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera/video; check camera access and other applications')
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / fps if args.source and 0 < fps < 240 else 0
            number = 0
            while not stop.is_set():
                before = time.monotonic()
                ok, image = cap.read()
                if not ok and args.source:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, image = cap.read()
                if not ok:
                    raise RuntimeError('Camera disconnected or frame read failed')
                item = (number, time.time(), image)
                with lock:
                    state['latest'] = item
                number += 1
                stop.wait(max(0, delay - (time.monotonic() - before)))
        except Exception as exc:
            fail(exc)
        finally:
            if cap is not None:
                cap.release()

    def recorder():
        writers = {}
        try:
            while not stop.is_set() or not recording.empty():
                try:
                    kind, image = recording.get(timeout=.1)
                except queue.Empty:
                    continue
                if kind not in writers:
                    h, w = image.shape[:2]
                    writer = cv2.VideoWriter(str(output / f'{kind}.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), 20, (w,h))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot create {kind}.mp4')
                    writers[kind] = writer
                writers[kind].write(image)
        except Exception as exc:
            fail(exc)
        finally:
            for writer in writers.values():
                writer.release()

    def inference():
        sock = None
        backend = None
        try:
            if args.backend == 'pc':
                from pc_backend import PCBackend
                backend = PCBackend()
            elif args.backend == 'pcie':
                from pcie_backend import PCIeBackend
                def progress(message):
                    with lock:
                        state['status'] = message
                backend = PCIeBackend(args.pbcopy,args.pbload,stop,args.infer_timeout,progress,
                                      diagnostics.emit,tool_timeout=args.tool_timeout,
                                      poll_interval=args.poll_interval,check_finite=args.check_intermediates,
                                      trace=args.board_trace)
                report['board'] = backend.info
            deadline = time.monotonic() + 180
            while args.backend == 'qemu' and not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError('QEMU build/start failed; see qemu.log')
                try:
                    sock = socket.create_connection(('127.0.0.1', args.port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('QEMU TCP startup timed out')
                    stop.wait(.25)
            if args.backend == 'qemu' and sock is None:
                return
            if sock is not None:
                connection[0] = sock
                sock.settimeout(900)
                if receive(sock, 8) != b'Y26READY':
                    raise ValueError('Unexpected QEMU greeting')
            run = secrets.randbits(64) or 1
            frame = 0
            previous_capture = -1
            while not stop.is_set():
                with lock:
                    item = state['latest']
                if item is None or item[0] == previous_capture:
                    stop.wait(.01)
                    continue
                capture_id, captured, image = item
                previous_capture = capture_id
                started = time.monotonic()
                tensor, transform = prepare_input(image)
                if args.save_frames:
                    save_image(output / f'source-{frame:06d}.png', image)
                with lock:
                    state['status'] = f'Processing #{frame} captured {datetime.fromtimestamp(captured):%H:%M:%S}'
                if backend is not None:
                    sent = time.monotonic()
                    candidates = backend.infer(tensor)
                else:
                    sock.sendall(HEADER.pack(b'Y26LIVE', 1, frame, run, tensor.nbytes, 0))
                    sock.sendall(tensor.astype('<f4', copy=False).tobytes())
                    sent = time.monotonic()
                    header = HEADER.unpack(receive(sock, HEADER.size))
                    if header != (b'Y26VDET\0', 1, frame, run, 300, 6):
                        raise ValueError('Result frame/session/shape mismatch')
                    candidates = np.frombuffer(receive(sock, 7200), dtype='<f4').reshape(300,6)
                if stop.is_set():
                    break
                classes = candidates[:,5]
                if (not np.isfinite(candidates).all() or np.any(classes != np.rint(classes))
                        or np.any((classes < 0) | (classes >= 80))):
                    raise ValueError('Invalid result values/classes')
                post_start = time.monotonic()
                boxes = postprocess(candidates, transform, .25)
                annotated = annotate(image, boxes)
                record = {'frame_id': frame, 'capture_id': capture_id, 'captured_unix': captured,
                          'elapsed_seconds': time.monotonic()-started,
                          'send_and_prepare_seconds': sent-started,
                          'transform': transform,
                          'postprocess_seconds': time.monotonic()-post_start,
                          'completed_monotonic': time.monotonic(),
                          'detections': boxes.tolist()}
                if args.backend == 'pcie':
                    record['pcie_timing'] = backend.last_timing
                with lock:
                    state['detected'] = (annotated, record)
                    state['results'].append(record)
                if args.save_frames:
                    np.save(output / f'candidates-{frame:06d}.npy', candidates)
                    save_image(output / f'detected-{frame:06d}.png', annotated)
                    write_json(output / f'result-{frame:06d}.json', record)
                print(f'Result {frame}: {len(boxes)} detections, {record["elapsed_seconds"]:.2f}s', flush=True)
                frame += 1
                if args.max_results and frame >= args.max_results:
                    if sock is not None:
                        sock.sendall(HEADER.pack(b'Y26LIVE',1,frame,run,0,0))
                    stop.wait(.5)
                    stop.set()
        except Exception as exc:
            if not stop.is_set():
                fail(exc)
        finally:
            if backend is not None and hasattr(backend, 'close'):
                try:
                    backend.close()
                except Exception as exc:
                    fail(exc)
            if sock is not None:
                sock.close()

    threads = []
    started = time.monotonic()
    try:
        if args.backend == 'qemu':
            log = (output / 'qemu.log').open('w', encoding='utf-8')
            command = [find_wsl(), '-d', args.distro, '--user', 'root', '--exec', 'env',
                       f'LIVE_PORT={args.port}', 'bash', windows_to_wsl(RISCV / 'build_accel_model.sh'),
                       '--precision','fp16','--mode','runtime','--demo','live']
            process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
        for target in (capture, recorder, inference):
            thread = threading.Thread(target=target, daemon=True)
            threads.append(thread)
            thread.start()
        window = f'YOLO26 {args.backend}: live preview | latest detection (Q/Esc exits)'
        if not args.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        next_record = time.monotonic()
        while not stop.is_set():
            with lock:
                latest, detected, status = state['latest'], state['detected'], state['status']
            raw = latest[2] if latest else None
            right = detected[0] if detected else None
            caption = 'Waiting for first result'
            if detected:
                record = detected[1]
                caption = f'Result #{record["frame_id"]} | {record["elapsed_seconds"]:.1f}s | age {time.time()-record["captured_unix"]:.1f}s'
            display = np.hstack((panel(raw, status), panel(right, caption)))
            if not args.no_display:
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 255 in (27, ord('q'), ord('Q')):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            now = time.monotonic()
            if now >= next_record and raw is not None:
                for kind, image in (('raw',raw), ('demo',display)):
                    try:
                        recording.put_nowait((kind,image))
                    except queue.Full:
                        state['record_dropped'] += 1
                next_record = now + .05
            if process is not None and process.poll() is not None and not stop.is_set():
                raise RuntimeError('QEMU exited; see qemu.log')
            if args.seconds and now-started >= args.seconds:
                break
            stop.wait(.01)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        fail(exc)
    finally:
        stop.set()
        if connection[0] is not None:
            try:
                connection[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if process is not None:
            stop_process(process)
        for thread in threads:
            thread.join(timeout=10)
        if log:
            log.close()
        cv2.destroyAllWindows()
        report.update(status='failed' if state['error'] else 'stopped', error=state['error'],
                      results=state['results'], recording_dropped=state['record_dropped'],
                      wall_seconds=time.monotonic()-started)
        results = state['results']
        if len(results) > 1:
            report['measured_result_fps'] = (len(results)-1) / (results[-1]['completed_monotonic']-results[0]['completed_monotonic'])
        if results:
            report['mean_postprocess_ms'] = float(np.mean([r['postprocess_seconds'] for r in results])*1000)
        write_json(output / 'report.json', report)
        if all(not thread.is_alive() for thread in threads):
            diagnostics.close()
        print(output, flush=True)
    if state['error']:
        raise RuntimeError(state['error'])


if __name__ == '__main__':
    main()
