"""Run a frame-by-frame YOLO26 FP16 RVV+AMU video demo in QEMU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
RISCV = WORKSPACE / "yolo26_riscv"
SIZE = 416
ROWS = 300
COLUMNS = 6
DEFAULT_VIDEO = ROOT / "samples" / "pexels-3796613.mp4"
LANDING_URL = "https://www.pexels.com/video/people-walking-on-the-street-3796613/"
DOWNLOAD_URL = "https://videos.pexels.com/video-files/3796613/3796613-hd_1920_1080_25fps.mp4"
DEFAULT_VIDEO_SHA256 = "fcd2af324d05ae09da2570ffd693da084f6afdb209b078b419c30741742b43c5"
COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2
import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def prepare_input(image: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    scale = min(SIZE / height, SIZE / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    horizontal, vertical = (SIZE - resized_width) / 2, (SIZE - resized_height) / 2
    left, top = round(horizontal - 0.1), round(vertical - 0.1)
    right, bottom = round(horizontal + 0.1), round(vertical + 0.1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    if padded.shape != (SIZE, SIZE, 3):
        raise ValueError(f"Unexpected preprocessed shape: {padded.shape}")
    tensor = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1), dtype=np.float32)
    tensor /= np.float32(255.0)
    transform = {
        "original_hw": [height, width], "resized_hw": [resized_height, resized_width],
        "scale": scale, "padding_ltrb": [left, top, right, bottom],
        "padding_value": 114, "interpolation": "cv2.INTER_LINEAR",
        "layout": "NCHW", "channels": "RGB", "dtype": "float32",
        "normalization": "uint8 / 255.0", "shape": [1, 3, SIZE, SIZE],
    }
    return tensor, transform


def download_default_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading demo video: {LANDING_URL}", flush=True)
    try:
        request = urllib.request.Request(
            DOWNLOAD_URL,
            headers={"User-Agent": "Mozilla/5.0", "Referer": LANDING_URL, "Accept": "video/mp4,*/*"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            if response.headers.get_content_type() != "video/mp4":
                raise ValueError(f"Unexpected download content type: {response.headers.get_content_type()}")
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            raise ValueError("Downloaded video is empty")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_video(source: Path, seconds: float, sample_fps: float) -> list[tuple[str, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError("Video reports an invalid frame rate")
    requested = int(round(seconds * sample_fps))
    frames = []
    try:
        for index in range(requested):
            timestamp = index / sample_fps
            source_index = round(timestamp * source_fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Video ended before requested frame {index} at {timestamp:.3f}s")
            frames.append((f"frame-{index:06d}", timestamp, image))
    finally:
        capture.release()
    return frames


def load_images(paths: list[Path]) -> list[tuple[str, float, np.ndarray]]:
    return [(path.stem, float(index), read_image(path)) for index, path in enumerate(paths)]


def build_input_stream(frames, directory: Path, run_id: int) -> list[dict]:
    original_directory = directory / "frames"
    result_directory = directory / "results"
    annotated_directory = directory / "annotated"
    detection_directory = directory / "detections"
    for path in (original_directory, result_directory, annotated_directory, detection_directory):
        path.mkdir(parents=True, exist_ok=False)
    records = []
    stream_path = directory / "frames.bin"
    with stream_path.open("wb") as stream:
        stream.write(struct.pack("<8sIIQII", b"Y26VID1", 1, len(frames), run_id,
                                 3 * SIZE * SIZE * 4, 0))
        for frame_id, (name, timestamp, image) in enumerate(frames):
            tensor, transform = prepare_input(image)
            frame_path = original_directory / f"frame-{frame_id:06d}.png"
            if not cv2.imwrite(str(frame_path), image):
                raise OSError(f"Cannot write {frame_path}")
            stream.write(struct.pack("<II", frame_id, 0))
            stream.write(tensor.astype("<f4", copy=False).tobytes())
            records.append({"frame_id": frame_id, "name": name, "timestamp_seconds": timestamp,
                            "source_frame": str(frame_path.relative_to(directory)), "transform": transform})
    return records


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected an absolute Windows path: {resolved}")
    relative = resolved.as_posix()[3:]
    return f"/mnt/{drive}/{relative}"


def find_wsl() -> str:
    installed = Path("C:/Program Files/WSL/wsl.exe")
    return str(installed) if installed.is_file() else "wsl.exe"


def launch_qemu(exchange: Path, distro: str, log_path: Path) -> tuple[subprocess.Popen, object]:
    command = [find_wsl(), "-d", distro, "--user", "root", "--exec", "bash",
               windows_to_wsl(RISCV / "build_accel_model.sh"), "--precision", "fp16",
               "--mode", "runtime", "--demo", "video", "--video-dir", windows_to_wsl(exchange)]
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
    except Exception:
        log.close()
        raise
    return process, log


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def read_raw_result(exchange: Path, frame_id: int, run_id: int) -> np.ndarray:
    stem = f"frame-{frame_id:06d}"
    done = (exchange / "results" / f"{stem}.done").read_bytes()
    if len(done) != 24:
        raise ValueError(f"Invalid completion marker length for frame {frame_id}")
    magic, version, completed_frame, completed_run = struct.unpack("<8sIIQ", done)
    if (magic, version, completed_frame, completed_run) != (b"Y26DONE\0", 1, frame_id, run_id):
        raise ValueError(f"Completion marker mismatch for frame {frame_id}")
    blob = (exchange / "results" / f"{stem}.bin").read_bytes()
    if len(blob) != 32 + ROWS * COLUMNS * 4:
        raise ValueError(f"Invalid result length for frame {frame_id}: {len(blob)}")
    magic, version, result_frame, result_run, rows, columns = struct.unpack_from("<8sIIQII", blob)
    if (magic, version, result_frame, result_run, rows, columns) != (
            b"Y26VDET\0", 1, frame_id, run_id, ROWS, COLUMNS):
        raise ValueError(f"Result header mismatch for frame {frame_id}")
    candidates = np.frombuffer(blob, dtype="<f4", offset=32).reshape(ROWS, COLUMNS).copy()
    if not np.isfinite(candidates).all():
        raise ValueError(f"Non-finite candidate in frame {frame_id}")
    classes = candidates[:, 5]
    rounded = np.rint(classes)
    if not np.array_equal(classes, rounded) or np.any((rounded < 0) | (rounded >= len(COCO80))):
        raise ValueError(f"Invalid class id in frame {frame_id}")
    return candidates


def completion_marker_ready(path: Path) -> bool:
    """A newly created semihosting file is visible before its payload is complete."""
    try:
        return path.stat().st_size == 24
    except FileNotFoundError:
        return False


def postprocess(candidates: np.ndarray, transform: dict, confidence: float) -> np.ndarray:
    boxes = candidates[candidates[:, 4] > np.float32(confidence)].copy()
    if not len(boxes):
        return boxes.reshape(0, COLUMNS)
    height, width = transform["original_hw"]
    left, top = transform["padding_ltrb"][:2]
    scale = np.float32(transform["scale"])
    boxes[:, 0] = np.clip((boxes[:, 0] - np.float32(left)) / scale, np.float32(0), np.float32(width))
    boxes[:, 1] = np.clip((boxes[:, 1] - np.float32(top)) / scale, np.float32(0), np.float32(height))
    boxes[:, 2] = np.clip((boxes[:, 2] - np.float32(left)) / scale, np.float32(0), np.float32(width))
    boxes[:, 3] = np.clip((boxes[:, 3] - np.float32(top)) / scale, np.float32(0), np.float32(height))
    return boxes


def annotate(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    output = image.copy()
    for box in boxes:
        class_id = int(box[5])
        x1, y1, x2, y2 = (int(round(float(value))) for value in box[:4])
        color = (37 * (class_id + 3) % 255, 17 * (class_id + 11) % 255, 29 * (class_id + 7) % 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{COCO80[class_id]} {float(box[4]):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(output, (x1, label_top), (x1 + text_width + 4, label_top + text_height + baseline + 4),
                      color, cv2.FILLED)
        cv2.putText(output, label, (x1 + 2, label_top + text_height + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def show_waiting(window: str, image: np.ndarray, frame_id: int, total: int) -> bool:
    display = image.copy()
    text = f"QEMU processing frame {frame_id + 1}/{total} - Esc/Q to exit"
    cv2.rectangle(display, (0, 0), (min(display.shape[1], 620), 38), (0, 0, 0), cv2.FILLED)
    cv2.putText(display, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.imshow(window, display)
    key = cv2.waitKey(50) & 0xff
    return key not in (27, ord("q"), ord("Q"))


def write_detection_files(directory: Path, frame_id: int, boxes: np.ndarray) -> tuple[Path, Path]:
    binary = directory / "detections" / f"frame-{frame_id:06d}.bin"
    payload = struct.pack("<8sII", b"Y26DETS1", 1, len(boxes)) + boxes.astype("<f4", copy=False).tobytes()
    binary.write_bytes(payload)
    data = [{"xyxy": [float(value) for value in box[:4]], "confidence": float(box[4]),
             "class_id": int(box[5]), "class_name": COCO80[int(box[5])]} for box in boxes]
    descriptor = directory / "detections" / f"frame-{frame_id:06d}.json"
    write_json(descriptor, data)
    return binary, descriptor


def legacy_check(records: list[dict], exchange: Path, baseline: Path) -> dict:
    checks = []
    for record in records:
        expected = baseline / record["name"] / "scalar-detections.bin"
        actual = exchange / "detections" / f"frame-{record['frame_id']:06d}.bin"
        if not expected.is_file():
            raise ValueError(f"Missing legacy baseline for {record['name']}: {expected}")
        identical = actual.read_bytes() == expected.read_bytes()
        checks.append({"sample": record["name"], "byte_identical": identical,
                       "actual_sha256": sha256(actual), "expected_sha256": sha256(expected)})
        if not identical:
            raise ValueError(f"New host postprocess differs from legacy result: {record['name']}")
    return {"status": "passed", "samples": checks}


def reference_check(records: list[dict], exchange: Path, confidence: float) -> list[dict]:
    import torch
    from ultralytics import YOLO, settings

    lock = json.loads((ROOT / "assets.lock.json").read_text(encoding="utf-8"))
    model_path = ROOT / lock["model"]["path"]
    if sha256(model_path) != lock["model"]["sha256"]:
        raise ValueError("Reference model is missing or changed")
    settings.update({"sync": False, "weights_dir": str(ROOT / "models"), "runs_dir": str(ROOT / "outputs")})
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = YOLO(str(model_path))
    selected = sorted({0, len(records) // 2, len(records) - 1})
    comparisons = []
    for frame_id in selected:
        record = records[frame_id]
        image = read_image(exchange / record["source_frame"])
        result = model.predict(source=image, imgsz=SIZE, rect=False, device="cpu", quantize=32,
                               nms=False, conf=confidence, max_det=ROWS, agnostic_nms=False,
                               augment=False, save=False, verbose=False, batch=1)[0]
        reference = result.boxes.data.detach().cpu().numpy().astype(np.float32, copy=False)
        blob = (exchange / "detections" / f"frame-{frame_id:06d}.bin").read_bytes()
        _, _, count = struct.unpack_from("<8sII", blob)
        actual = np.frombuffer(blob, dtype="<f4", offset=16).reshape(count, COLUMNS)
        count_match = len(actual) == len(reference)
        class_match = count_match and np.array_equal(actual[:, 5], reference[:, 5])
        comparisons.append({
            "frame_id": frame_id, "qemu_detections": len(actual), "reference_detections": len(reference),
            "count_match": count_match, "class_sequence_match": bool(class_match),
            "max_coordinate_abs_difference_pixels": (
                float(np.max(np.abs(actual[:, :4] - reference[:, :4]))) if count_match and len(actual) else None),
            "max_confidence_abs_difference": (
                float(np.max(np.abs(actual[:, 4] - reference[:, 4]))) if count_match and len(actual) else None),
            "policy": "Measured differences only; no empirical tolerance is used to hide mismatches.",
        })
    return comparisons


def make_video(exchange: Path, records: list[dict], fps: float) -> str | None:
    images = [read_image(exchange / "annotated" / f"frame-{record['frame_id']:06d}.png") for record in records]
    if len({image.shape for image in images}) != 1:
        return None
    height, width = images[0].shape[:2]
    output = exchange / "annotated.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError("Cannot create annotated video")
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()
    return str(output.relative_to(exchange))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path, help="Video source; defaults to the pinned Pexels sample.")
    source.add_argument("--images", nargs="+", type=Path, help="Images used as a frame sequence for validation.")
    parser.add_argument("--output", type=Path, help="Fresh output directory; default includes a timestamp.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window.")
    parser.add_argument("--prepare-only", action="store_true", help="Only extract and preprocess frames.")
    parser.add_argument("--skip-reference", action="store_true", help="Skip PyTorch checks of first/middle/last frames.")
    parser.add_argument("--legacy-baseline", type=Path,
                        help="Require host-postprocessed images to byte-match an existing bus/zidane result directory.")
    args = parser.parse_args()
    if args.seconds <= 0 or args.sample_fps <= 0 or not 0 <= args.confidence <= 1:
        parser.error("seconds and sample-fps must be positive; confidence must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / "outputs" / "video-demo" / timestamp).resolve()
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    run_id = secrets.randbits(64) or 1

    if args.images:
        paths = [path.resolve(strict=True) for path in args.images]
        frames = load_images(paths)
        source_info = {"type": "images", "files": [{"path": str(path), "sha256": sha256(path)} for path in paths]}
    else:
        source = (args.source or DEFAULT_VIDEO).resolve()
        if not source.exists() and source == DEFAULT_VIDEO.resolve():
            download_default_video(source)
        source = source.resolve(strict=True)
        source_digest = sha256(source)
        if source == DEFAULT_VIDEO.resolve() and source_digest != DEFAULT_VIDEO_SHA256:
            raise ValueError("Default demo video hash differs from the recorded Pexels asset")
        frames = extract_video(source, args.seconds, args.sample_fps)
        source_info = {"type": "video", "path": str(source), "sha256": source_digest,
                       "landing_url": LANDING_URL if source == DEFAULT_VIDEO.resolve() else None,
                       "download_url": DOWNLOAD_URL if source == DEFAULT_VIDEO.resolve() else None}
    records = build_input_stream(frames, output, run_id)
    report = {
        "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "source": source_info, "configuration": {"frames": len(records), "seconds": args.seconds,
                                                   "sample_fps": args.sample_fps, "confidence": args.confidence,
                                                   "precision": "FP16", "backend": "RVV+AMU", "input": [1, 3, 416, 416]},
        "protocol": {"input": "Y26VID1 v1", "raw_result": "Y26VDET v1 [300,6] FP32",
                     "completion": "Y26DONE v1, published after result close"},
        "frames_bin_sha256": sha256(output / "frames.bin"), "frames": records,
    }
    write_json(output / "report.json", report)
    print(f"Prepared {len(records)} frames in {output}", flush=True)
    if args.prepare_only:
        return

    process, log = launch_qemu(output, args.distro, output / "qemu-launch.log")
    window = "YOLO26 QEMU video demo"
    started = time.monotonic()
    completed = []
    try:
        for record in records:
            frame_id = record["frame_id"]
            image = read_image(output / record["source_frame"])
            marker = output / "results" / f"frame-{frame_id:06d}.done"
            while not completion_marker_ready(marker):
                if process.poll() is not None:
                    raise RuntimeError(f"QEMU exited with code {process.returncode} before frame {frame_id}")
                if not args.no_display and not show_waiting(window, image, frame_id, len(records)):
                    raise KeyboardInterrupt
                if args.no_display:
                    time.sleep(0.2)
            candidates = read_raw_result(output, frame_id, run_id)
            boxes = postprocess(candidates, record["transform"], args.confidence)
            annotated = annotate(image, boxes)
            annotated_path = output / "annotated" / f"frame-{frame_id:06d}.png"
            if not cv2.imwrite(str(annotated_path), annotated):
                raise OSError(f"Cannot write {annotated_path}")
            binary, descriptor = write_detection_files(output, frame_id, boxes)
            record.update({"detections": len(boxes), "classes": [COCO80[int(box[5])] for box in boxes],
                           "result_received_seconds": time.monotonic() - started,
                           "annotated_frame": str(annotated_path.relative_to(output)),
                           "detections_binary": str(binary.relative_to(output)),
                           "detections_json": str(descriptor.relative_to(output))})
            completed.append(frame_id)
            report["status"] = "running"
            report["completed_frames"] = completed
            write_json(output / "report.json", report)
            print(f"Frame {frame_id + 1}/{len(records)}: {len(boxes)} detections", flush=True)
            if not args.no_display:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xff in (27, ord("q"), ord("Q")):
                    raise KeyboardInterrupt
        return_code = process.wait(timeout=120)
        if return_code:
            raise RuntimeError(f"QEMU build/run failed with code {return_code}")
    except KeyboardInterrupt:
        stop_process(process)
        report["status"] = "cancelled"
        write_json(output / "report.json", report)
        print("Demo cancelled by user", file=sys.stderr)
        return
    except Exception:
        stop_process(process)
        report["status"] = "failed"
        report["completed_frames"] = completed
        write_json(output / "report.json", report)
        raise
    finally:
        log.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.legacy_baseline:
        report["legacy_regression"] = legacy_check(records, output, args.legacy_baseline.resolve(strict=True))
    report["annotated_video"] = make_video(output, records, args.sample_fps)
    if not args.skip_reference:
        report["pytorch_reference"] = reference_check(records, output, args.confidence)
    report["status"] = "passed"
    report["wall_seconds_with_build_and_qemu"] = time.monotonic() - started
    report["qemu_log"] = "qemu-launch.log"
    write_json(output / "report.json", report)
    print(f"Video demo passed: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
