"""Check diagnostic and performance images with the same accepted model config."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys

from host_video import ROOT, dependencies, save_json, even_frame, verify_movie
from video_pipeline import run as pipeline
from video_acceptance import check_frames, clean_frame_evidence, video_summary

CONFIG = {'m': 128, 'n': 32, 'pack': 25, 'compute': 0, 'graph': 11}
REVISION = 'dual-image-v1'


def run(args, output, pipeline_fn=None, golden_factory=None):
    from video_golden import VideoGolden
    from optimization_report import report_pages
    pipeline_fn = pipeline_fn or pipeline
    golden_factory = golden_factory or VideoGolden
    report = dict(revision=REVISION, status='running', stage='preflight',
                  selected=dict(CONFIG), upload_mode='serial', runs={}, frames=[])
    golden = None
    try:
        video = dependencies()
        for field, name in (('firmware_sha256', 'demo-fast.bin'),
                            ('diagnostic_sha256', 'demo-pcie.bin')):
            report[field] = hashlib.sha256((ROOT/'firmware'/name).read_bytes()).hexdigest()
        golden = golden_factory(output)  # Verify the frozen auditor before board access.
        refs = {}
        def expected(index, logical):
            key = hashlib.sha256(logical).hexdigest()
            if key not in refs:
                refs[key] = golden.predict(logical)[0]
            return refs[key]
        for mode, count in (('diagnostic', 3), ('performance', 30)):
            report['stage'] = mode
            save_json(output/'report.json', report)
            directory = output/mode
            directory.mkdir()
            options = copy.copy(args)
            options.mode, options.max_frames, options.load = mode, count, True
            print('MODE CHECK: {} {} frames; fixed config {}; serial upload'.format(
                mode, count, CONFIG), flush=True)
            result = pipeline_fn(options, directory, selected=CONFIG, retain=True, upload_mode='serial')
            summary = video_summary(result, directory)
            summary['results'] = result.get('results', [])
            for key in ('cold_start_seconds', 'open_configure_seconds', 'wall_seconds', 'data_crc'):
                summary[key] = result.get(key)
            report['runs'][mode] = summary
            report['completed_frames'] = summary['completed_frames']
            report['recorded_frames'] = summary['recorded_frames']
            report['failure'] = summary['failure']
            save_json(output/'report.json', report)
            if (summary['status'] != 'completed' or summary['completed_frames'] != count or
                    summary['recorded_frames'] != count or
                    summary['submitted_frames'] != list(range(count)) or
                    summary['inference_completed_frames'] != count):
                raise RuntimeError('{}: {}; incomplete frame sequence'.format(mode, summary['error']))
            if mode == 'diagnostic' and any(not r.get('board_profile') for r in summary['results']):
                raise ValueError('Diagnostic image did not provide full node timing')
            if mode == 'performance' and any(r.get('cycles') is not None for r in summary['results']):
                raise ValueError('Performance run unexpectedly includes CPU profiling')
            checks = check_frames(video, directory, result, expected)
            summary['checks'] = checks
            if mode == 'performance': report['frames'] = checks
            # Only the transport and instrumentation changed: require bit equality.
            if any(not c['display_equal'] or c['different_words'] for c in checks):
                raise ValueError(mode+': golden mismatch; all evidence retained')
            if mode == 'performance':
                earlier = report['runs']['diagnostic']['checks']
                if [c['input_sha256'] for c in checks[:3]] != [c['input_sha256'] for c in earlier]:
                    raise ValueError('The two images did not process the same first three inputs')
            shape = even_frame(video, video.read_image(directory/(
                'frame-{:06d}-source.png'.format(count-1)))).shape
            summary['decoded_mp4_frames'] = verify_movie(video, directory/'demo.mp4', count, shape)
            save_json(output/'report.json', report)
        report.update(status='passed', stage='completed')
        for mode in report['runs']:
            clean_frame_evidence(output/mode)
        config_path = ROOT/'yolo26_pc/outputs/optimization/selected.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(config_path, dict({k: report[k] for k in
            ('status', 'selected', 'firmware_sha256', 'diagnostic_sha256', 'upload_mode')},
            revision=REVISION, report_path=str(output/'report.json')))
    except BaseException as exc:
        report.update(status='failed', error='{}: {}'.format(type(exc).__name__, exc))
    finally:
        if golden is not None: golden.close()
        save_json(output/'report.json', report)
        for page in report_pages(report): print(page, flush=True)
        print('MODE ACCEPTANCE '+report['status'].upper(), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'yolo26_pc/samples/pexels-3796613.mp4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pbcopy', default='/rv/fpga/bin/pbcopy')
    parser.add_argument('--pbload', default='/rv/fpga/bin/pbload')
    parser.add_argument('--reset-dir', default='/rv/fpga/bin')
    parser.add_argument('--infer-timeout', type=float, default=900)
    parser.add_argument('--tool-timeout', type=float, default=30)
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()
    args.no_display, args.playback_fps = not args.display, 0
    if (args.output/'report.json').exists(): parser.error('Output already contains a report')
    args.output.mkdir(parents=True, exist_ok=True)
    return 0 if run(args, args.output)['status'] == 'passed' else 1


if __name__ == '__main__': sys.exit(main())
