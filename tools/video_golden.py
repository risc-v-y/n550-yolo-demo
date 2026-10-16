"""Independent full-graph video golden using the packaged finite-RTL auditor.

Only software-predicted outputs feed subsequent nodes. Board candidates are used
solely by compare(), after the golden has been generated or its cache verified.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import time

from full_replay import NUMERIC_RULE, ROOT, Replay, verified_binary

FIELDS = ('x1', 'y1', 'x2', 'y2', 'score', 'class')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def save_json(path, data):
    save(path, (json.dumps(data, indent=2) + '\n').encode('utf-8'))


def compare(actual, expected):
    """Strict FP32 word comparison, including signed zero; no tolerance."""
    if len(actual) != 7200 or len(expected) != 7200:
        raise ValueError('Golden comparison requires two 7200-byte candidate arrays')
    a, b = struct.unpack('<1800f', actual), struct.unpack('<1800f', expected)
    if not all(math.isfinite(v) for v in a + b):
        raise ValueError('Non-finite candidates in golden comparison')
    aw, bw = struct.unpack('<1800I', actual), struct.unpack('<1800I', expected)
    different = [i for i in range(1800) if aw[i] != bw[i]]
    first = None
    if different:
        i = different[0]
        first = {'element': i, 'candidate': i // 6, 'field': FIELDS[i % 6],
                 'actual': a[i], 'golden': b[i],
                 'actual_bits': '0x{:08x}'.format(aw[i]), 'golden_bits': '0x{:08x}'.format(bw[i])}
    return {'status': 'EXACT' if not different else 'DIFFERENT', 'different_elements': len(different),
            'field_differences': {name: sum(i % 6 == col for i in different) for col, name in enumerate(FIELDS)},
            'first_difference': first,
            'max_coordinate_difference': max(abs(a[i] - b[i]) for i in range(1800) if i % 6 < 4),
            'max_score_difference': max(abs(a[i] - b[i]) for i in range(4, 1800, 6)),
            'actual_sha256': sha(actual), 'golden_sha256': sha(expected)}


class VideoGolden:
    def __init__(self, directory, root=ROOT, replay_factory=Replay):
        if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
            raise RuntimeError('Video golden requires Linux x86_64; use the host menu')
        self.directory = Path(directory)
        self.binary = verified_binary(root)
        release = load(root / 'firmware/host-verify.json')
        manifest_bytes = (root / 'reference/nodes/manifest.json').read_bytes().replace(b'\r\n', b'\n')
        self.manifest = json.loads(manifest_bytes)
        self.constants = (root / 'model/constants.bin').read_bytes()
        if (sha(manifest_bytes) != release['reference_manifest_sha256'] or
                sha(self.constants) != release['sources']['model/constants.bin']):
            raise ValueError('Golden graph/weights changed during asset verification')
        self.tensors, self.nodes = self.manifest['tensors'], self.manifest['nodes']
        if [n['id'] for n in self.nodes] != list(range(len(self.nodes))):
            raise ValueError('Golden nodes are not ordered')
        self.output_id = self.nodes[-1]['outputs'][-1]
        if self.output_id < 0 or self.tensors[self.output_id]['bytes'] != 7200:
            raise ValueError('Golden graph has unexpected final output')
        self.last_use = {tid: n['id'] for n in self.nodes for tid in n['inputs']}
        self.output_count = sum(tid >= 0 for n in self.nodes for tid in n['outputs'])
        self.identity = {'version': 1, 'numeric_rule': NUMERIC_RULE, 'auditor_sha256': release['sha256'],
                         'reference_manifest_sha256': sha(manifest_bytes),
                         'constants_sha256': sha(self.constants),
                         'client_sha256': sha(Path(__file__).with_name('full_replay.py').read_bytes().replace(b'\r\n', b'\n')),
                         'runner_sha256': sha(Path(__file__).read_bytes().replace(b'\r\n', b'\n'))}
        self.fingerprint = sha(json.dumps(self.identity, sort_keys=True).encode('utf-8'))
        self.replay_factory, self.replay = replay_factory, None

    def close(self):
        if self.replay is not None:
            replay, self.replay = self.replay, None
            replay.close()

    def predict(self, data, progress=None):
        if len(data) != self.tensors[0]['bytes']:
            raise ValueError('Golden input length mismatch')
        live, hashes = {0: data}, []
        try:
            if self.replay is None:
                self.replay = self.replay_factory(16, binary=self.binary)
            for node in self.nodes:
                if progress and (node['id'] % 32 == 0 or node is self.nodes[-1]):
                    progress(node['id'])
                inputs = {}
                for tid in node['inputs']:
                    tensor = self.tensors[tid]
                    inputs[tid] = (self.constants[tensor['offset']:tensor['offset'] + tensor['bytes']]
                                   if tensor['constant'] else live[tid])
                predicted = self.replay.run(node, self.tensors, inputs)
                ids = [tid for tid in node['outputs'] if tid >= 0]
                if set(predicted) != set(ids):
                    raise ValueError('Golden node output IDs mismatch')
                for tid in ids:
                    if len(predicted[tid]) != self.tensors[tid]['bytes']:
                        raise ValueError('Golden node output length mismatch')
                    live[tid] = predicted[tid]
                    hashes.append({'node': node['id'], 'tensor': tid, 'sha256': sha(predicted[tid])})
                for tid in list(live):
                    if tid != self.output_id and self.last_use.get(tid, -1) <= node['id']:
                        del live[tid]
            return live[self.output_id], hashes
        except Exception as exc:
            self.close()
            raise RuntimeError('Golden node {} failed: {}'.format(node['id'] if 'node' in locals() else 'startup', exc)) from exc
        except BaseException:
            self.close()
            raise

    def check(self, frame_id, data, actual):
        started = time.monotonic()
        prefix = self.directory / 'golden-{:06d}'.format(frame_id)
        metadata_path, binary_path = prefix.with_suffix('.json'), prefix.with_suffix('.bin')
        input_sha = sha(data)
        metadata = load(metadata_path) if metadata_path.exists() else None
        cached = bool(metadata and metadata.get('fingerprint') == self.fingerprint and
                      metadata.get('input_sha256') == input_sha and metadata.get('frame_id') == frame_id)
        if cached:
            expected = binary_path.read_bytes()
            if (len(expected) != 7200 or sha(expected) != metadata['output_sha256'] or
                    metadata['nodes'] != len(self.nodes) or metadata['outputs'] != self.output_count):
                raise ValueError('Golden cache corrupted for frame {}; no automatic overwrite'.format(frame_id))
        else:
            print('VIDEO GOLDEN frame={} starting (host CPU, no board)'.format(frame_id), flush=True)
            def progress(node):
                save_json(self.directory / 'golden-progress.json',
                          {'frame_id': frame_id, 'node': node, 'nodes': len(self.nodes),
                           'fingerprint': self.fingerprint, 'input_sha256': input_sha})
                print('VIDEO GOLDEN frame={} node={}/{} elapsed={:.1f}s'.format(
                    frame_id, node, len(self.nodes), time.monotonic() - started), flush=True)
            expected, hashes = self.predict(data, progress)
            # Commit metadata last. A partial frame is recomputed on the next report.
            save(binary_path, expected)
            metadata = {'version': 1, 'frame_id': frame_id, 'fingerprint': self.fingerprint,
                        'identity': self.identity, 'input_sha256': input_sha, 'output_sha256': sha(expected),
                        'nodes': len(self.nodes), 'outputs': self.output_count, 'node_hashes': hashes,
                        'generation_seconds': time.monotonic() - started}
            save_json(metadata_path, metadata)
        result = compare(actual, expected)
        result.update(frame_id=frame_id, input_sha256=input_sha, fingerprint=self.fingerprint,
                      nodes=metadata['nodes'], outputs=metadata['outputs'], cached=cached,
                      generation_seconds=metadata['generation_seconds'], check_seconds=time.monotonic() - started)
        print('VIDEO GOLDEN frame={} {} different={}/1800{}'.format(
            frame_id, result['status'], result['different_elements'], ' (verified cache)' if cached else ''), flush=True)
        return result


def verify_frames(directory, records, eligible, video):
    """Continue across numerical differences; errors never become a numeric pass."""
    started = time.monotonic()
    rows, engine = [], None
    interrupted = False
    identity = None
    try:
        if eligible:
            engine = VideoGolden(directory)
            identity = engine.identity
        for record in records:
            frame_id = record['frame_id']
            if frame_id not in eligible:
                rows.append({'frame_id': frame_id, 'status': 'UNVERIFIED', 'error': 'Frame evidence check failed'})
                continue
            try:
                image = video.read_image(directory / 'source-{:06d}.png'.format(frame_id))
                tensor, _ = video.prepare_input(image)
                data = tensor.astype('<f4', copy=False).tobytes()
                actual = (directory / 'candidates-{:06d}.bin'.format(frame_id)).read_bytes()
                if sha(data) != record['input_sha256'] or sha(actual) != record['output_sha256']:
                    raise ValueError('Frame data changed before golden comparison')
                rows.append(engine.check(frame_id, data, actual))
            except Exception as exc:
                rows.append({'frame_id': frame_id, 'status': 'ERROR', 'error': '{}: {}'.format(type(exc).__name__, exc)})
                print('VIDEO GOLDEN frame={} ERROR: {}'.format(frame_id, exc), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('VIDEO GOLDEN interrupted; completed golden frames kept. Menu 17 resumes offline.', flush=True)
    except Exception as exc:
        rows.append({'frame_id': None, 'status': 'ERROR', 'error': '{}: {}'.format(type(exc).__name__, exc)})
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                rows.append({'frame_id': None, 'status': 'ERROR', 'error': 'Golden cleanup failed: ' + str(exc)})
    exact = sum(r['status'] == 'EXACT' for r in rows)
    different = sum(r['status'] == 'DIFFERENT' for r in rows)
    errors = sum(r['status'] == 'ERROR' for r in rows)
    status = ('NO_FRAMES' if not records else 'DIFFERENT' if different else 'ERROR' if errors else
              'PASSED' if exact == len(records) else 'INCOMPLETE')
    return {'version': 1, 'status': status, 'numeric_rule': NUMERIC_RULE, 'identity': identity,
            'expected_frames': len(records), 'checked_frames': exact + different,
            'exact_frames': exact, 'different_frames': different, 'error_count': errors,
            'interrupted': interrupted, 'elapsed_seconds': time.monotonic() - started, 'frames': rows}
