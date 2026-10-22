"""Offline evidence gates shared by fresh tuning and video-only continuation."""
import hashlib
import json
import math
from pathlib import Path
from host_video import save_json
from optimization_report import compare_display, drawing_signature
from video_pipeline import Preprocessor


def checked_selection(path,firmware_sha,frames,refs):
    from board_profile import validate_saved_profile
    path=Path(path)
    if path.is_dir(): path=path/'report.json'
    raw=path.read_bytes(); report=json.loads(raw)
    if report.get('firmware_sha256')!=firmware_sha:
        raise ValueError('Saved selection belongs to different firmware; refusing board access')
    selected=report.get('selected',{})
    if (set(selected)!={'m','n','pack','compute','graph'} or
        selected['m'] not in (32,64,128) or selected['n'] not in (32,64,128) or
        any(not isinstance(selected[k],int) or selected[k]<0 or selected[k]&~mask
            for k,mask in (('pack',3),('compute',7),('graph',15)))):
        raise ValueError('Saved selection is missing or invalid')
    for field in ('baseline','selected_measurements'):
        rounds=report.get(field,[])
        if len(rounds)!=2 or any(len(rows)!=3 for rows in rounds):
            raise ValueError('Saved selection lacks two complete three-frame passes: '+field)
        for rows in rounds:
            for i,row in enumerate(rows):
                signature=drawing_signature(refs[i],frames[i]['transform'])
                if (row.get('input_sha256')!=frames[i]['input_sha256'] or
                    row.get('display_equal') is not True or row.get('golden_signature')!=signature or
                    row.get('actual_signature')!=signature):
                    raise ValueError('Saved input/golden/display differs: {} frame {}'.format(field,i))
                p=row['profile']; validated=validate_saved_profile(p,p['frame'])
                if validated['options'] or row.get('model_cycles')!=validated['model_cycles']:
                    raise ValueError('Saved timing is diagnostic or differs from its checked profile')
    return report,{'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()}


def video_summary(result,directory):
    return {'directory':str(directory),'status':result.get('status'),
            'completed_frames':len(result.get('results',[])),
            'inference_completed_frames':result.get('inference_completed_frames'),
            'submitted_frames':result.get('submitted_frames',[]),
            'recorded_frames':result.get('recorded_frames',0),
            'upload_mode':result.get('upload_mode'),'failure':result.get('failure',{}),
            'error':result.get('error')}


def check_frames(video,directory,result,expected,expected_inputs=None):
    directory=Path(directory); checks=[]; prep=Preprocessor(video)
    for i,entry in enumerate(result['results']):
        if entry['frame_id']!=i: raise ValueError('Frame IDs are missing, repeated or out of order')
        prefix='frame-{:06d}'.format(i)
        raw=(directory/(prefix+'-input.bin')).read_bytes()
        actual=(directory/(prefix+'-output.bin')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['input_sha256']:
            raise ValueError('Saved input hash mismatch')
        if hashlib.sha256(actual).hexdigest()!=entry['output_sha256']:
            raise ValueError('Saved result hash mismatch')
        if expected_inputs is not None and raw!=expected_inputs[i]['input']:
            raise ValueError('Transport comparison did not use the same input')
        original=video.read_image(directory/(prefix+'-source.png'))
        rebuilt,mapping=prep.prepare(original)
        if rebuilt!=raw or mapping!=entry['transform']:
            raise ValueError('Saved source image/mapping does not reproduce input')
        logical=(video.np.frombuffer(raw,dtype=video.np.uint8).astype('<f4')/video.np.float32(255)).tobytes()
        golden=expected(i,logical)
        check=compare_display(actual,golden,entry['transform'])
        a=video.annotate(original,video.postprocess(video.np.frombuffer(actual,dtype='<f4').reshape(300,6),entry['transform'],.25))
        b=video.annotate(original,video.postprocess(video.np.frombuffer(golden,dtype='<f4').reshape(300,6),entry['transform'],.25))
        check['drawing_pixels_equal']=bool(video.np.array_equal(a,b))
        check['display_equal']=check['display_equal'] and check['drawing_pixels_equal']
        checks.append(dict(check,frame_id=i,input_sha256=hashlib.sha256(logical).hexdigest()))
        if not check['display_equal']: (directory/(prefix+'-golden.bin')).write_bytes(golden)
        print('VIDEO GOLDEN frame={} display={} words_changed={}'.format(i,check['display_equal'],check['different_words']),flush=True)
    return checks


def interval_records(result,checks):
    # Discard startup/first completion. Compare the same two subsequent input
    # hashes in each of two passes; intervals include real host/PCIe scheduling.
    rows=[]
    for entry,check in zip(result['results'][1:],checks[1:]):
        value=entry.get('completed_interval_seconds')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Missing/invalid completed-frame interval')
        rows.append(dict(check,frame_seconds=value))
    return rows


def clean_frame_evidence(directory):
    directory=Path(directory).resolve()
    for path in directory.glob('frame-*'):
        if path.resolve().parent!=directory or not path.is_file():
            raise ValueError('Evidence path escaped acceptance')
        path.unlink()
