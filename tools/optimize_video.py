"""One host job: original release baseline, measured selection, 30-frame gate.

The original finite-RTL host verifier is unchanged. No candidate is selected
by QEMU duration, and no interrupted/write-failed run becomes a timing win.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from host_video import ROOT, dependencies, save_json, run as baseline_video, verify_movie, even_frame
from video_pipeline import Preprocessor, run as pipeline
from optimization_report import compare_display, repeatable_improvement, report_pages

BASELINE='2b32358c13e31456e1505f534768f8aaf662bba9'


def load(args,image=None,digest=None):
    command=['bash',str(ROOT/'tools/load_board.sh'),'--reset-dir',args.reset_dir,'--pbcopy',args.pbcopy]
    if image is not None: command+=['--image',str(image),'--sha256',digest]
    subprocess.run(command,cwd=ROOT,check=True)


def baseline_image(directory):
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT)
    try: git('cat-file','-e',BASELINE)
    except subprocess.CalledProcessError:
        subprocess.run(['git','fetch','origin',BASELINE],cwd=ROOT,check=True)
    checks=git('show',BASELINE+':firmware/SHA256SUMS').decode().splitlines()
    digests=[line.split()[0] for line in checks if line.split()[1:] == ['firmware/demo-pcie.bin']]
    if len(digests)!=1: raise ValueError('Original release checksum missing/duplicated')
    raw=git('show',BASELINE+':firmware/demo-pcie.bin')
    if hashlib.sha256(raw).hexdigest()!=digests[0]: raise ValueError('Original release checksum mismatch')
    path=directory/'baseline.bin'; path.write_bytes(raw)
    return path,digests[0]


def samples(video,path):
    cap=video.cv2.VideoCapture(str(path)); prep=Preprocessor(video); result=[]
    try:
        for i in range(3):
            ok,image=cap.read()
            if not ok: raise ValueError('Acceptance video needs at least three frames')
            raw,mapping=prep.prepare(image)
            logical=video.np.frombuffer(raw,dtype=video.np.uint8).astype('<f4')/video.np.float32(255)
            old,_=video.prepare_input(image)
            if logical.tobytes()!=old.tobytes(): raise ValueError('UINT8 preprocessing changed original input')
            result.append({'input':raw,'logical':logical.tobytes(),'transform':mapping,
                           'input_sha256':hashlib.sha256(logical.tobytes()).hexdigest()})
    finally: cap.release()
    return result


def alternatives(selected,caps):
    for m in (32,64,128):
        for n in (32,64,128):
            if max(m,n)<=caps['max_rows'] and m*n*4<=caps['accumulator_bytes']:
                yield dict(selected,m=m,n=n)


def benchmark(board,config,frames,golden,directory,label):
    board.configure(config)
    rounds=[]
    for repeat in range(2):
        records=[]
        for i,sample in enumerate(frames):
            tick=time.monotonic()
            raw=board.infer_u8(sample['input'])
            wall=time.monotonic()-tick
            check=compare_display(raw,golden[i],sample['transform'])
            profile=board.last_profile
            if profile is None: raise ValueError('Missing acceptance node profile')
            record=dict(check,input_sha256=sample['input_sha256'],model_cycles=profile['model_cycles'],
                        service_cycles=profile['service_cycles'],serial_pcie_seconds=wall,
                        profile=profile,timing=board.last_timing)
            records.append(record)
            print('TUNE {} repeat={} frame={} cycles={} display={}'.format(label,repeat+1,i,
                  record['model_cycles'],check['display_equal']),flush=True)
            if not check['display_equal']:
                (directory/('{}-{}-{}-actual.bin'.format(label,repeat,i))).write_bytes(raw)
        rounds.append(records)
    return rounds


def run(args,output):
    from pcie_stream import StreamBackend
    from video_golden import VideoGolden
    report={'status':'running','baseline_commit':BASELINE,'frames':[],'candidates':[],
            'firmware_sha256':hashlib.sha256((ROOT/'firmware/demo-pcie.bin').read_bytes()).hexdigest()}
    save_json(output/'report.json',report)
    golden=None; board=None; temporary=None
    try:
        video=dependencies(); frames=samples(video,args.source)
        golden=VideoGolden(output)
        refs=[]
        for i,sample in enumerate(frames):
            print('Original golden frame={}'.format(i),flush=True)
            refs.append(golden.predict(sample['logical'])[0])
        temporary=Path(tempfile.mkdtemp(prefix='baseline-',dir=str(output)))
        image,digest=baseline_image(temporary); report['baseline_sha256']=digest
        load(args,image,digest)
        baselines=[]
        for repeat in range(2):
            directory=temporary/str(repeat); directory.mkdir()
            old=copy.copy(args); old.load=False; old.check=False; old.max_frames=3
            result=baseline_video(old,directory)
            if result['status']!='completed' or len(result['results'])!=3:
                raise RuntimeError('Original release baseline did not complete')
            records=[]
            for i,record in enumerate(result['results']):
                if record.get('frame_id')!=i or record.get('input_sha256')!=frames[i]['input_sha256']:
                    raise ValueError('Baseline frame/input differs from candidate input')
                check=compare_display((directory/'candidates-{:06d}.bin'.format(i)).read_bytes(),refs[i],frames[i]['transform'])
                profile=record['pcie_timing']['board_profile']
                records.append(dict(check,input_sha256=frames[i]['input_sha256'],model_cycles=profile['model_cycles'],
                                    elapsed_seconds=record['elapsed_seconds'],profile=profile))
            if not all(r['display_equal'] for r in records):
                raise RuntimeError('Original baseline already differs visibly from golden')
            baselines.append(records)
        report['baseline']=baselines
        load(args)
        board=StreamBackend(args.pbcopy,args.pbload,infer_timeout=args.infer_timeout,
                            tool_timeout=args.tool_timeout,profile=True)
        report['capabilities']=board.stream
        selected={'m':32,'n':32,'pack':0,'compute':0,'graph':0}
        best=benchmark(board,selected,frames,refs,output,'current')
        if not all(r['display_equal'] for repeat in best for r in repeat):
            raise RuntimeError('Current conservative implementation failed display gate')
        def consider(config):
            nonlocal best,selected
            if config==selected: return
            name='candidate-{}'.format(len(report['candidates']))
            measured=benchmark(board,config,frames,refs,output,name)
            win=repeatable_improvement(best,measured)
            report['candidates'].append({'config':config,'selected':win,'rounds':measured})
            if win: selected=config; best=measured
            report['selected']=selected
            save_json(output/'report.json',report)
        for config in list(alternatives(selected,board.stream)): consider(config)
        for pack in (1,2,3): consider(dict(selected,pack=pack))
        for flag in (1,2,4): consider(dict(selected,compute=selected['compute']|flag))
        for flag in (1,2,4,8): consider(dict(selected,graph=selected['graph']|flag))
        report['selected']=selected; report['selected_measurements']=best
        board.close(); board=None
        final_args=copy.copy(args); final_args.load=False; final_args.max_frames=30
        final=output/'video'; final.mkdir()
        result=pipeline(final_args,final,selected=selected,retain=True)
        if result['status']!='completed' or len(result['results'])!=30:
            raise RuntimeError('Thirty-frame continuous run did not complete')
        for entry in result['results']:
            i=entry['frame_id']; prefix='frame-{:06d}'.format(i)
            raw=(final/(prefix+'-input.bin')).read_bytes()
            logical=(video.np.frombuffer(raw,dtype=video.np.uint8).astype('<f4')/video.np.float32(255)).tobytes()
            if hashlib.sha256(raw).hexdigest()!=entry['input_sha256']: raise ValueError('Saved input hash mismatch')
            actual=(final/(prefix+'-output.bin')).read_bytes()
            if hashlib.sha256(actual).hexdigest()!=entry['output_sha256']: raise ValueError('Saved result hash mismatch')
            expected=golden.predict(logical)[0]
            check=compare_display(actual,expected,entry['transform'])
            original=video.read_image(final/(prefix+'-source.png'))
            a=video.annotate(original,video.postprocess(video.np.frombuffer(actual,dtype='<f4').reshape(300,6),entry['transform'],.25))
            b=video.annotate(original,video.postprocess(video.np.frombuffer(expected,dtype='<f4').reshape(300,6),entry['transform'],.25))
            check['drawing_pixels_equal']=bool(video.np.array_equal(a,b))
            check['display_equal']=check['display_equal'] and check['drawing_pixels_equal']
            report['frames'].append(dict(check,frame_id=i,input_sha256=hashlib.sha256(logical).hexdigest()))
            print('FINAL frame={} display={} words_changed={}'.format(i,check['display_equal'],check['different_words']),flush=True)
            if not check['display_equal']: (final/(prefix+'-golden.bin')).write_bytes(expected)
        report['recorded_frames']=verify_movie(video,final/'demo.mp4',30,even_frame(video,original).shape)
        report['cold_start_seconds']=result.get('cold_start_seconds')
        report['open_configure_seconds']=result.get('open_configure_seconds')
        report['wall_seconds']=result.get('wall_seconds')
        report['cycles']={key:[r['cycles'][key] for r in result['results'] if key in r.get('cycles',{})]
                          for key in ('model_cycles','service_cycles')}
        keys=('decode_seconds','prepare_seconds','upload_seconds','run_roundtrip_seconds','download_seconds',
              'postprocess_seconds','display_submit_seconds','record_seconds','completed_interval_seconds')
        report['stages']={k:[r[k] for r in result['results'] if isinstance(r.get(k),(int,float))] for k in keys}
        report['status']='passed' if all(r['display_equal'] for r in report['frames']) else 'failed'
        report['full_profile']='collected in two matched three-frame passes per candidate'
        if report['status']=='passed':
            config_path=ROOT/'yolo26_pc/outputs/optimization/selected.json'
            config_path.parent.mkdir(parents=True,exist_ok=True)
            save_json(config_path,{k:report[k] for k in ('status','selected','firmware_sha256')})
            # Delete only files made by this successful acceptance, under its
            # own resolved run directory. Failure material remains untouched.
            for path in final.glob('frame-*'):
                if path.resolve().parent!=final.resolve(): raise ValueError('Evidence path escaped acceptance')
                path.unlink()
            if temporary.resolve().parent!=output.resolve() or not temporary.name.startswith('baseline-'):
                raise ValueError('Temporary baseline path escaped acceptance')
            shutil.rmtree(temporary); temporary=None
        else:
            report['error']='Display gate failed on final video; candidate was NOT installed'
    except BaseException as exc:
        report.update(status='failed',error='{}: {}'.format(type(exc).__name__,exc))
    finally:
        if board is not None: board.close()
        if golden is not None: golden.close()
        save_json(output/'report.json',report)
        for page in report_pages(report): print(page,flush=True)
        print('OPTIMIZATION '+report['status'].upper(),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'yolo26_pc/samples/pexels-3796613.mp4')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pbcopy',default='/rv/fpga/bin/pbcopy')
    parser.add_argument('--pbload',default='/rv/fpga/bin/pbload')
    parser.add_argument('--reset-dir',default='/rv/fpga/bin')
    parser.add_argument('--infer-timeout',type=float,default=900)
    parser.add_argument('--tool-timeout',type=float,default=30)
    parser.add_argument('--display',action='store_true')
    args=parser.parse_args(); args.no_display=not args.display; args.playback_fps=0
    args.output.mkdir(parents=True,exist_ok=True)
    return 0 if run(args,args.output)['status']=='passed' else 1


if __name__=='__main__': sys.exit(main())
