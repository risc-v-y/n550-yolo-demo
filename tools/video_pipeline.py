"""Bounded file-video pipeline; main thread owns GUI, one worker owns PCIe.

Normal artifacts: demo.mp4 + report.json. Diagnostic evidence is created on
failure or by the explicit acceptance runner, never by an automatic golden
pass hidden in normal playback.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time
from host_video import ROOT, dependencies, check_host, check_display, writer, even_frame, save_png, save_json


class Preprocessor:
    def __init__(self,video):
        self.v=video
        self.padded=video.np.empty((416,416,3),dtype=video.np.uint8)
        self.chw=video.np.empty((3,416,416),dtype=video.np.uint8)
        self.resized=None

    def prepare(self,image):
        v=self.v
        h,w=image.shape[:2]
        scale=min(416/h,416/w)
        rw,rh=round(w*scale),round(h*scale)
        left,top=round((416-rw)/2-.1),round((416-rh)/2-.1)
        if self.resized is None or self.resized.shape!=(rh,rw,3):
            self.resized=v.np.empty((rh,rw,3),dtype=v.np.uint8)
        v.cv2.resize(image,(rw,rh),dst=self.resized,interpolation=v.cv2.INTER_LINEAR)
        self.padded.fill(114)
        self.padded[top:top+rh,left:left+rw]=self.resized
        v.np.copyto(self.chw,self.padded[...,::-1].transpose(2,0,1))
        mapping={'original_hw':[h,w],'resized_hw':[rh,rw],'scale':scale,
                 'padding_ltrb':[left,top,416-rw-left,416-rh-top],
                 'layout':'NCHW','channels':'RGB','dtype':'uint8','shape':[1,3,416,416],
                 'normalization':'board LUT: FP32(uint8 / 255) -> FP16 RNE',
                 'padding_value':114,'interpolation':'cv2.INTER_LINEAR'}
        # Immutable bytes own their lifetime after the reusable buffers change.
        return self.chw.tobytes(),mapping


def saved_configuration(root=ROOT,mode='diagnostic'):
    path=root/'yolo26_pc/outputs/optimization/selected.json'
    default={'selected':{'m':128,'n':32,'pack':25,'compute':0,'graph':11},'upload_mode':'serial'}
    if not path.is_file(): return default
    record=json.loads(path.read_text(encoding='utf-8'))
    image='demo-fast.bin' if mode=='performance' else 'demo-pcie.bin'
    digest=hashlib.sha256((root/'firmware'/image).read_bytes()).hexdigest()
    recorded=record.get('diagnostic_sha256',record.get('firmware_sha256')) if mode=='diagnostic' else record.get('firmware_sha256')
    if record.get('status')!='passed' or recorded!=digest:
        return default
    if record.get('upload_mode','serial') not in ('serial','overlap'):
        raise ValueError('Invalid saved upload mode')
    return dict(record,upload_mode=record.get('upload_mode','serial'))


def configuration(root=ROOT):
    return saved_configuration(root)['selected']


def run(args,output,backend_factory=None,selected=None,retain=False,upload_mode=None):
    from pcie_stream import StreamBackend
    backend_factory=backend_factory or StreamBackend
    mode=getattr(args,'mode','diagnostic')
    fast=mode=='performance'
    evidence=[]
    stop=threading.Event()
    prepared,completed,recording=[queue.Queue(maxsize=2) for _ in range(3)]
    free_draw=queue.Queue(maxsize=3)
    errors=queue.Queue()
    events=deque(maxlen=128)
    threads=[]
    report={'version':3,'id':output.name,'status':'starting','source':str(args.source),
            'transport':'PCIe v2 UINT8 double buffer','results':[],
            'verification':'NOT_RUN','full_profile':'NOT_COLLECTED','mode':mode,
            'data_crc':'DISABLED' if fast else 'ENABLED',
            'requested_frames':args.max_frames,'recorded_frames':0,'submitted_frames':[],
            'inference_completed_frames':0}
    start=time.monotonic()
    v=None; visible=None; window=False; last_display=None; failed_item=[None]

    def put(q,item):
        while not stop.is_set():
            try: q.put(item,timeout=.1); return True
            except queue.Full: pass
        return False

    def get(q):
        while not stop.is_set():
            try: return q.get(timeout=.1)
            except queue.Empty: pass
        raise InterruptedError('Pipeline stopped')

    def guarded(fn):
        try: fn()
        except BaseException as exc:
            errors.put(exc); stop.set()

    def decode():
        cap=v.cv2.VideoCapture(str(args.source))
        prep=Preprocessor(v)
        try:
            declared=cap.get(v.cv2.CAP_PROP_FRAME_COUNT)
            i=0
            while not stop.is_set() and (not args.max_frames or i<args.max_frames):
                tick=time.monotonic(); ok,image=cap.read(); decoded=time.monotonic()
                if not ok:
                    if not i or (math.isfinite(declared) and declared>i):
                        raise ValueError('Video ended before declared frame count at {}'.format(i))
                    break
                raw,mapping=prep.prepare(image)
                if not put(prepared,{'frame_id':i,'image':image,'input':raw,'transform':mapping,
                        'input_sha256':None if fast else hashlib.sha256(raw).hexdigest(),'captured':tick,
                        'decode_seconds':decoded-tick,'prepare_seconds':time.monotonic()-decoded}): break
                i+=1
            put(prepared,None)
        finally: cap.release()

    def inference():
        board=None
        try:
            opened=time.monotonic()
            board=backend_factory(args.pbcopy,args.pbload,stop,args.infer_timeout,
                                  diagnostic=None if fast else events.append,tool_timeout=args.tool_timeout,
                                  mode=mode,profile=not fast)
            saved=saved_configuration(mode=mode)
            chosen=saved['selected'] if selected is None else selected
            scheduling=upload_mode or getattr(args,'upload_mode',None) or saved['upload_mode']
            if scheduling not in ('serial','overlap'): raise ValueError('Invalid upload mode')
            report['upload_mode']=scheduling
            board.configure(chosen)
            report['selected']=chosen; report['capabilities']=board.stream
            report['open_configure_seconds']=time.monotonic()-opened
            current=get(prepared)
            if current is None: raise ValueError('No video input')
            def upload(item,active_frame_id):
                failed_item[0]=item
                report['transfer']={'stage':'input_upload_readback','frame_id':item['frame_id'],
                                    'active_frame_id':active_frame_id,'slot':item['frame_id']%2+1}
                board.upload(item['input'],item['frame_id']+1)
            def begin(item):
                failed_item[0]=item
                report['transfer']={'stage':'commit','frame_id':item['frame_id'],'active_frame_id':None}
                board.begin()
                report['submitted_frames'].append(item['frame_id'])
            upload(current,None); begin(current)
            while not stop.is_set():
                following=get(prepared)
                if scheduling=='overlap' and following is not None:
                    upload(following,current['frame_id'])
                failed_item[0]=current
                report['transfer']={'stage':'wait_result','frame_id':current['frame_id'],
                                    'active_frame_id':current['frame_id']}
                current['raw']=board.finish()
                report['inference_completed_frames']+=1
                current['timing']=dict(board.last_timing)
                if not fast: current['profile']=getattr(board,'last_profile',None)
                if not put(completed,current): break
                if following is None: break
                if scheduling=='serial': upload(following,None)
                current=following; failed_item[0]=current
                begin(current)
            put(completed,None)
        except BaseException as exc:
            report['failure']=dict(report.get('transfer',{}),**getattr(exc,'details',{}))
            raise
        finally:
            if board is not None: board.close()

    def record():
        movie=None; expected=0
        try:
            while not stop.is_set():
                item=get(recording)
                if item is None: break
                frame_id,image,entry=item
                if frame_id!=expected: raise ValueError('Recording frame order changed')
                tick=time.monotonic()
                frame=even_frame(v,image)
                if movie is None: movie=writer(v,output/'demo.mp4',report['playback_fps'],frame)
                movie.write(frame)
                if not fast: entry['record_seconds']=time.monotonic()-tick
                expected+=1; report['recorded_frames']=expected
                put(free_draw,image)
        finally:
            if movie is not None: movie.release()

    try:
        v=dependencies()
        # Preflight artifacts are temporary; normal output stays video+summary.
        import tempfile
        if fast:
            if not args.no_display: check_display()
            if not args.source.is_file(): raise FileNotFoundError(str(args.source))
            cap=v.cv2.VideoCapture(str(args.source))
            try:
                if not cap.isOpened(): raise ValueError('Cannot open input video')
                fps=cap.get(v.cv2.CAP_PROP_FPS)
                check={'source_fps':fps if math.isfinite(fps) and fps>0 else 25}
            finally: cap.release()
        else:
            with tempfile.TemporaryDirectory(prefix='host-check-',dir=str(output)) as check_dir:
                check=check_host(v,args.source,Path(check_dir),display=not args.no_display)
        if args.load:
            tick=time.monotonic()
            command=['bash',str(ROOT/'tools/load_board.sh'),'--reset-dir',args.reset_dir,
                     '--pbcopy',args.pbcopy,'--mode',mode]
            if fast:
                loaded=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
                if loaded.returncode:
                    detail=loaded.stdout+'\n'+loaded.stderr
                    (output/'load-error.log').write_text(detail,encoding='utf-8')
                    raise RuntimeError('Firmware load failed ({}): {}. Full log: {}'.format(
                        loaded.returncode,detail[-1200:],output/'load-error.log'))
            else: subprocess.run(command,cwd=ROOT,check=True)
            report['load_seconds']=time.monotonic()-tick
        report['cold_start_seconds']=time.monotonic()-start
        report['preflight_seconds']=report['cold_start_seconds']-report.get('load_seconds',0.)
        report['firmware_load_seconds']=report.get('load_seconds')
        report['playback_fps']=args.playback_fps or check['source_fps'] or 25
        report['firmware_sha256']=hashlib.sha256((ROOT/'firmware'/('demo-fast.bin' if fast else 'demo-pcie.bin')).read_bytes()).hexdigest()
        report['status']='running'; save_json(output/'report.json',report)
        if not args.no_display:
            v.cv2.namedWindow('YOLO PCIe',v.cv2.WINDOW_NORMAL); window=True
        for fn in (decode,inference,record):
            thread=threading.Thread(target=guarded,args=(fn,),daemon=True)
            threads.append(thread); thread.start()
        draw_count=0
        while not stop.is_set():
            try: item=completed.get(timeout=.02)
            except queue.Empty: item=False
            if item is None: break
            if item is not False:
                if item['frame_id']!=len(report['results']): raise ValueError('Frame loss or duplicate')
                visible=item
                tick=time.monotonic()
                # Diagnostic transport validates all candidates; fast mode only
                # checks the values actually used for drawing below.
                raw=v.np.frombuffer(item['raw'],dtype='<f4').reshape(300,6)
                boxes=v.postprocess(raw,item['transform'],.25)
                # Only values about to reach drawing/indexing need checks here.
                if fast:
                    for box in boxes:
                        if not v.np.isfinite(box).all() or not 0<=box[5]<80 or box[5]!=int(box[5]):
                            raise ValueError('Invalid detection for drawing')
                try: draw=free_draw.get_nowait()
                except queue.Empty:
                    if draw_count<3:
                        draw=v.np.empty_like(item['image']); draw_count+=1
                    else: draw=get(free_draw)
                if draw.shape!=item['image'].shape: draw=v.np.empty_like(item['image'])
                v.annotate(item['image'],boxes,draw)
                post=time.monotonic()-tick
                entry={k:item[k] for k in ('frame_id','input_sha256','transform','decode_seconds','prepare_seconds')}
                entry.update(item['timing']); entry.update(detections=len(boxes),postprocess_seconds=post,
                    output_sha256=None if fast else hashlib.sha256(item['raw']).hexdigest())
                if fast:
                    for key in ('input_sha256','output_sha256','decode_seconds','prepare_seconds','postprocess_seconds'):
                        entry.pop(key,None)
                elif item.get('profile') is not None: entry['board_profile']=item['profile']
                tick=time.monotonic()
                if window: v.cv2.imshow('YOLO PCIe',draw)
                key=v.cv2.waitKey(1) if window else -1
                submitted=time.monotonic()
                entry['display_submit_seconds']=submitted-tick if window else None
                entry['capture_to_submit_seconds']=submitted-item['captured'] if window else None
                entry['completed_interval_seconds']=submitted-last_display if last_display is not None else None
                last_display=submitted
                if retain and fast:
                    evidence.append((item,entry))  # No extra I/O/hash/CRC in the timed flow.
                elif retain:
                    prefix='frame-{:06d}'.format(item['frame_id'])
                    (output/(prefix+'-input.bin')).write_bytes(item['input'])
                    (output/(prefix+'-output.bin')).write_bytes(item['raw'])
                    save_png(v,output/(prefix+'-source.png'),item['image'])
                report['results'].append(entry)
                if not put(recording,(item['frame_id'],draw,entry)): break
                if not fast: print('VIDEO frame={} completed; model cycles={}'.format(item['frame_id'],entry['cycles']['model_cycles']),flush=True)
                if key&255 in (27,ord('q'),ord('Q')): raise KeyboardInterrupt()
            if window:
                key=v.cv2.waitKey(10)
                if key&255 in (27,ord('q'),ord('Q')) or v.cv2.getWindowProperty('YOLO PCIe',v.cv2.WND_PROP_VISIBLE)<1:
                    raise KeyboardInterrupt()
        if not errors.empty(): raise errors.get()
        put(recording,None)
        for thread in threads: thread.join(timeout=args.tool_timeout+5)
        if any(t.is_alive() for t in threads): raise RuntimeError('Pipeline worker failed to stop')
        if not errors.empty(): raise errors.get()
        if report['recorded_frames']!=len(report['results']): raise ValueError('Recording completion count mismatch')
        report['status']='completed'
    except KeyboardInterrupt:
        report.update(status='stopped',error='Interrupted by user')
    except BaseException as exc:
        report.update(status='failed',error='{}: {}'.format(type(exc).__name__,exc))
    finally:
        stop.set()
        for thread in threads: thread.join(timeout=args.tool_timeout+5)
        if window: v.cv2.destroyAllWindows()
        if report['status']!='completed':
            item=failed_item[0] or visible
            if item is not None:
                (output/'failure-input.bin').write_bytes(item['input'])
                save_json(output/'failure-frame.json',dict(report.get('failure',{}),
                          frame_id=item['frame_id'],transform=item['transform']))
                save_png(v,output/'failure-source.png',item['image'])
                if 'raw' in item: (output/'failure-output.bin').write_bytes(item['raw'])
            (output/'diagnostics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
            if report.get('failure',{}).get('expected_hex'):
                save_json(output/'request-readback.json',report['failure'])
        report['wall_seconds']=time.monotonic()-start
        # Acceptance reuses the actual performance image/transport. Evidence
        # is materialized only after inference, display and recorder shutdown.
        for item,entry in evidence:
            prefix='frame-{:06d}'.format(item['frame_id'])
            (output/(prefix+'-input.bin')).write_bytes(item['input'])
            (output/(prefix+'-output.bin')).write_bytes(item['raw'])
            save_png(v,output/(prefix+'-source.png'),item['image'])
            entry['input_sha256']=hashlib.sha256(item['input']).hexdigest()
            entry['output_sha256']=hashlib.sha256(item['raw']).hexdigest()
        report['timing_note']='Pipeline stages overlap. Display submission ends after OpenCV events; remote X11 presentation is not measured.'
        report['startup_note']='cold_start is host preflight plus optional firmware load; OPEN/configure is separate. Acceptance may load before this pipeline.'
        save_json(output/'report.json',report)
        print('VIDEO {}: {} frames, recorded={}'.format(report['status'].upper(),len(report['results']),report['recorded_frames']),flush=True)
        if report.get('error'): print(report['error'],flush=True)
        print('Output: '+str(output),flush=True)
    return report
