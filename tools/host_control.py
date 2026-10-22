"""Python 3.8+ host menu. Explicit capture/restore choices access the board."""
import argparse
from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def workspace():
    path=ROOT/'yolo26_pc/outputs/host-menu'
    path.mkdir(parents=True,exist_ok=True)
    return path


def save_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value)+'\n',encoding='utf-8')
    temporary.replace(path)


@contextmanager
def guard(path):
    import fcntl
    with path.open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Capture/update already active. Do not start another controller.') from None
        # Close without LOCK_UN: a detached child may inherit this open lock.
        yield stream


def board_lock():
    return Path(tempfile.gettempdir())/'yolo26-pcie.lock'


def update():
    with guard(workspace()/'control.lock'),guard(board_lock()):
        subprocess.run(['git','pull','--ff-only'],cwd=ROOT,check=True)
    print('Update complete. Board firmware has not been reloaded.')


def launch(sample='bus'):
    directory=workspace()
    with guard(directory/'control.lock') as lock:
        with guard(board_lock()):
            pass
        for name in ('pbcopy','pbload'):
            if not (Path('/rv/fpga/bin')/name).is_file():
                raise ValueError('Missing /rv/fpga/bin/'+name)
        run=Path(tempfile.mkdtemp(prefix='run-',dir=str(directory)))
        result=run/'capture'
        command=[sys.executable,'-u',str(ROOT/'yolo26_pc/node_capture.py'),
                 '--sample',sample,'--recover-paused','--pbcopy','/rv/fpga/bin/pbcopy',
                 '--pbload','/rv/fpga/bin/pbload','--output',str(result)]
        with (run/'terminal.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        save_json(directory/'last-launch.json',{'pid':process.pid,'result':str(result),
                                              'log':str(run/'terminal.log')})
    print('Background capture started; SSH disconnect will not stop it.')
    print('PID:',process.pid,'\nResult:',result,'\nLog:',run/'terminal.log')
    return process


def launch_full():
    directory=workspace()
    with guard(directory/'control.lock') as lock:
        with guard(board_lock()): pass
        for name in ('pbcopy','pbload','reset1.sh','reset.sh'):
            if not (Path('/rv/fpga/bin')/name).is_file():
                raise ValueError('Missing /rv/fpga/bin/'+name)
        base=ROOT/'yolo26_pc/outputs/full-check'; base.mkdir(parents=True,exist_ok=True)
        result=Path(tempfile.mkdtemp(prefix='run-',dir=str(base)))
        command=[sys.executable,'-u',str(ROOT/'tools/full_check.py'),'--output',str(result)]
        log=result/'terminal.log'
        with log.open('wb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        save_json(directory/'last-full.json',{'directory':str(result)})
        save_json(directory/'last-launch.json',{'pid':child.pid,'result':str(result),'log':str(log)})
    print('Full BUS-only check started in background; complete saved BUS evidence is reused.')
    print('PID:',child.pid,'\nResult:',result,
          '\nDiagnostics: 1 progress; 3 saved summary (3 pages).')
    return child


def full_pages():
    pointer=workspace()/'last-full.json'
    if not pointer.exists():
        raise ValueError('No full check started; use Diagnostics > 2, or bash tools/host.sh full-check.')
    directory=Path(json.loads(pointer.read_text(encoding='utf-8'))['directory'])
    from full_check import saved_report,render,bus_capture
    try: report=saved_report(directory)
    except LookupError:
        raw=bus_capture(directory)
        if not (raw/'summary.json').is_file():
            raise ValueError('BUS analysis and complete capture are missing in '+str(directory)+
                             '. Existing files were preserved; no new inference was started.')
        print('Reanalyzing saved BUS once with current RTL rule; no board access.',flush=True)
        with guard(workspace()/'control.lock'):
            subprocess.run([sys.executable,str(ROOT/'tools/full_check.py'),'--report',str(directory),
                            '--reanalyze'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        report=json.loads((directory/'bus-rtl-analysis.json').read_text(encoding='utf-8'))
    print('BUS report source:',report.get('source',{}).get('path',str(directory)))
    for page in (1,2,3):
        print(render(report,page),end='',flush=True)
        if input('Enter: next/return; q: menu: ').strip().lower()=='q': break


def restore():
    with guard(workspace()/'control.lock'):
        subprocess.run(['bash',str(ROOT/'tools/load_board.sh'),'--reset-dir','/rv/fpga/bin',
                        '--pbcopy','/rv/fpga/bin/pbcopy'],cwd=ROOT,check=True)


def export_png():
    subprocess.run([sys.executable,str(ROOT/'tools/render_results.py'),'--latest'],cwd=ROOT,check=True)


def cleanup():
    # The child owns both locks for the entire scan/delete operation.
    subprocess.run([sys.executable,str(ROOT/'tools/cleanup_host.py'),'--apply'],cwd=ROOT,check=True)


def pack_probe():
    subprocess.run([sys.executable,str(ROOT/'tools/pack_probe.py'),'--load'],cwd=ROOT,check=True)


def normal_replay():
    subprocess.run([sys.executable,str(ROOT/'tools/normal_replay.py'),'--load'],cwd=ROOT,check=True)


def video(check=False, continuous=False):
    options=[]
    if continuous:
        source=input('Video path [Enter: bundled sample]: ').strip().strip('"')
        count=input('Consecutive frames [Enter: 30; 0: entire video]: ').strip() or '30'
        display=input('Display completed frames? [Y/n]: ').strip().lower()
        if not count.isdigit() or display not in ('','y','yes','n','no'):
            raise ValueError('Use a nonnegative integer frame count and Y/n for display')
        if source:
            path=Path(source).expanduser().resolve()
            if not path.is_file(): raise ValueError('Video file does not exist: '+str(path))
            options+=['--source',str(path)]
        options+=['--max-frames',count]
        if display in ('n','no'): options+=['--no-display']
    with guard(workspace()/'control.lock'):
        command=[sys.executable,str(ROOT/'tools/host_video.py')]
        if check: command+=['--check','--no-display']
        else: command+=['--load']+options
        subprocess.run(command,cwd=ROOT,check=True)


def select_video_run():
    """Shared selection honors latest.json when the host clock is backdated."""
    base=ROOT/'yolo26_pc/outputs/host-video'
    runs=[]
    pointer=base/'latest.json'
    if pointer.exists():
        path=Path(json.loads(pointer.read_text(encoding='utf-8'))['directory'])
        if (path/'report.json').is_file(): runs.append(path.resolve())
    for item in sorted(base.glob('*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        metadata=json.loads(item.read_text(encoding='utf-8'))
        if metadata.get('host_check') and not metadata.get('results') and metadata.get('status')=='completed':
            continue
        if item.parent.resolve() not in runs: runs.append(item.parent.resolve())
    for i,path in enumerate(runs,1):
        metadata=json.loads((path/'report.json').read_text(encoding='utf-8'))
        print('{}: {} | {} | {} frames'.format(i,path.name,metadata.get('status'),len(metadata.get('results',[]))))
    answer=input('Report number/path [Enter: 1; q: cancel]: ').strip().strip('"')
    if answer.lower()=='q': return
    answer=answer or '1'
    if answer.isdigit():
        if not 1<=int(answer)<=len(runs): raise ValueError('No such saved video report; provide its directory')
        directory=runs[int(answer)-1]
    else:
        directory=Path(answer).expanduser().resolve()
    if not (directory/'report.json').is_file(): raise ValueError('Missing report.json in '+str(directory))
    print('Selected:',directory)
    return directory


def video_pages():
    """Select/flag/recheck old or new video runs; never launches the board."""
    directory=select_video_run()
    if directory is None: return
    metadata=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    if metadata.get('version')==3:
        from performance_report import analyze,render
        summary=analyze(directory)
        for page in (1,2,3):
            print(render(summary,page),flush=True)
            if input('Enter: next/return; q: menu: ').strip().lower()=='q': return
        print('Normal recording:',directory/'demo.mp4')
        print('Full golden comparison is in Whole-pipeline acceptance; normal frames are not saved.')
        return
    marked=input('Wrong-box frame IDs, comma separated [Enter: none]: ').strip()
    ids=[]
    if marked:
        try: ids=[int(v.strip()) for v in marked.split(',')]
        except ValueError: raise ValueError('Frame IDs must be integers separated by commas') from None
    note=input('Describe the wrong boxes [Enter: visual review]: ').strip() if ids else ''
    from host_video import dependencies
    from video_report import flag_frame,analyze,render
    with guard(workspace()/'control.lock'):
        # Reject invalid IDs before writing any flags.
        metadata=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        if any(v not in [r['frame_id'] for r in metadata['results']] for v in ids):
            raise ValueError('Only completed frame IDs may be flagged')
        for frame_id in ids: flag_frame(directory,frame_id,note or 'Marked for visual review')
        summary=analyze(directory,dependencies())
    for page in (1,2,3):
        print(render(summary,page),end='',flush=True)
        if input('Enter: next/return; q: menu: ').strip().lower()=='q': return
    print('Report and PNG directory:',directory)
    if summary['preview'] and input('Open preview and flagged PNGs in eog? [y/N]: ').strip().lower()=='y':
        import os
        import shutil
        if not os.environ.get('DISPLAY') or not shutil.which('eog'):
            raise ValueError('eog needs the working graphical SSH/desktop session; PNGs are already saved')
        files=[str(directory/summary['preview'])]
        files += [str(directory/'detected-{:06d}.png'.format(r['frame_id'])) for r in summary['flags']
                  if (directory/'detected-{:06d}.png'.format(r['frame_id'])).is_file()]
        subprocess.Popen(['eog','--new-instance']+files,cwd=ROOT,start_new_session=True,
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def performance_pages():
    directory=select_video_run()
    if directory is None: return
    mhz=float(input('Actual N550 core MHz [Enter: unknown, cycles only]: ').strip() or '0')
    from performance_report import analyze,render
    with guard(workspace()/'control.lock'):
        result=analyze(directory,mhz)
    for page in (1,2,3):
        print(render(result,page),end='',flush=True)
        if input('Enter: next/return; q: menu: ').strip().lower()=='q': break
    if 'pipeline' in result:
        print('Normal-mode summary:',directory/'report.json','; full node timing was not collected.')
    else:
        print('Saved performance pages and node-timing.csv:',directory)


def accuracy_pages():
    directory=select_video_run()
    if directory is None: return
    if json.loads((directory/'report.json').read_text(encoding='utf-8')).get('version')==3:
        raise ValueError('Normal runs retain MP4 and summary only. Select an older detailed run or use host_video.py --detailed.')
    frames=input('Frame IDs [Enter: first/middle/last; comma separated]: ').strip()
    labels=input('Complete manual labels JSON [Enter: visual comparison only]: ').strip().strip('"')
    import secrets
    from accuracy_review import analyze
    output=ROOT/'yolo26_pc/outputs/accuracy-review'/('review-'+secrets.token_hex(4))
    with guard(workspace()/'control.lock'):
        result=analyze(directory,output,[int(i) for i in frames.split(',')] if frames else None,
                       Path(labels).expanduser().resolve() if labels else None)
    if input('Open comparison PNGs in eog? [y/N]: ').strip().lower()=='y':
        import os
        import shutil
        if not os.environ.get('DISPLAY') or not shutil.which('eog'):
            raise ValueError('eog needs the working graphical session; PNGs are saved')
        files=[str(output/'comparison-{:06d}.png'.format(r['frame_id'])) for r in result['frames']]
        subprocess.Popen(['eog','--new-instance']+files,cwd=ROOT,start_new_session=True,
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def candidates():
    base=ROOT/'yolo26_pc/outputs'
    paths=[]
    pointer=base/'node-capture/latest.json'
    if pointer.exists():
        paths.append(Path(json.loads(pointer.read_text(encoding='utf-8'))['directory']))
    paths.extend(p.parent for pattern in ('node-capture/*/summary.json','host-menu/run-*/capture/summary.json')
                 for p in sorted(base.glob(pattern),key=lambda p:p.stat().st_mtime,reverse=True))
    result=[]
    for path in paths:
        path=path.resolve()
        if ((path/'summary.json').is_file() and path not in result and
                'runs' not in json.loads((path/'summary.json').read_text(encoding='utf-8'))):
            result.append(path)
    return result


def selected():
    path=workspace()/'selected.json'
    if path.exists():
        result=Path(json.loads(path.read_text(encoding='utf-8'))['directory'])
        if (result/'summary.json').is_file():
            return result
    runs=candidates()
    if not runs:
        raise ValueError('No saved capture report. Use bash tools/host.sh capture, or Diagnostics > 1 for its log.')
    return runs[0]


def choose():
    runs=candidates()
    if not runs:
        raise ValueError('No saved reports found.')
    for i,path in enumerate(runs,1):
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        print(i,summary.get('sample'),summary.get('status'),str(path))
    answer=input('Select report number (Enter cancels): ').strip()
    if not answer:
        return
    if not answer.isdigit() or not 1<=int(answer)<=len(runs):
        raise ValueError('Invalid report number')
    save_json(workspace()/'selected.json',{'directory':str(runs[int(answer)-1])})


def pages(kind):
    directory=selected()
    print('Saved report:',directory)
    script=ROOT/'yolo26_pc/node_report.py'
    numbers=(1,2,3) if kind=='status' else (2,3)
    for index,page in enumerate(numbers):
        command=[sys.executable,str(script),str(directory),'--page',str(page)]
        command+=['--view',kind]
        subprocess.run(command,cwd=ROOT,check=True)
        prompt='Enter: next page; q: return: ' if index<len(numbers)-1 else 'Enter: return to menu: '
        if input(prompt).strip().lower()=='q':
            break


def progress():
    pointer=workspace()/'last-launch.json'
    if not pointer.exists():
        print('No background diagnostic job. Use Diagnostics > 3 for a saved full BUS report.')
        return
    job=json.loads(pointer.read_text(encoding='utf-8'))
    print('PID:',job['pid'],'\nResult:',job['result'],'\nLog:',job['log'])
    summary=Path(job['result'])/'summary.json'
    if not summary.exists(): summary=Path(job['result'])/'report.json'
    if summary.exists():
        print('Job status:',json.loads(summary.read_text(encoding='utf-8')).get('status'))
    else:
        try:
            with guard(workspace()/'control.lock'):
                print('No running menu job and no final report. Inspect the log below.')
        except RuntimeError:
            print('Menu job active; select Diagnostics > 1 again to refresh.')
    with Path(job['log']).open(encoding='utf-8',errors='replace') as stream:
        print(''.join(deque(stream,maxlen=12)),end='')


# Display and dispatch share these ordered action names; numbers are not copied.
def launch_optimization():
    directory=workspace()
    with guard(directory/'control.lock') as lock:
        with guard(board_lock()): pass
        base=ROOT/'yolo26_pc/outputs/optimization'; base.mkdir(parents=True,exist_ok=True)
        result=Path(tempfile.mkdtemp(prefix='run-',dir=str(base)))
        log=result/'terminal.log'
        command=[sys.executable,'-u',str(ROOT/'tools/optimize_video.py'),'--output',str(result)]
        with log.open('wb') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        save_json(base/'latest.json',{'directory':str(result)})
        save_json(directory/'last-launch.json',{'pid':child.pid,'result':str(result),'log':str(log)})
    print('Optimization acceptance started in background. SSH disconnect will not stop it.')
    print('Main menu 11: report; Diagnostics > 1: progress. Output:',result)
    return child


def optimization_pages():
    from optimization_report import report_pages
    pointer=ROOT/'yolo26_pc/outputs/optimization/latest.json'
    if not pointer.is_file(): raise ValueError('No optimization acceptance has been started')
    directory=Path(json.loads(pointer.read_text(encoding='utf-8'))['directory'])
    path=directory/'report.json'
    if not path.is_file(): print('Starting; see Diagnostics > 1 for progress'); return
    for page in report_pages(json.loads(path.read_text(encoding='utf-8'))):
        print(page,flush=True)
        if input('Enter: next/return; q: menu: ').strip().lower()=='q': break


MAIN_MENU=(
    ('update','Update repository'),
    ('video-check','Check host video processing (no board access)'),
    ('video','Run 3 video frames (pipeline, display + MP4 + summary)'),
    ('video-continuous','Run selected video (default 30 frames; 0 = entire video)'),
    ('video-report','Saved video report / golden / PNGs (offline)'),
    ('performance','Timing summary / slowest nodes (offline, 3 pages)'),
    ('accuracy','Detection quality comparisons (offline PNGs)'),
    ('diagnostics','Diagnostics submenu'),
    ('cleanup','Clean disposable files (no board access)'),
    ('optimize','Whole-pipeline acceptance / automatic tuning (background)'),
    ('optimization-report','Whole-pipeline acceptance report (3 pages)'),
)
DIAGNOSTIC_MENU=(
    ('progress','Background diagnostic progress / log'),
    ('full-check','Full BUS-only check (background, reuse saved evidence)'),
    ('full-report','Saved BUS summary (offline, 3 pages)'),
    ('restore','Reload normal demo firmware'),
    ('pack-probe','First input-block probe (preserve original input)'),
    ('normal-replay','Load current firmware + preserved-input replay/golden/cycles'),
)


def action_handlers():
    return {'update':update,'video-check':lambda:video(check=True),'video':video,
            'video-continuous':lambda:video(continuous=True),'video-report':video_pages,
            'performance':performance_pages,'accuracy':accuracy_pages,
            'diagnostics':diagnostics,'cleanup':cleanup,'progress':progress,
            'full-check':launch_full,'full-report':full_pages,'restore':restore,
            'pack-probe':pack_probe,
            'normal-replay':normal_replay,
            'optimize':launch_optimization,'optimization-report':optimization_pages,
            # Retain named developer commands without duplicating menu entries.
            'capture':launch,'status':lambda:pages('status'),
            'difference':lambda:pages('difference'),'png':export_png}


def menu_loop(title,entries,exit_label,handlers,aliases=None):
    aliases=aliases or {}
    while True:
        print('\n'+title)
        for number,(_,label) in enumerate(entries,1):
            print('{} {}'.format(number,label))
        print('0 '+exit_label)
        answer=input('Choice: ').strip()
        if answer=='0':
            return
        choices={str(number):name for number,(name,_) in enumerate(entries,1)}
        action=choices.get(answer,aliases.get(answer))
        if action is None:
            print('Choose one of the displayed options.')
            continue
        try:
            handlers[action]()
            if action=='update':
                # Reload updated menu code only after a successful fast-forward pull.
                import os
                os.execv(sys.executable,[sys.executable,str(ROOT/'tools/host_control.py')])
        except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
            print('ERROR:',exc)


def diagnostics():
    menu_loop('YOLO diagnostics',DIAGNOSTIC_MENU,'Return to main menu',action_handlers())


def main():
    handlers=action_handlers()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=tuple(handlers))
    args=parser.parse_args()
    if sys.platform!='linux':
        parser.error('Run this menu on the Linux host.')
    try:
        if args.action:
            handlers[args.action]()
        else:
            # The unchanged golden runner may still print its old report number.
            # Keep that offline alias so existing golden cache hashes stay valid.
            menu_loop('YOLO host menu',MAIN_MENU,'Exit',handlers,{'17':'video-report'})
        return 0
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        print('ERROR:',exc)
        return 1
    except (EOFError,KeyboardInterrupt):
        print('\nMenu closed; a background capture remains independent.')
        return 0


if __name__=='__main__':
    sys.exit(main())
