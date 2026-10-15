"""Python 3.8+ host menu. Explicit capture/probe/restore choices access the board."""
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


def launch(sample='bus',probe=False):
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
        if probe:
            command=[sys.executable,'-u',str(ROOT/'tools/tile_probe.py'),'--load','--output',str(result)]
            try:
                previous=selected()
                summary=json.loads((previous/'summary.json').read_text(encoding='utf-8'))
                if (summary.get('sample')=='bus' and (previous/'first_difference.json').is_file()
                        and (previous/'tensors/0003.bin').is_file()):
                    command+=['--capture',str(previous)]
            except ValueError:
                pass
        with (run/'terminal.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        save_json(directory/'last-launch.json',{'pid':process.pid,'result':str(result),
                                              'log':str(run/'terminal.log')})
        if probe:
            save_json(directory/'last-probe.json',{'directory':str(result)})
    print('Background '+('probe (loads diagnostic firmware)' if probe else 'capture')+' started; SSH disconnect will not stop it.')
    print('PID:',process.pid,'\nResult:',result,'\nLog:',run/'terminal.log')
    return process


def probe_pages():
    pointer=workspace()/'last-probe.json'
    if not pointer.exists(): raise ValueError('No menu probe yet; use option 8.')
    directory=Path(json.loads(pointer.read_text(encoding='utf-8'))['directory'])
    if not (directory/'summary.json').exists():
        progress(); return
    for page in (1,2,3):
        command=[sys.executable,str(ROOT/'tools/tile_probe.py'),'--report',str(directory),'--page',str(page)]
        if page==1: command.append('--reanalyze')
        subprocess.run(command,check=True)
        if input('Enter: next/return; q: menu: ').strip().lower()=='q': break


def restore():
    with guard(workspace()/'control.lock'):
        subprocess.run(['bash',str(ROOT/'tools/load_board.sh'),'--reset-dir','/rv/fpga/bin',
                        '--pbcopy','/rv/fpga/bin/pbcopy'],cwd=ROOT,check=True)


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
        raise ValueError('No saved capture report. Use option 5 to capture, or 6 to inspect its log.')
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
    if kind=='analysis':
        script=ROOT/'tools/analyze_first_node.py'
        if not script.exists():
            raise ValueError('Analysis tool not installed; update repository first.')
        numbers=(1,2,3,4)
    else:
        script=ROOT/'yolo26_pc/node_report.py'
        numbers=(1,2,3) if kind=='status' else (2,3)
    for index,page in enumerate(numbers):
        command=[sys.executable,str(script),str(directory),'--page',str(page)]
        if kind!='analysis':
            command+=['--view',kind]
        subprocess.run(command,cwd=ROOT,check=True)
        prompt='Enter: next page; q: return: ' if index<len(numbers)-1 else 'Enter: return to menu: '
        if input(prompt).strip().lower()=='q':
            break


def progress():
    pointer=workspace()/'last-launch.json'
    if not pointer.exists():
        print('No menu-launched capture. Older saved reports: options 2/3/7.')
        return
    job=json.loads(pointer.read_text(encoding='utf-8'))
    print('PID:',job['pid'],'\nResult:',job['result'],'\nLog:',job['log'])
    summary=Path(job['result'])/'summary.json'
    if summary.exists():
        print('Finished status:',json.loads(summary.read_text(encoding='utf-8')).get('status'))
    else:
        try:
            with guard(workspace()/'control.lock'):
                print('No running menu job and no final report. Inspect the log below.')
        except RuntimeError:
            print('Menu job active; select option 6 again to refresh.')
    with Path(job['log']).open(encoding='utf-8',errors='replace') as stream:
        print(''.join(deque(stream,maxlen=12)),end='')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=('update','capture','status','difference','analysis','progress','probe','probe-report','restore'))
    args=parser.parse_args()
    if sys.platform!='linux':
        parser.error('Run this menu on the Linux host.')
    actions={'1':update,'2':lambda:pages('status'),'3':lambda:pages('difference'),
             '4':lambda:pages('analysis'),'5':launch,'6':progress,'7':choose,
             '8':lambda:launch(probe=True),'9':probe_pages,'10':restore}
    names={'update':'1','status':'2','difference':'3','analysis':'4','capture':'5','progress':'6',
           'probe':'8','probe-report':'9','restore':'10'}
    while True:
        try:
            if args.action:
                actions[names[args.action]]()
                return 0
            print('\nYOLO host menu\n1 Update repository\n2 Saved capture report\n3 Saved first difference'
                  '\n4 Analyze first convolution\n5 Start BUS capture (background)'
                  '\n6 Capture/probe progress/log\n7 Select saved result'
                  '\n8 Load + run 3-tile probe (background, replaces firmware)'
                  '\n9 Reanalyze saved probe (offline, 3 pages)\n10 Reload normal demo firmware\n0 Exit')
            answer=input('Choice: ').strip()
            if answer=='0':
                return 0
            if answer not in actions:
                print('Choose 0-10.'); continue
            actions[answer]()
            if answer=='1':
                # Reload updated menu code only after a successful fast-forward pull.
                import os
                os.execv(sys.executable,[sys.executable,str(ROOT/'tools/host_control.py')])
        except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
            print('ERROR:',exc)
            if args.action:
                return 1
        except (EOFError,KeyboardInterrupt):
            print('\nMenu closed; a background capture remains independent.')
            return 0


if __name__=='__main__':
    sys.exit(main())
