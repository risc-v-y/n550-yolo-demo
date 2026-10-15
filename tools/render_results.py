"""Export saved detection drawings as PNG, entirely offline, without board I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'yolo26_pc'))
from png_output import write_png,png_size


def read_json(path): return json.loads(path.read_text(encoding='utf-8'))


def drawings(directory):
    directory=Path(directory)
    if directory.is_file():
        if directory.suffix.lower() not in ('.svg','.png'): raise ValueError('Select a result directory, SVG or PNG')
        return [directory.resolve()]
    if not directory.is_dir(): return []
    paths=list(directory.rglob('*.svg'))
    paths+=list(directory.rglob('*annotated.png'))+list(directory.rglob('*detected.png'))
    summary=directory/'summary.json'
    if summary.is_file():
        # Reused captures may live outside this full-check report directory.
        for sample in read_json(summary).get('samples',{}).values():
            if not isinstance(sample,dict) or not sample.get('directory'): continue
            capture=Path(sample['directory'])
            if not capture.is_absolute(): capture=directory/capture
            if capture.resolve()!=directory.resolve() and capture.is_dir():
                paths+=list(capture.glob('*.svg'))+list(capture.glob('*annotated.png'))
    return sorted(set(p.resolve() for p in paths))


def latest():
    base=ROOT/'yolo26_pc/outputs'; candidates=[]
    for name,key in (('host-menu/last-full.json','directory'),('host-menu/selected.json','directory'),
                     ('node-capture/latest.json','directory')):
        pointer=base/name
        if pointer.is_file(): candidates.append(Path(read_json(pointer)[key]))
    candidates+=sorted((p for pattern in ('board-pcie/*','node-capture/*','host-menu/run-*/capture')
                         for p in base.glob(pattern) if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True)
    seen=set()
    for directory in candidates:
        if directory in seen: continue
        seen.add(directory)
        if drawings(directory): return directory
        print('No saved drawing in '+str(directory)+'; checking earlier results.',flush=True)
    raise ValueError('No saved SVG/PNG drawings. A run that failed before capture has no image to export.')


def export(directory):
    paths=drawings(directory)
    if not paths: raise ValueError('No saved drawings in '+str(directory))
    entries=[]
    # A source SVG is authoritative when a same-named PNG also exists.
    outputs=set()
    for path in sorted(paths,key=lambda p:(p.suffix.lower()!='.svg',str(p))):
        target=path.with_suffix('.png')
        if target in outputs: continue
        source=path.read_bytes()
        size=write_png(source,target) if path.suffix.lower()=='.svg' else png_size(source)
        raw=target.read_bytes(); outputs.add(target)
        entries.append({'source':str(path),'source_sha256':hashlib.sha256(source).hexdigest(),
                        'png':str(target),'png_sha256':hashlib.sha256(raw).hexdigest(),'size':list(size)})
        print('PNG: {} ({}x{})'.format(target,*size),flush=True)
    base=Path(directory) if Path(directory).is_dir() else Path(directory).parent
    # Preserve numerical summaries and original SVGs. Only add an export sidecar.
    temporary=base/'png-export.tmp'
    temporary.write_text(json.dumps({'version':1,'images':entries},indent=2)+'\n',encoding='utf-8')
    temporary.replace(base/'png-export.json')
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',nargs='?',type=Path)
    p.add_argument('--latest',action='store_true',help='Find the latest available drawing, including older results if needed')
    args=p.parse_args()
    if bool(args.directory)==bool(args.latest): p.error('Choose a directory/file OR --latest')
    try:
        directory=latest() if args.latest else args.directory
        print('Saved source: '+str(directory),flush=True)
        entries=export(directory)
        print('PNG export complete: {} image(s). No inference or PCIe access.'.format(len(entries)),flush=True)
        return 0
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print('PNG export failed: '+str(exc),file=sys.stderr); return 1


if __name__=='__main__': raise SystemExit(main())
