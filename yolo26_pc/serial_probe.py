"""Read-only Linux debug serial discovery (Python 3.10+, stdlib)."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys


def read_text(path):
    try:
        return path.read_text(errors='replace').strip()
    except OSError:
        return ''


def port_info(path):
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISCHR(info.st_mode):
        raise ValueError(f'{path} is not a character device')
    aliases = []
    for directory in ('/dev/serial/by-id', '/dev/serial/by-path'):
        for alias in Path(directory).glob('*'):
            if alias.resolve() == resolved:
                aliases.append(str(alias))
    usb = {}
    device = Path('/sys/class/tty') / resolved.name / 'device'
    if device.exists():
        device = device.resolve()
        for parent in (device, *device.parents):
            if (parent / 'idVendor').exists():
                usb = {key: read_text(parent / key) for key in
                       ('idVendor', 'idProduct', 'manufacturer', 'product', 'serial')}
                break
    locks = []
    for name in {path.name, resolved.name}:
        for directory in ('/run/lock', '/var/lock'):
            lock = Path(directory) / ('LCK..' + name)
            if lock.exists() and str(lock.resolve()) not in locks:
                locks.append(str(lock.resolve()))
    return {
        'device': str(resolved), 'aliases': sorted(aliases), 'usb': usb,
        'read_write_access': os.access(resolved, os.R_OK | os.W_OK),
        'mode': stat.filemode(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid,
        'lock_files': sorted(locks), 'visible_holders': [],
    }


def discover():
    paths = []
    for entry in sorted(Path('/sys/class/tty').glob('*')):
        path = Path('/dev') / entry.name
        if (entry / 'device').exists() and path.exists():
            paths.append(path)
    ports = []
    for path in paths:
        try:
            ports.append(port_info(path))
        except (OSError, ValueError):
            continue  # A USB device may disappear during enumeration.
    return ports


def add_holders(ports):
    """Inspect only visible /proc fds; never open a serial device for discovery."""
    devices = {}
    for port in ports:
        try:
            devices.setdefault(os.stat(port['device']).st_rdev, []).append(port)
        except OSError:
            pass
    restricted = 0
    for process in Path('/proc').glob('[0-9]*'):
        try:
            descriptors = list((process / 'fd').iterdir())
        except PermissionError:
            restricted += 1
            continue
        except OSError:
            continue
        matched = set()
        for descriptor in descriptors:
            try:
                info = descriptor.stat()
                if stat.S_ISCHR(info.st_mode) and info.st_rdev in devices:
                    matched.add(info.st_rdev)
            except PermissionError:
                restricted += 1
            except OSError:
                pass
        for number in matched:
            for port in devices[number]:
                port['visible_holders'].append({
                    'pid': int(process.name), 'name': read_text(process / 'comm')})
    return restricted


def show(report, as_json):
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print('Occupancy covers visible processes only; no holder found does not prove idle.')
    print(f"Restricted process/fd inspections: {report['restricted_inspections']}")
    if not report['ports']:
        print('No candidate serial devices found. Check USB connection and Linux drivers.')
    for port in report['ports']:
        print(f"\n{port['device']}  read+write={port['read_write_access']}  "
              f"{port['mode']} uid={port['uid']} gid={port['gid']}")
        if port['usb']:
            print('  USB: ' + json.dumps(port['usb'], ensure_ascii=False))
        for alias in port['aliases']:
            print('  alias: ' + alias)
        print('  visible holders: ' + (str(port['visible_holders']) or '[]'))
        print('  lock files: ' + str(port['lock_files']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print a shareable JSON report')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Run this utility on the Linux computer physically connected to the board.')
    report = {'mode': 'list', 'ports': [],
              'restricted_inspections': 0}
    try:
        report['ports'] = discover()
        report['restricted_inspections'] = add_holders(report['ports'])
        show(report, args.json)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        report['scan'] = {'status': 'failed', 'error': str(exc)}
        show(report, args.json)
        return 1
    except KeyboardInterrupt:
        print('Cancelled.', file=sys.stderr)
        return 130



if __name__ == '__main__':
    sys.exit(main())
