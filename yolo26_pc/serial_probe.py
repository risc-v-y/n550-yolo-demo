"""Linux serial discovery and explicit N550 HELLO probe (Python 3.10+, stdlib)."""
import argparse
import errno
import json
import os
from pathlib import Path
import select
import stat
import sys
import time


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


class LinuxSerial:
    """Small transport for UARTBackend; raw 8N1 with exclusive-open protection."""
    def __init__(self, path, baud):
        import fcntl
        import termios

        speed = getattr(termios, f'B{baud}', None)
        if speed is None:
            raise ValueError(f'Unsupported baud rate: {baud}')
        self.fd = None
        self.original = None
        self.exclusive = False
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            # flock cooperates with other probes; TIOCEXCL rejects new tty opens.
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            self.exclusive = True
            self.original = termios.tcgetattr(self.fd)
            raw = termios.tcgetattr(self.fd)
            raw[0] = raw[1] = raw[3] = 0
            raw[2] = termios.CLOCAL | termios.CREAD | termios.CS8
            raw[4] = raw[5] = speed
            raw[6][termios.VMIN] = raw[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, raw)
        except BaseException:
            self.close()
            raise

    def read(self, count):
        if not select.select([self.fd], [], [], .1)[0]:
            return b''
        try:
            data = os.read(self.fd, count)
        except BlockingIOError:
            return b''
        if not data:
            raise OSError('Serial device disconnected')
        return data

    def write(self, data):
        if not select.select([], [self.fd], [], 3)[1]:
            raise TimeoutError('Serial write timed out')
        try:
            return os.write(self.fd, data)
        except BlockingIOError:
            return 0

    def close(self):
        import fcntl
        import termios

        if self.fd is None:
            return
        try:
            if self.original is not None:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.original)
        except OSError:
            pass
        finally:
            try:
                if self.exclusive:
                    fcntl.ioctl(self.fd, termios.TIOCNXCL)
            except OSError:
                pass
            finally:
                os.close(self.fd)
                self.fd = None


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
    if 'probe' in report:
        print('\nProbe: ' + json.dumps(report['probe'], ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, metavar='DEVICE',
                        help='Send HELLO to ONLY this device; requires running demo.elf')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--json', action='store_true', help='Print a shareable JSON report')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Run this utility on the Linux computer physically connected to the board.')
    report = {'mode': 'probe' if args.probe else 'list', 'ports': [],
              'restricted_inspections': 0}
    transport = None
    try:
        report['ports'] = [port_info(args.probe)] if args.probe else discover()
        report['restricted_inspections'] = add_holders(report['ports'])
        if args.probe:
            port = report['ports'][0]
            if not port['read_write_access']:
                raise PermissionError('No read/write access; ask the administrator to grant serial-device access.')
            if port['visible_holders'] or port['lock_files']:
                raise OSError(errno.EBUSY, 'Port has an open holder or lock file; close the terminal/demo first.')
            from uart_backend import UARTBackend

            started = time.monotonic()
            transport = LinuxSerial(port['device'], args.baud)
            backend = UARTBackend(port['device'], baud=args.baud, transport=transport,
                                  progress=lambda message: print(message, file=sys.stderr))
            if backend.info['dlf_bits'] not in (4, 5, 6) or backend.info['actual_baud'] <= 0:
                raise ValueError('Invalid UART divider/baud in board INFO')
            report['probe'] = {'status': 'passed', 'board': backend.info,
                               'seconds': round(time.monotonic() - started, 3)}
        show(report, args.json)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        report['probe' if args.probe else 'scan'] = {'status': 'failed', 'error': str(exc)}
        show(report, args.json)
        return 1
    except KeyboardInterrupt:
        print('Cancelled.', file=sys.stderr)
        return 130
    finally:
        if transport is not None:
            transport.close()


if __name__ == '__main__':
    sys.exit(main())
