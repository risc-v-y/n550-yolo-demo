"""Run the shipped pip bootstrap with a broken distutils; no board/network."""
import importlib.abc
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WHEELS = ROOT / 'host_deps/py38-linux-x86_64'
PIP = WHEELS / 'pip-24.1.2-py3-none-any.whl'
BROKEN_DISTUTILS = '''import importlib.abc, sys, sysconfig
sysconfig._PIP_USE_SYSCONFIG = False
class BrokenDistutils(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "distutils" or fullname.startswith("distutils."):
            raise ImportError("simulated host distutils/sysconfig mismatch")
sys.meta_path.insert(0, BrokenDistutils())
'''


class SetupTests(unittest.TestCase):
    def bootstrap(self):
        shell = (ROOT / 'tools/setup_host_video.sh').read_text(encoding='utf-8')
        matches = re.findall(r'PYTHONPATH="[^"\n]*pip[^"\n]*" "\$python_bin" -c \'(.*?)\'', shell, re.S)
        self.assertEqual(len(matches), 1, 'Expected one actual pip bootstrap in setup script')
        return matches[0]

    def child(self, code, *args):
        # -I excludes user configuration; insert only the supplied pip wheel.
        source = BROKEN_DISTUTILS + '\nsys.path.insert(0, {!r})\n'.format(str(PIP)) + code
        return subprocess.run([sys.executable, '-I', '-c', source, *args],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=120)

    def test_original_bootstrap_reproduces_host_failure(self):
        result = self.child('import runpy; runpy.run_module("pip", run_name="__main__")', '--version')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('simulated host distutils/sysconfig mismatch', result.stderr)

    def test_shipped_bootstrap_avoids_distutils(self):
        result = self.child(self.bootstrap(), '--version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('pip 24.1.2', result.stdout)

    def test_offline_target_install_with_broken_distutils(self):
        with tempfile.TemporaryDirectory(prefix='yolo-video-install-') as name:
            target = Path(name) / 'lib'
            wheels = [next(WHEELS.glob('numpy-*.whl')), next(WHEELS.glob('opencv_python-*.whl'))]
            # Explicit foreign tags let Linux/Windows and Python 3.8+ test the
            # real CPython-3.8 Linux installation, without importing its ELF.
            result = self.child(self.bootstrap(), '--isolated', '--disable-pip-version-check',
                                '--no-cache-dir', 'install', '--no-index', '--no-deps',
                                '--only-binary=:all:', '--platform', 'manylinux2014_x86_64',
                                '--python-version', '3.8', '--implementation', 'cp', '--abi', 'cp38',
                                '--target', str(target), *map(str, wheels))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((target / 'numpy/__init__.py').is_file())
            self.assertTrue((target / 'cv2/cv2.abi3.so').is_file())
            self.assertTrue(any((target / 'numpy.libs').iterdir()))
            self.assertTrue(any((target / 'opencv_python.libs').iterdir()))


if __name__ == '__main__': unittest.main()
