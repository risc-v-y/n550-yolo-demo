"""Test actual C FP16 packing and old/new AMU kernels. Linux + repo tools only."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT/'yolo26_riscv'


class Packed(C.Structure):
    _fields_ = [('data', C.c_void_p), ('m', C.c_size_t), ('k', C.c_size_t), ('bytes', C.c_size_t)]


def storage(size):
    owner = C.create_string_buffer(size+64)
    return owner, (C.addressof(owner)+63)//64*64


class PackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        binary = Path(cls.temp.name)/'pack.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                        '-shared', '-fPIC', '-DAMU_PACK_ONLY', str(SRC/'amu.c'), '-o', str(binary)], check=True)
        cls.lib = C.CDLL(str(binary))
        cls.lib.amu_packed_weight_bytes.argtypes = [C.c_size_t, C.c_size_t]
        cls.lib.amu_packed_weight_bytes.restype = C.c_size_t
        cls.lib.amu_half.argtypes = [C.c_float, C.POINTER(C.c_uint16)]
        cls.lib.amu_pack_weights.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t,
                                            C.c_void_p, C.c_size_t, C.POINTER(Packed)]

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_fp16_rounding_and_invalid_values(self):
        rng = random.Random(20261019)
        bits = [0, 0x80000000, 0x33000000, 0x33800000, 0x387fc000, 0x38800000,
                0x3f801000, 0x3f803000, 0x477fe000, 0x477ff000, 0x7f800000,
                0xff800000, 0x7fc00000]+[rng.getrandbits(32) for _ in range(20000)]
        for word in bits:
            value = struct.unpack('<f', struct.pack('<I', word))[0]
            actual = C.c_uint16()
            status = self.lib.amu_half(value, C.byref(actual))
            try:
                expected = struct.unpack('<H', struct.pack('<e', value))[0]
                valid = expected & 0x7c00 != 0x7c00
            except OverflowError: valid = False
            self.assertEqual(status == 0, valid, hex(word))
            if valid: self.assertEqual(actual.value, expected, hex(word))

    def check_pack(self, raw, m, k, lda):
        size = self.lib.amu_packed_weight_bytes(m, k)
        owner, address = storage(size)
        source = C.create_string_buffer(raw)
        descriptor = Packed()
        self.assertEqual(self.lib.amu_pack_weights(source, m, k, lda, address, size, C.byref(descriptor)), 0)
        self.assertEqual((descriptor.data, descriptor.m, descriptor.k, descriptor.bytes), (address, m, k, size))
        expected = bytearray(size)
        kt = (k+31)//32
        for row in range(m):
            for j in range(k):
                value = struct.unpack_from('<f', raw, (row*lda+j)*4)[0]
                offset = (((row//32)*kt+j//32)*1024+(row%32)*32+j%32)*2
                struct.pack_into('<e', expected, offset, value)
        self.assertEqual(C.string_at(address, size), expected)
        return size

    def test_tails_stride_padding_and_rejections(self):
        for m, k in ((1, 1), (16, 27), (32, 32), (33, 33), (65, 65), (3, 576), (1, 1152)):
            lda = k+3
            raw = b''.join(struct.pack('<f', (i%127-63)/17.) for i in range(m*lda))
            self.check_pack(raw, m, k, lda)
        owner, address = storage(2048)
        source = (C.c_float*4)(1, 2, 3, 4)
        for m, k, lda, target, capacity in ((1, 1, 1, address+1, 2048), (1, 1, 1, address, 2047),
                                          (1, 2, 1, address, 2048), (0, 1, 1, address, 2048)):
            descriptor = Packed(123, 1, 1, 2048)
            self.assertNotEqual(self.lib.amu_pack_weights(source, m, k, lda, target, capacity, C.byref(descriptor)), 0)
            self.assertEqual(descriptor.data, None)
        source[0] = float('inf')
        descriptor = Packed()
        self.assertNotEqual(self.lib.amu_pack_weights(source, 1, 1, 1, address, 2048, C.byref(descriptor)), 0)
        self.assertEqual(descriptor.data, None)
        self.assertEqual(self.lib.amu_packed_weight_bytes(C.c_size_t(-1).value, 33), 0)

    def test_all_frozen_amu_weights(self):
        manifest = json.loads((ROOT/'reference/nodes/manifest.json').read_text())
        raw = (ROOT/'model/constants.bin').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['model_files_sha256']['constants.bin'])
        total = values = layers = 0
        for node in manifest['nodes']:
            if node['op'] != 0 or node['params'][4] != 1: continue
            weight = manifest['tensors'][node['inputs'][1]]
            m, ic, kh, kw = weight['shape']
            k = ic*kh*kw
            total += self.check_pack(raw[weight['offset']:weight['offset']+weight['bytes']], m, k, k)
            layers += 1; values += m*k
        self.assertEqual((layers, values, total), (94, 2391664, 5033984))
        print('PASS all 94 frozen weights: 2,391,664 active FP16 bits and zero padding', flush=True)


def qemu_test():
    tc = str(ROOT/'toolchain/riscv-elf-toolchain/bin/riscv64-unknown-elf')
    qemu = str(ROOT/'toolchain/qemu_linux/bin/qemu-system-riscv64')
    out = SRC/'build/n550-board'
    out.mkdir(parents=True, exist_ok=True)
    elf = out/'prepacked-test.elf'
    flags = ['-march=rv64gcv_zicbom_zifencei_xewmatrix1p0', '-mabi=lp64d', '-mcmodel=medany',
             '-mno-relax', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-ffreestanding',
             '-fno-builtin', '-fno-tree-vectorize', '-mstringop-strategy=scalar', '-ffp-contract=off',
             '-ffunction-sections', '-fdata-sections', '-DMODEL_PROFILE=1', '-nostdlib', '-nostartfiles',
             '-Wl,--gc-sections', '-T', str(SRC/'board/ddr.ld')]
    subprocess.run([tc+'-gcc', *flags, str(SRC/'board/start.S'), str(SRC/'amu.c'),
                    str(SRC/'board/tests/prepacked_qemu.c'), '-o', str(elf)], check=True)
    result = subprocess.run([qemu, '-machine', 'virt', '-cpu',
                             'rv64,v=true,vlen=512,matrix=on,xtlen=65536,xtrlen=512,xelen=32',
                             '-m', '128M', '-smp', '1', '-nographic', '-monitor', 'none', '-serial', 'none',
                             '-bios', 'none', '-kernel', str(elf), '-semihosting-config',
                             'enable=on,target=native', '-no-reboot'], capture_output=True, timeout=120)
    text = (result.stdout+result.stderr).decode(errors='replace')
    print(text, end='', flush=True)
    if result.returncode or 'PREPACKED AMU PASS' not in text:
        raise RuntimeError('Prepacked AMU QEMU test failed')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PackingTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    qemu_test()
