# Source from the repository root in an already connected, halted RISC-V GDB.
# These addresses belong only to the shipped firmware/selftest.{elf,bin}.
python
import hashlib
from pathlib import Path
import gdb


class _YoloRvvStoreProbe:
    ENTRY = 0x80000000
    BEFORE = 0x80000b26
    AFTER = 0x80000c2a
    PREPARE = (0x800001dc, 0x800001ec, 0x80000a8e)
    HASHES = {
        "selftest.elf": "98113263f62b1dfea37aeeb9ea40af2e9815819ae7d2111129889129de5d23f4",
        "selftest.bin": "a55a99f26466bf9e4d5125fc7eb106b6734a047aa869797489eb6bb799a1025d",
    }

    def __init__(self):
        self.ready = None

    @staticmethod
    def say(message):
        gdb.write("[YOLO RVV probe] " + message + "\n")

    @staticmethod
    def stopped():
        if gdb.parameter("non-stop"):
            raise gdb.GdbError("Use all-stop GDB mode (non-stop must be off).")
        thread = gdb.selected_thread()
        if thread is None or not thread.is_stopped():
            raise gdb.GdbError("Connect to the target and halt the boot hart first.")
        gdb.execute("frame 0", to_string=True)
        return (gdb.selected_inferior().num, thread.global_num)

    @staticmethod
    def pc():
        return int(gdb.parse_and_eval("$pc"))

    @staticmethod
    def no_breakpoints():
        active = [bp.number for bp in (gdb.breakpoints() or ())
                  if bp.is_valid() and bp.enabled]
        if active:
            raise gdb.GdbError("Enabled breakpoints/watchpoints already exist: "
                               + ", ".join(map(str, active))
                               + ". Review and delete or disable them first.")

    def verify_firmware(self):
        firmware = Path.cwd() / "firmware"
        for name, expected in self.HASHES.items():
            path = firmware / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise gdb.GdbError("Wrong/missing " + str(path)
                                   + "; cd to the repository root and use the shipped selftest.")
        image = (firmware / "selftest.bin").read_bytes()
        inferior = gdb.selected_inferior()
        # Check the entry and every probe location, without reading test buffers.
        for address, size in [(self.ENTRY, 256)] + [
                (address, 8) for address in self.PREPARE + (self.BEFORE, self.AFTER)]:
            offset = address - self.ENTRY
            if bytes(inferior.read_memory(address, size)) != image[offset:offset + size]:
                raise gdb.GdbError("Target code differs at 0x%x; load selftest.bin first." % address)
        elf = str(firmware / "selftest.elf")
        quoted = '"' + elf.replace("\\", "\\\\").replace('"', '\\"') + '"'
        gdb.execute("symbol-file " + quoted, to_string=True)
        self.say("Selftest hashes and target code checkpoints match; symbols loaded (no download).")

    def advance(self, address, context):
        if self.stopped() != context:
            raise gdb.GdbError("The selected inferior/hart changed; restart the probe.")
        self.no_breakpoints()
        breakpoint = None
        try:
            breakpoint = gdb.Breakpoint("*0x%x" % address,
                                        type=gdb.BP_HARDWARE_BREAKPOINT, temporary=True)
            self.say("Continue to 0x%x with one temporary hardware breakpoint. "
                     "If it is not reached, press Ctrl+C to stop and report the PC." % address)
            gdb.execute("continue")
            if self.stopped() != context or self.pc() != address:
                raise gdb.GdbError("Unexpected stop at 0x%x (wanted 0x%x)." % (self.pc(), address))
        finally:
            # Temporary breakpoints disappear only when hit; also remove missed ones.
            if breakpoint is not None and breakpoint.is_valid():
                breakpoint.delete()
        self.say("Stopped at 0x%x; the probe breakpoint has been removed." % address)

    def failure_state(self):
        try:
            self.stopped()
            self.say("Halted PC=0x%x. Probe state cleared; no automatic retry." % self.pc())
            inferior = gdb.selected_inferior()
            fields = (("test_stage", 0x800033a0, 4), ("error", 0x80003398, 8),
                      ("detail", 0x80003fb8, 8))
            values = ["%s=%d" % (name, int.from_bytes(bytes(inferior.read_memory(addr, size)),
                                                     "little", signed=(name == "detail")))
                      for name, addr, size in fields]
            self.say("Last retained diagnostics: " + " ".join(values))
        except Exception as exc:
            self.say("Could not read halted diagnostics: " + str(exc))

    def run(self, after):
        started = False
        try:
            context = self.stopped()
            self.no_breakpoints()
            if after:
                if self.ready != context or self.pc() != self.BEFORE:
                    raise gdb.GdbError("Run yolo-rvv-before first and stay halted at 0x%x." % self.BEFORE)
                self.ready = None
                started = True
                self.advance(self.AFTER, context)
                self.say("AFTER: cache invalidation/fences finished; result checks have not started.")
            else:
                self.ready = None
                self.verify_firmware()
                started = True
                # Re-enter startup: clear BSS and rerun selftest, without a board reset or load.
                gdb.execute("set $pc = 0x%x" % self.ENTRY)
                for address in self.PREPARE + (self.BEFORE,):
                    self.advance(address, context)
                self.ready = context
                self.say("BEFORE: all three RVV output rows and fences executed; cache invalidation has not started.")
            self.say("Keep GDB halted. In another host terminal at the repository root:")
            self.say("# Read 192 bytes of vector_out (three rows), preserving raw DDR data and logs.")
            self.say("python3 tools/read_ddr.py 0x80003500 0xc0 --width 32")
            self.say("Expected active row starts: 40400000, 41300000, 41d00000; guards: c49a4000.")
        except BaseException as exc:
            self.ready = None
            if started:
                self.failure_state()
            raise gdb.GdbError("Probe aborted: " + (str(exc) or type(exc).__name__)) from None


class _YoloRvvProbeCommand(gdb.Command):
    """Probe shipped selftest RVV stores using one temporary hardware breakpoint.

    yolo-rvv-before restarts selftest and stops before cache invalidation.
    yolo-rvv-after resumes that session and stops before result comparison.
    Read DDR in another terminal while halted at each point.
    Requires a connected, halted boot hart, GDB Python and the repository cwd.
    """

    def __init__(self, name, probe, after):
        super().__init__(name, gdb.COMMAND_USER)
        self.probe = probe
        self.after = after

    def invoke(self, argument, from_tty):
        self.dont_repeat()
        if argument.strip():
            raise gdb.GdbError("This command takes no arguments; cd to the repository root first.")
        self.probe.run(self.after)


# Re-sourcing the script preserves the current BEFORE/AFTER session.
if "_yolo_rvv_store_probe" not in globals():
    _yolo_rvv_store_probe = _YoloRvvStoreProbe()
    _YoloRvvProbeCommand("yolo-rvv-before", _yolo_rvv_store_probe, False)
    _YoloRvvProbeCommand("yolo-rvv-after", _yolo_rvv_store_probe, True)
gdb.write("[YOLO RVV probe] Commands ready: yolo-rvv-before, yolo-rvv-after. Target has not been resumed.\n")
end
