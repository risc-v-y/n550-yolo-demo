OUTPUT_ARCH(riscv)
ENTRY(_start)
/* Platform hands over with DDR, clocks, UART pins and DCache enabled.
 * Reserve this entire 32 MiB region exclusively before loading via JTAG. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M }
SECTIONS {
    .text : ALIGN(64) { KEEP(*(.text.init)) *(.text .text.*) } > RAM
    .rodata : ALIGN(64) { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
    .data : ALIGN(64) {
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.sdata .sdata.* .data .data.*)
    } > RAM
    .bss (NOLOAD) : ALIGN(64) {
        __bss_start = .;
        *(.sbss .sbss.* .bss .bss.* COMMON)
        . = ALIGN(64);
        __bss_end = .;
    } > RAM
    .pcie_mailbox 0x81f00000 (NOLOAD) : ALIGN(64) {
        KEEP(*(.pcie_mailbox))
    } > RAM
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
    ASSERT(__bss_end <= 0x81f00000, "Board model overlaps PCIe mailbox reserve")
    ASSERT(SIZEOF(.pcie_mailbox) <= 256, "PCIe mailbox exceeds wire layout")
    /DISCARD/ : { *(.comment .eh_frame .eh_frame_hdr) }
}
