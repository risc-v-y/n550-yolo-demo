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
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
    ASSERT(__bss_end <= __stack_top - 256K, "Board model exceeds reserved DDR region")
    /DISCARD/ : { *(.comment .eh_frame .eh_frame_hdr) }
}
