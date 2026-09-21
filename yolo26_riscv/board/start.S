/* M-mode handoff: DDR/clocks/pins/DCache configured by the platform. */
.section .text.init,"ax",@progbits
.globl _start
_start:
    csrci mstatus,8
    csrw mie,zero
    csrr t0,mhartid
    bnez t0,1f
    .option push
    .option norelax
    la gp,__global_pointer$
    .option pop
    la sp,__stack_top
    la t0,2f
    csrw mtvec,t0
    li t0,0x6600
    csrs mstatus,t0
    csrw fcsr,zero
    fence.i
    la t0,__bss_start
    la t1,__bss_end
3:  bgeu t0,t1,4f
    sd zero,0(t0)
    addi t0,t0,8
    j 3b
4:  call main
    tail platform_exit
1:  wfi
    j 1b
.balign 4
2:  csrr a0,mcause
    csrr a1,mepc
    csrr a2,mtval
    tail platform_trap
