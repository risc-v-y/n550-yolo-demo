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
    la t0,board_trap_stack_end
    csrw mscratch,t0
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
2:  csrrw sp,mscratch,sp
    addi sp,sp,-256
    sd zero,0(sp)
    sd ra,8(sp)
    sd gp,24(sp)
    sd tp,32(sp)
    sd t0,40(sp)
    sd t1,48(sp)
    sd t2,56(sp)
    sd s0,64(sp)
    sd s1,72(sp)
    sd a0,80(sp)
    sd a1,88(sp)
    sd a2,96(sp)
    sd a3,104(sp)
    sd a4,112(sp)
    sd a5,120(sp)
    sd a6,128(sp)
    sd a7,136(sp)
    sd s2,144(sp)
    sd s3,152(sp)
    sd s4,160(sp)
    sd s5,168(sp)
    sd s6,176(sp)
    sd s7,184(sp)
    sd s8,192(sp)
    sd s9,200(sp)
    sd s10,208(sp)
    sd s11,216(sp)
    sd t3,224(sp)
    sd t4,232(sp)
    sd t5,240(sp)
    sd t6,248(sp)
    csrr t0,mscratch
    sd t0,16(sp)
    la t0,5f
    csrw mtvec,t0
    .option push
    .option norelax
    la gp,__global_pointer$
    .option pop
    mv a3,sp
    csrr a0,mcause
    csrr a1,mepc
    csrr a2,mtval
    tail platform_trap
.balign 4
5:  wfi
    j 5b
.section .bss.trap_stack,"aw",@nobits
.balign 64
board_trap_stack:
    .skip 4096
board_trap_stack_end:
