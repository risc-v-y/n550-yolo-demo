/* Exercise actual mailbox session/CRC and two invocations of the probe. */
#include "../pcie.h"
#include "../model_io.h"
#include "../cache.h"
#include "../tile_probe.h"
#include "../uart.h"
extern void clp_qemu_exit(uintptr_t code) __attribute__((noreturn));
static uintptr_t host_call(uintptr_t op,uintptr_t argument) {
    register uintptr_t a0 __asm__("a0")=op;
    register uintptr_t a1 __asm__("a1")=argument;
    __asm__ volatile(".option push\n.option norvc\n.balign 16\nslli zero,zero,31\n"
                     "ebreak\nsrai zero,zero,7\n.option pop"
                     : "+r"(a0),"+r"(a1) :: "memory");
    return a0;
}
static void file(const char *name,void *data,size_t bytes,int writing) {
    size_t length=0; while(name[length]) ++length;
    uintptr_t args[3]={(uintptr_t)name,writing?5:1,length};
    uintptr_t handle=host_call(1,(uintptr_t)args);
    if(handle==(uintptr_t)-1) clp_qemu_exit(10);
    uintptr_t transfer[3]={handle,(uintptr_t)data,bytes};
    if(host_call(writing?5:6,(uintptr_t)transfer) || host_call(2,(uintptr_t)&handle)) clp_qemu_exit(11);
}
static void submit(unsigned command,unsigned sequence,unsigned frame) {
    PcieRequest r={.session=1234,.frame=frame,.command=command,.sequence=sequence};
    if(command==PCIE_RUN) { r.input_bytes=board_model_input_size(); r.input_crc=board_crc32(board_model_input(),r.input_bytes); }
    r.crc=board_crc32(&r,60);
    const unsigned char *s=(const unsigned char *)&r;
    volatile unsigned char *d=(volatile unsigned char *)&board_pcie_mailbox.request;
    for(unsigned i=0;i<64;++i) d[i]=s[i];
    board_clean((const void *)d,64);
    board_pcie_mailbox.doorbell[0]=sequence;
    board_clean((const void *)board_pcie_mailbox.doorbell,64);
    board_pcie_poll();
}
void __wrap_board_pcie_loop(int failure) {
    board_pcie_start(failure);
    if(failure || board_pcie_mailbox.info.nodes!=1 || board_pcie_mailbox.info.reserved!=2 ||
       board_pcie_mailbox.info.output_bytes!=sizeof(TileProbe)) clp_qemu_exit(20);
    submit(PCIE_OPEN,1,0);
    for(unsigned i=0;i<2;++i) {
        file("input.bin",board_model_input(),board_model_input_size(),0);
        board_clean(board_model_input(),board_model_input_size());
        submit(PCIE_RUN,i+2,i+1);
        PcieStatus status=board_pcie_mailbox.status;
        if(status.state!=PCIE_DONE || status.frame!=i+1 || status.sequence!=i+2 ||
           status.output_bytes!=sizeof(TileProbe) || status.output_crc!=board_crc32(&board_tile_probe,sizeof(TileProbe)) ||
           status.crc!=board_crc32(&status,60) || tile_probe_finish()) clp_qemu_exit(21);
        file(i?"probe-2.bin":"probe-1.bin",&board_tile_probe,sizeof(TileProbe),1);
    }
    board_uart_text("QEMU TILE PROBE TWO RUNS PASS\n");
    clp_qemu_exit(0);
}
