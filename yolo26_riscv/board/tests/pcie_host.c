/* Native protocol fixture: actual pcie.c, shared-file DDR, synthetic inference.
 * Does not test the model, RISC-V instructions, caches or physical PCIe. */
#define _DEFAULT_SOURCE
#include "../pcie.h"
#include "../model_io.h"
#include "../diagnostics.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#define INPUT ((unsigned char *)(uintptr_t)0x80800000)
#define OUTPUT ((unsigned char *)(uintptr_t)0x80a00000)
#define CONTROL ((volatile uint32_t *)(uintptr_t)0x80a20000)
unsigned char *board_model_input(void) { return INPUT; }
const unsigned char *board_model_output(void) { return OUTPUT; }
uint32_t board_model_input_size(void) { return 3*416*416*4; }
uint32_t board_model_output_size(void) { return 7200; }
int board_model_run(void) {
    ++CONTROL[0];
    while(CONTROL[2]) usleep(1000);
    if(CONTROL[1]) { board_diag_error(0x31,-77); return -2; }
    float *out=(float *)OUTPUT;
    for(unsigned i=0;i<300;++i) {
        out[6*i]=((float *)INPUT)[0]; out[6*i+1]=20;
        out[6*i+2]=100; out[6*i+3]=120;
        out[6*i+4]=i==0 ? .9f : 0; out[6*i+5]=CONTROL[3] ? 80 : 0;
    }
    if(CONTROL[4]) { uint32_t nan=0x7fc00000; memcpy(out,&nan,4); }
    board_diag_stage(DIAG_DONE);
    return 0;
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    int fd=open(argv[1],O_RDWR);
    if(fd<0) return 3;
    if(mmap((void *)(uintptr_t)0x80000000,32*1024*1024,PROT_READ|PROT_WRITE,
            MAP_SHARED|MAP_FIXED,fd,0)==MAP_FAILED) return 4;
    close(fd);
    board_diag_init(); board_pcie_init();
    /* Mirror diagnostics into fixture DDR; protocol layout remains unchanged. */
    board_pcie_mailbox.info.diagnostic_address=0x80a10000;
    PcieInfo info=board_pcie_mailbox.info;
    board_pcie_mailbox.info.crc=board_crc32(&info,60);
    board_pcie_start(0);
    puts("READY"); fflush(stdout);
    for(;;) {
        board_pcie_poll();
        memcpy((void *)(uintptr_t)0x80a10000,(const void *)&board_diag,sizeof(board_diag));
        usleep(1000);
    }
}
