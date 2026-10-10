#ifndef YOLO_BOARD_UART_H
#define YOLO_BOARD_UART_H
#include <stdint.h>
int board_uart_init(void);
int board_uart_put(unsigned char value);
int board_uart_text(const char *text);
int board_uart_u64(uint64_t value);
int board_uart_i64(int64_t value);
int board_uart_hex(uint64_t value);
extern uint32_t board_uart_baud, board_uart_dlf_bits;
/* Zero means ready; negative means uninitialized, init failure or TX timeout.
 * A failed debug UART is disabled until explicitly initialized again. */
extern volatile int board_uart_status;
#endif
