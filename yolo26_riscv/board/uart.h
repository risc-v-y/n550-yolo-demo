#ifndef YOLO_BOARD_UART_H
#define YOLO_BOARD_UART_H
#include <stdint.h>
int board_uart_init(void);
int board_uart_get(unsigned char *value);
int board_uart_put(unsigned char value);
extern uint32_t board_uart_baud, board_uart_dlf_bits;
#endif
