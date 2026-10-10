#ifndef YOLO_BOARD_MODEL_IO_H
#define YOLO_BOARD_MODEL_IO_H
#include <stdint.h>
#include <stddef.h>
uint32_t board_crc32(const void *data, size_t bytes);
unsigned char *board_model_input(void);
const unsigned char *board_model_output(void);
uint32_t board_model_input_size(void);
uint32_t board_model_output_size(void);
int board_model_run(void);
#endif
