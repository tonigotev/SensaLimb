#ifndef MAIN_H
#define MAIN_H

#include "stm32c0xx_hal.h"

extern I2C_HandleTypeDef  hi2c1;
extern UART_HandleTypeDef huart2;

void Error_Handler(void);

#endif /* MAIN_H */
