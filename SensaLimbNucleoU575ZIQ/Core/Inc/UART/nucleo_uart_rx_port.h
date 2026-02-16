// Optional Nucleo UART RX integration helpers.
#ifndef NUCLEO_UART_RX_PORT_H
#define NUCLEO_UART_RX_PORT_H

#include <stddef.h>
#include <stdint.h>

#include "nucleo_uart_rx.h"

// Initialize with a static ring buffer and handlers.
void nucleo_uart_rx_port_init(UART_HandleTypeDef *huart);

// Call from main loop.
void nucleo_uart_rx_port_process(void);

// Call from UART RX callbacks (IRQ byte or DMA block).
void nucleo_uart_rx_port_on_rx_byte(UART_HandleTypeDef *huart, uint8_t byte);
void nucleo_uart_rx_port_on_rx_block(UART_HandleTypeDef *huart,
                                     const uint8_t *data,
                                     size_t len);

#endif
