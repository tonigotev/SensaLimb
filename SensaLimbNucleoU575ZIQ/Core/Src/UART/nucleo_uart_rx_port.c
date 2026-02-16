// Optional Nucleo UART RX integration helpers.
#include "nucleo_uart_rx_port.h"

#define NUCLEO_UART_RX_RING_SIZE   (NUCLEO_UART_MAX_FRAME_LEN * 4u)

static UART_HandleTypeDef *g_port_huart = NULL;
static uint8_t g_port_ring[NUCLEO_UART_RX_RING_SIZE];

void nucleo_uart_rx_port_init(UART_HandleTypeDef *huart) {
    g_port_huart = huart;
    nucleo_uart_rx_init(huart, g_port_ring, sizeof(g_port_ring));
}

void nucleo_uart_rx_port_process(void) {
    nucleo_uart_rx_process();
}

void nucleo_uart_rx_port_on_rx_byte(UART_HandleTypeDef *huart, uint8_t byte) {
    if (huart != g_port_huart) return;
    nucleo_uart_rx_push_byte(byte);
}

void nucleo_uart_rx_port_on_rx_block(UART_HandleTypeDef *huart,
                                     const uint8_t *data,
                                     size_t len) {
    if (huart != g_port_huart) return;
    nucleo_uart_rx_push_block(data, len);
}
