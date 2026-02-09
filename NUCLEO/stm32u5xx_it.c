#include "stm32u5xx_hal.h"

#include "UART/nucleo_uart_rx_port.h"

#define UART_RX_DMA_BUF_SIZE  1024u

// must replace if not using huart1
extern UART_HandleTypeDef huart1;

static UART_HandleTypeDef *g_uart = &huart1;
static uint8_t g_uart_dma_buf[UART_RX_DMA_BUF_SIZE];
static size_t g_uart_dma_last_pos = 0u;

void uart_dma_idle_init(UART_HandleTypeDef *huart) {
    if (huart) g_uart = huart;
    g_uart_dma_last_pos = 0u;

    nucleo_uart_rx_port_init(g_uart);
    (void)HAL_UART_Receive_DMA(g_uart, g_uart_dma_buf, UART_RX_DMA_BUF_SIZE);
    __HAL_UART_ENABLE_IT(g_uart, UART_IT_IDLE);
}

static void uart_dma_idle_handle(UART_HandleTypeDef *huart) {
    if (!huart) return;

    size_t pos = UART_RX_DMA_BUF_SIZE - __HAL_DMA_GET_COUNTER(huart->hdmarx);
    size_t len = (pos >= g_uart_dma_last_pos)
               ? (pos - g_uart_dma_last_pos)
               : (UART_RX_DMA_BUF_SIZE - g_uart_dma_last_pos + pos);

    if (len) {
        size_t first = UART_RX_DMA_BUF_SIZE - g_uart_dma_last_pos;
        if (len <= first) {
            nucleo_uart_rx_port_on_rx_block(huart, &g_uart_dma_buf[g_uart_dma_last_pos], len);
        } else {
            nucleo_uart_rx_port_on_rx_block(huart, &g_uart_dma_buf[g_uart_dma_last_pos], first);
            nucleo_uart_rx_port_on_rx_block(huart, &g_uart_dma_buf[0], len - first);
        }
        g_uart_dma_last_pos = pos;
    }
}

void USART1_IRQHandler(void) {
    if (__HAL_UART_GET_FLAG(g_uart, UART_FLAG_IDLE)) {
        __HAL_UART_CLEAR_IDLEFLAG(g_uart);
        uart_dma_idle_handle(g_uart);
    }
    HAL_UART_IRQHandler(g_uart);
}
