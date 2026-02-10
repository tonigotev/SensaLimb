// Main control loop implementation.
#include "control_loop.h"

#include <stdint.h>

#include "core_cm33.h"
#include "ad7606.h"
#include "UART/nucleo_uart_rx_port.h"
#include "rms_filter.h"
#include "ann_infer.h"

// Provided by stm32u5xx_it.c
void uart_dma_idle_init(UART_HandleTypeDef *huart);

// Replace with your actual UART handle.
extern UART_HandleTypeDef huartX;

static CircularList g_rms_lists[CHANNELS];
static float g_rms_values[CHANNELS][SAMPLE_RATE];
static float g_features[CHANNELS][SAMPLES];
static RmsFilter g_rms_filter;
static uint16_t g_sample[CHANNELS];

static uint32_t g_period_cycles = 0;
static uint32_t g_tick_start = 0;

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void control_init(void) {
    dwt_init();
    ad7606_init();

    uart_dma_idle_init(&huartX);

    for (size_t i = 0; i < CHANNELS; i++) {
        circular_list_init(&g_rms_lists[i], RMS_WINDOW);
    }

    rms_filter_init(&g_rms_filter,
                    CHANNELS,
                    SAMPLE_RATE,
                    RMS_WINDOW,
                    RMS_DOWNSAMPLE,
                    g_rms_lists,
                    g_rms_values,
                    g_features);

    g_period_cycles = (uint32_t)((uint64_t)SystemCoreClock / SAMPLE_RATE);
    g_tick_start = DWT->CYCCNT;
}

void control_tick(void) {
    ad7606_read_all_channels(g_sample, CHANNELS);
    rms_filter_update(&g_rms_filter, g_sample);

    if (rms_filter_ready(&g_rms_filter)) {
        rms_filter_build_features(&g_rms_filter);
        rms_filter_reset(&g_rms_filter);

        ann_parse(&g_features[0][0], CHANNELS, SAMPLES);

        // motor logic here
        // if (movement_changed(pred)) {
        //     motor_set_angle(pred);
        // }
    }

    nucleo_uart_rx_port_process();
    while (DWT->CYCCNT - g_tick_start < g_period_cycles) {}
    g_tick_start += g_period_cycles;
}
