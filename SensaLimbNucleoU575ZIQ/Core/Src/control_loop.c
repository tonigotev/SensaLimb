// Main control loop implementation.
#include "control_loop.h"

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#include "main.h"
#include "core_cm33.h"
#include "ad7606.h"
#include "UART/nucleo_uart_rx_port.h"
#include "rms_filter.h"
#include "ann_infer.h"

// Provided by stm32u5xx_it.c
void uart_dma_idle_init(UART_HandleTypeDef *huart);

extern UART_HandleTypeDef huart1;

static CircularList g_rms_lists[CHANNELS];
static float g_rms_values[CHANNELS][SAMPLE_RATE];
static float g_features[CHANNELS][SAMPLES];
static RmsFilter g_rms_filter;

static volatile uint32_t g_sample_tick = 0;
static volatile bool g_sampling_enabled = false;
static uint32_t g_last_pred_tick = 0;
static uint32_t g_last_stats_tick = 0;
static uint32_t g_last_led_tick = 0;
static uint32_t g_last_sample_tick_print = 0;

static uint32_t g_last_feat_cycles = 0;
static uint32_t g_last_ann_cycles = 0;
static uint32_t g_pred_count_1s = 0;
static uint64_t g_pred_cycles_sum_1s = 0;
static uint32_t g_pred_cycles_max_1s = 0;
static int64_t g_pred_sum_milli_1s = 0;
static int32_t g_pred_min_milli_1s = 0;
static int32_t g_pred_max_milli_1s = 0;
static volatile float g_angle_history_deg = 0.0f;
#define ANGLE_HISTORY_LAG_SAMPLES 100u
#define ANGLE_CTX_RING_LEN (SAMPLE_RATE + ANGLE_HISTORY_LAG_SAMPLES + 16u)
static float g_angle_ctx_ring[ANGLE_CTX_RING_LEN];
static float g_angle_ctx_window[SAMPLES];

static float angle_ctx_ring_get(uint32_t sample_idx) {
    return g_angle_ctx_ring[sample_idx % ANGLE_CTX_RING_LEN];
}

static void debug_print_pred_stats(void) {
    if (g_pred_count_1s == 0u) {
        return;
    }
    uint32_t avg_cycles = (uint32_t)(g_pred_cycles_sum_1s / g_pred_count_1s);
    uint32_t avg_us = (uint32_t)(((uint64_t)avg_cycles * 1000000ULL) / SystemCoreClock);
    uint32_t max_us = (uint32_t)(((uint64_t)g_pred_cycles_max_1s * 1000000ULL) / SystemCoreClock);
    int32_t avg_pred_milli = (int32_t)(g_pred_sum_milli_1s / (int64_t)g_pred_count_1s);
    int32_t avg_whole = avg_pred_milli / 1000;
    int32_t avg_frac = avg_pred_milli % 1000;
    if (avg_frac < 0) {
        avg_frac = -avg_frac;
    }
    int32_t min_whole = g_pred_min_milli_1s / 1000;
    int32_t min_frac = g_pred_min_milli_1s % 1000;
    if (min_frac < 0) {
        min_frac = -min_frac;
    }
    int32_t max_whole = g_pred_max_milli_1s / 1000;
    int32_t max_frac = g_pred_max_milli_1s % 1000;
    if (max_frac < 0) {
        max_frac = -max_frac;
    }
    char buf[160];
    int n = snprintf(buf, sizeof(buf),
                     "pred/s=%lu avg_pred=%ld.%03ld min=%ld.%03ld max=%ld.%03ld avg=%lu cycles (%lu us) max=%lu us\r\n",
                     (unsigned long)g_pred_count_1s,
                     (long)avg_whole, (long)avg_frac,
                     (long)min_whole, (long)min_frac,
                     (long)max_whole, (long)max_frac,
                     (unsigned long)avg_cycles,
                     (unsigned long)avg_us,
                     (unsigned long)max_us);
    if (n > 0) {
        HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)n, 10);
    }
}

static void debug_print_sample_rate(uint32_t sample_tick) {
    uint32_t delta = sample_tick - g_last_sample_tick_print;
    g_last_sample_tick_print = sample_tick;
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "samples/s=%lu\r\n", (unsigned long)delta);
    if (n > 0) {
        HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)n, 10);
    }
}

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void control_init(void) {
    dwt_init();
    ad7606_init();

    uart_dma_idle_init(&huart1);
    ann_init();

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
    control_enable_sampling();
}

void control_on_sample_tick_isr(void) {
    if (!g_sampling_enabled) {
        return;
    }
    uint16_t sample[CHANNELS];
    ad7606_read_all_channels(sample, CHANNELS);
    rms_filter_update(&g_rms_filter, sample);
    g_sample_tick++;
    g_angle_ctx_ring[g_sample_tick % ANGLE_CTX_RING_LEN] = g_angle_history_deg;
}

void control_enable_sampling(void) {
    g_sampling_enabled = true;
}

void control_tick(void) {
    uint32_t sample_tick = g_sample_tick;

    if (sample_tick >= (SAMPLE_RATE + ANGLE_HISTORY_LAG_SAMPLES) &&
        rms_filter_ready(&g_rms_filter) &&
        ((uint32_t)(sample_tick - g_last_pred_tick) >= 50u)) {
        g_last_pred_tick = sample_tick;

        uint32_t t0 = DWT->CYCCNT;
        __disable_irq();
        rms_filter_build_features(&g_rms_filter);
        uint32_t window_start = sample_tick - SAMPLE_RATE;
        for (size_t j = 0; j < SAMPLES; j++) {
            uint32_t block_start = window_start + (uint32_t)(j * RMS_DOWNSAMPLE);
            float sum = 0.0f;
            for (size_t k = 0; k < RMS_DOWNSAMPLE; k++) {
                uint32_t idx = block_start + (uint32_t)k;
                sum += angle_ctx_ring_get(idx - ANGLE_HISTORY_LAG_SAMPLES);
            }
            g_angle_ctx_window[j] = sum / (float)RMS_DOWNSAMPLE;
        }
        __enable_irq();
        g_last_feat_cycles = DWT->CYCCNT - t0;

        t0 = DWT->CYCCNT;
        float pred = ann_predict_with_history(&g_features[0][0], CHANNELS, SAMPLES, g_angle_ctx_window, SAMPLES);
        g_last_ann_cycles = DWT->CYCCNT - t0;
        if (pred < 0.0f) {
            pred = 0.0f;
        } else if (pred > 160.0f) {
            pred = 160.0f;
        }
        g_angle_history_deg = pred;
        uint32_t pred_cycles = g_last_feat_cycles + g_last_ann_cycles;
        g_pred_cycles_sum_1s += pred_cycles;
        if (pred_cycles > g_pred_cycles_max_1s) {
            g_pred_cycles_max_1s = pred_cycles;
        }
        g_pred_count_1s++;
        int32_t pred_milli = (int32_t)(pred * 1000.0f);
        g_pred_sum_milli_1s += pred_milli;
        if (g_pred_count_1s == 1u) {
            g_pred_min_milli_1s = pred_milli;
            g_pred_max_milli_1s = pred_milli;
        } else {
            if (pred_milli < g_pred_min_milli_1s) {
                g_pred_min_milli_1s = pred_milli;
            }
            if (pred_milli > g_pred_max_milli_1s) {
                g_pred_max_milli_1s = pred_milli;
            }
        }

        // motor logic here
        // if (movement_changed(pred)) {
        //     motor_set_angle(pred);
        // }
    }

    nucleo_uart_rx_port_process();

    if ((uint32_t)(sample_tick - g_last_stats_tick) >= SAMPLE_RATE) {
        debug_print_sample_rate(sample_tick);
        debug_print_pred_stats();
        g_pred_count_1s = 0;
        g_pred_cycles_sum_1s = 0;
        g_pred_cycles_max_1s = 0;
        g_pred_sum_milli_1s = 0;
        g_pred_min_milli_1s = 0;
        g_pred_max_milli_1s = 0;
        g_last_stats_tick = sample_tick;
    }
    if ((uint32_t)(sample_tick - g_last_led_tick) >= (SAMPLE_RATE / 2u)) {
        HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin);
        HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
        g_last_led_tick = sample_tick;
    }
}
