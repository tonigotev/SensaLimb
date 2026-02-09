#include <stdint.h>
#include <math.h>
#include "ad7606.h"
#include "ad7606_pins.h"
#include "core_cm33.h"

static inline void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline void delay_us(uint32_t us) {
    uint32_t cycles = (uint32_t)(((uint64_t)SystemCoreClock * us) / 1000000ULL);
    uint32_t start = DWT->CYCCNT;
    while ((DWT->CYCCNT - start) < cycles) {}
}

void main(void) {
    dwt_init();
    ad7606_init();

    uint16_t channels[2000][DEFINED_CHANNELS];
    // 2khz sampling rate on 180mhz clock
    // 180mhz / 2khz = 90k cycles per sample
    // 1/180mhz = 5.56ns
    // 90k * 5.56ns = 500.4us
    uint32_t period = convert_us_to_cycles(500);
    uint32_t t = DWT->CYCCNT;
    for (int i = 0; i < 2000; i++) {
        ad7606_read_all_channels(channels[i], DEFINED_CHANNELS);
        while(DWT->CYCCNT - t < period){}
        t += period;
    }

    // We downsample by 10 to get 200 samples
    // We average the samples to get the RMS
    float sum[DEFINED_CHANNELS] = {0};
    float rms[200][DEFINED_CHANNELS] = {0};
    for (int g = 0; g < 200; g++) {
        for (int i = 0; i < DEFINED_CHANNELS; i++) {
            for (int j = 0; j < 10; j++) {
                float v = (float)channels[g * 10 + j][i];
                sum[i] += v * v;
            }
            rms[g][i] = sqrtf(sum[i] / 10.0f);
            sum[i] = 0.0f;
        }
    }

}
