#ifndef __AD7606_H
#define __AD7606_H

#include <stdint.h>

// Force fake ADC data path; set to 0 when using real ADC.
#define AD7606_FORCE_FAKE 0

// Debug: hold CS and RD statically low so you can meter them at the chip.
// Set to 1, flash, measure CS+RD at ADC pins, then set back to 0.
#define AD7606_DEBUG_HOLD_CS_RD_LOW 0

void ad7606_init(void);
void ad7606_read_all_channels(uint16_t *channels, uint8_t num_channels);
void ad7606_normalize_channels(const uint16_t *channels, float *out, uint8_t num_channels);
uint32_t ad7606_get_busy_miss_count(void);
uint32_t ad7606_get_busy_seen_count(void);

#endif
