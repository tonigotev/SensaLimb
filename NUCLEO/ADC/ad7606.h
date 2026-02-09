#ifndef __AD7606_H
#define __AD7606_H

#include <stdint.h>

void ad7606_init(void);
void ad7606_read_all_channels(uint16_t *channels, uint8_t num_channels);
void ad7606_normalize_channels(const uint16_t *channels, float *out, uint8_t num_channels);

#endif