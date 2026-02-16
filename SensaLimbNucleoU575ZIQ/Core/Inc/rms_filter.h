// Streaming RMS + downsampling helpers.
#ifndef RMS_FILTER_H
#define RMS_FILTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "circular_list.h"

typedef struct {
    size_t channels;
    size_t sample_rate;
    size_t window;
    size_t downsample;
    size_t rms_count;
    CircularList *lists;
    float (*rms_values)[SAMPLE_RATE];
    float (*features)[SAMPLES];
} RmsFilter;

void rms_filter_init(RmsFilter *filter,
                     size_t channels,
                     size_t sample_rate,
                     size_t window,
                     size_t downsample,
                     CircularList *lists,
                     float rms_values[CHANNELS][SAMPLE_RATE],
                     float features[CHANNELS][SAMPLES]);

void rms_filter_update(RmsFilter *filter, const uint16_t *samples);
bool rms_filter_ready(const RmsFilter *filter);
void rms_filter_build_features(const RmsFilter *filter);
void rms_filter_reset(RmsFilter *filter);

#endif
