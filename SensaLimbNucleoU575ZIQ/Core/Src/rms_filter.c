// Streaming RMS + downsampling helpers.
#include "rms_filter.h"

#include <math.h>

void rms_filter_init(RmsFilter *filter,
                     size_t channels,
                     size_t sample_rate,
                     size_t window,
                     size_t downsample,
                     CircularList *lists,
                     float rms_values[CHANNELS][SAMPLE_RATE],
                     float features[CHANNELS][SAMPLES]) {
    if (!filter || !lists || !rms_values || !features) return;
    filter->channels = channels;
    filter->sample_rate = sample_rate;
    filter->window = window;
    filter->downsample = downsample;
    filter->rms_count = 0;
    filter->lists = lists;
    filter->rms_values = rms_values;
    filter->features = features;
}

void rms_filter_update(RmsFilter *filter, const uint16_t *samples) {
    if (!filter || !samples) return;
    if (filter->rms_count >= filter->sample_rate) return;

    for (size_t ch = 0; ch < filter->channels; ch++) {
        circular_list_add(&filter->lists[ch], (float)samples[ch], NULL);
        float sum_sq = circular_list_sum_sq(&filter->lists[ch]);
        filter->rms_values[ch][filter->rms_count] = sqrtf(sum_sq / (float)filter->window);
    }
    filter->rms_count++;
}

bool rms_filter_ready(const RmsFilter *filter) {
    if (!filter) return false;
    return filter->rms_count == filter->sample_rate;
}

void rms_filter_build_features(const RmsFilter *filter) {
    if (!filter || !filter->rms_values || !filter->features) return;
    size_t samples = filter->sample_rate / filter->downsample;

    for (size_t ch = 0; ch < filter->channels; ch++) {
        for (size_t j = 0; j < samples; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < filter->downsample; k++) {
                sum += filter->rms_values[ch][j * filter->downsample + k];
            }
            filter->features[ch][j] = sum / (float)filter->downsample;
        }
    }
}

void rms_filter_reset(RmsFilter *filter) {
    if (!filter) return;
    filter->rms_count = 0;
}
