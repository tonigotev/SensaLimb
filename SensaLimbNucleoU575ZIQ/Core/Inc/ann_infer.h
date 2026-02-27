// ANN inference stubs.
#ifndef ANN_INFER_H
#define ANN_INFER_H

#include <stddef.h>

void ann_init(void);
float ann_predict(const float *features, size_t channels, size_t samples, float angle_value);
float ann_predict_with_history(const float *features,
                               size_t channels,
                               size_t samples,
                               const float *angle_history,
                               size_t history_samples);

#endif
