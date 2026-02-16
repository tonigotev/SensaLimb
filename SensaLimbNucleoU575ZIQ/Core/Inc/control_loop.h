// Control loop interface.
#ifndef CONTROL_LOOP_H
#define CONTROL_LOOP_H

#include <stddef.h>

#define SAMPLE_RATE 1000
#define CHANNELS 2
#define RMS_WINDOW 50
#define RMS_DOWNSAMPLE 5
#define SAMPLES (SAMPLE_RATE / RMS_DOWNSAMPLE)

void control_init(void);
void control_tick(void);

#endif
