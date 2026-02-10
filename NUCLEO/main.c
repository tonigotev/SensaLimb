#include "control_loop.h"

int main(void) {
    control_init();
    while (1) {
        control_tick();
    }
}
