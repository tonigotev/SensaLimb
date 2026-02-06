// Stub action implementations for BMS UART frames.
#include "nucleo_uart_rx.h"

void nucleo_on_safe_req(uint16_t fault_code) {
    (void)fault_code;
}

void nucleo_on_scd_event(void) {
}

void nucleo_on_low_batt_mode(void) {
}

void nucleo_on_low_batt_lock(void) {
}

void nucleo_on_cur_latched(void) {
}

void nucleo_on_low_batt_warn(uint8_t c1_pct, uint8_t c2_pct) {
    (void)c1_pct;
    (void)c2_pct;
}

void nucleo_on_last_fault(uint16_t reason, uint32_t count) {
    (void)reason;
    (void)count;
}
