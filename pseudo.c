#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "bq.h"

// Boot sequence
void system_boot(void) {
    bms_i2c_init();
  
    bq_enter_cfgupdate();
    bq_configure_2s_basic();   // writes CUV/COV/SCD + alarm settings
    // bq_cfg_set_occ(...) either disabled or set later when charger is known
    bq_exit_cfgupdate();
  
    // read "last fault" and send to nucleo
    nucleo_send_ok_status();
  }

  // Main loop
  void main_loop(void) {
    while (1) {
      if (alert_flag) {
        bq_handle_alert_and_clear();
        alert_flag = 0;
      }
  
      if (time_to_poll_status()) {
        uint16_t min_mv, max_mv;
        bq_get_cell_minmax(&min_mv, &max_mv);   // reads 0x14/0x16 for 2S
        uint16_t bat;
        bq_get_battery_status(&bat);            // reads 0x12
  
        pack_state_t st = decide_ok_or_warn(min_mv, max_mv, bat);
  
        nucleo_send_status(st, min_mv, max_mv, /*chg*/1, /*dsg*/1);
      }
    }
  }

  // Alert interrupt service routine
  void bq_alert_isr(void) {
    alert_flag = 1;  // nothing else
  }
  
  void handle_alert(void) {
    uint16_t alarm;
    bq_get_alarm_status(&alarm);   // 0x62
  
    uint8_t sa, sb, sta, stb;
    bq_get_safety_alert_a(&sa);    // 0x02
    bq_get_safety_alert_b(&sb);    // 0x04
    bq_get_safety_status_a(&sta);  // 0x03
    bq_get_safety_status_b(&stb);  // 0x05
  
    fault_t f = decode_fault(sa, sb, sta, stb);
  
    // Tell Nucleo to stop *if it still has power*
    if (f == FAULT_CUV || f == FAULT_COV || f == FAULT_OCC) {
      nucleo_send_safe_req(f);
    }
  
    // Store last fault for next boot
    last_fault = f;
  
    // Clear alarm latch
    bq_clear_alarm_status(alarm);  // write 1s to 0x62
  }
  
  