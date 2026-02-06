
#ifndef BQ_H
#define BQ_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    BMS_OK = 0,
    BMS_ERR_I2C,
    BMS_ERR_TIMEOUT,
    BMS_ERR_CRC,
    BMS_ERR_BAD_PARAM,
} bms_status_t;

typedef struct {
    void *i2c_handle;
    uint8_t i2c_addr_7b;
    bool crc_enabled;
    uint32_t xfer_timeout_ms;
    void *nucleo_uart;
    uint8_t nucleo_max_payload;
    uint32_t nucleo_timeout_ms;
} bq_ctx_t;

typedef struct {
    uint16_t alarm;
    uint8_t sa;
    uint8_t sb;
    uint8_t sta;
    uint8_t stb;
} bq_alert_snapshot_t;

#define BQ_I2C_ADDR_7B          0x08

#define BQ_SUBCMD_LO            0x3E
#define BQ_SUBCMD_HI            0x3F
#define BQ_BUF_START            0x40
#define BQ_BUF_END              0x5F
#define BQ_BUF_CHECKSUM         0x60
#define BQ_BUF_LENGTH           0x61

#define BQ_SAFETY_ALERT_A       0x02
#define BQ_SAFETY_STATUS_A      0x03
#define BQ_SAFETY_ALERT_B       0x04
#define BQ_SAFETY_STATUS_B      0x05
#define SA_COV                  (1u << 7)
#define SA_CUV                  (1u << 6)
#define SA_SCD                  (1u << 5)
#define SA_OCD1                 (1u << 4)
#define SA_OCD2                 (1u << 3)
#define SA_OCC                  (1u << 2)
#define SS_COV                  (1u << 7)
#define SS_CUV                  (1u << 6)
#define SS_SCD                  (1u << 5)
#define SS_OCD1                 (1u << 4)
#define SS_OCD2                 (1u << 3)
#define SS_OCC                  (1u << 2)
#define SS_CURLATCH             (1u << 1)
#define SS_REGOUT               (1u << 0)
#define SB_OTD                  (1u << 7)
#define SB_OTC                  (1u << 6)
#define SB_UTD                  (1u << 5)
#define SB_UTC                  (1u << 4)
#define SB_OTINT                (1u << 3)
#define SB_HWD                  (1u << 2)
#define SB_VREF                 (1u << 1)
#define SB_VSS                  (1u << 0)
#define SSB_OTD                 (1u << 7)
#define SSB_OTC                 (1u << 6)
#define SSB_UTD                 (1u << 5)
#define SSB_UTC                 (1u << 4)
#define SSB_OTINT               (1u << 3)
#define SSB_HWD                 (1u << 2)
#define SSB_VREF                (1u << 1)
#define SSB_VSS                 (1u << 0)
#define BQ_ALARM_SSA            (1u << 15)
#define BQ_ALARM_SSB            (1u << 14)
#define BQ_ALARM_SAA            (1u << 13)
#define BQ_ALARM_SAB            (1u << 12)
#define BQ_ALARM_XCHG           (1u << 11)
#define BQ_ALARM_XDSG           (1u << 10)
#define BQ_ALARM_SHUTV          (1u << 9)
#define BQ_ALARM_CB             (1u << 8)
#define BQ_ALARM_POR            (1u << 0)

#define BQ_FAULT_NONE           0
#define BQ_FAULT_SCD            1
#define BQ_FAULT_OCD            2
#define BQ_FAULT_CUV            3
#define BQ_FAULT_COV            4
#define BQ_FAULT_OCC            5
#define BQ_FAULT_OTINT          6
#define BQ_FAULT_VREF_VSS       7
#define BQ_FAULT_HWD            8

#define BQ_BATTERY_STATUS       0x12
#define BQ_CELL1_VOLT_mV        0x14
#define BQ_CELL2_VOLT_mV        0x16
#define BQ_STACK_VOLT_mV        0x26
#define BQ_CURR_USERA           0x3A

#define BQ_ALARM_STATUS         0x62
#define BQ_ALARM_RAW_STATUS     0x64
#define BQ_ALARM_ENABLE         0x66

#define BQ_SUBCMD_SET_CFGUPDATE 0x0090
#define BQ_SUBCMD_EXIT_CFGUPDATE 0x0092
#define BQ_SUBCMD_SHUTDOWN      0x0010
#define BQ_SUBCMD_DEEPSLEEP     0x000F
#define BQ_SUBCMD_EXIT_DEEPSLEEP 0x000E
#define BQ_SUBCMD_FET_ENABLE    0x0022


#define DM_I2C_ADDRESS          0x9016
#define DM_DEFAULT_ALARM_MASK   0x901C
#define DM_FET_OPTIONS          0x901E
#define DM_ENABLED_PROT_A       0x9024
#define DM_ENABLED_PROT_B       0x9025
#define DM_DSG_FET_PROT_A       0x9026
#define DM_CHG_FET_PROT_A       0x9027
#define DM_BOTH_FET_PROT_B      0x9028
#define DM_CUV_THRESH_mV        0x902E
#define DM_CUV_DELAY            0x9030
#define DM_CUV_RECOV_HYST       0x9031
#define DM_COV_THRESH_mV        0x9032
#define DM_COV_DELAY            0x9034
#define DM_COV_RECOV_HYST       0x9035
#define DM_OCC_THRESH_2mV       0x9036
#define DM_OCC_DELAY            0x9037
#define DM_SCD_THRESH           0x903C
#define DM_SCD_DELAY            0x903D

#define BQ_CFG_SKIP_U1          0xFFu
#define BQ_CFG_SKIP_U2          0xFFFFu

#define BQ_CFG_DEFAULT_ALARM_MASK     0xFC01u
#define BQ_CFG_DEFAULT_FET_OPTIONS    BQ_CFG_SKIP_U2
#define BQ_CFG_ENABLED_PROT_A_VAL     0xF9u
#define BQ_CFG_ENABLED_PROT_B_VAL     0x02u
#define BQ_CFG_DSG_FET_PROT_A_VAL     0xF1u
#define BQ_CFG_CHG_FET_PROT_A_VAL     0xC1u
#define BQ_CFG_BOTH_FET_PROT_B_VAL    0x07u

#define BQ_ADSCAN_PERIOD_MS          10u

#define BQ_CFG_CUV_THRESH_mV          3200u
#define BQ_CFG_CUV_DELAY_TARGET_MS    1000u
#define BQ_CFG_CUV_RECOV_HYST_VAL     100u

#define BQ_CFG_COV_THRESH_mV          4200u
#define BQ_CFG_COV_DELAY_TARGET_MS    300u
#define BQ_CFG_COV_RECOV_HYST_VAL     50u

#define BQ_CFG_OCC_THRESH_2mV_VAL     BQ_CFG_SKIP_U1
#define BQ_CFG_OCC_DELAY_VAL          BQ_CFG_SKIP_U1
#define BQ_CFG_SCD_THRESH_VAL         4u
#define BQ_SCD_DELAY_FASTEST_CODE     0u  // TODO confirm from TRM table
#define BQ_CFG_SCD_DELAY_VAL          BQ_SCD_DELAY_FASTEST_CODE

#define BQ_LOW_BATT_WARN_PCT          15u
#define BQ_LOW_BATT_CLEAR_PCT         20u

bms_status_t bq_init(bq_ctx_t *ctx, void *i2c_handle, uint8_t addr_7b, bool crc_enabled, uint32_t timeout_ms);
bms_status_t bq_read_u8(uint8_t reg, uint8_t *out);
bms_status_t bq_read_u16(uint8_t reg, uint16_t *out_le);     // little-endian on wire
bms_status_t bq_write_u8(uint8_t reg, uint8_t v);
bms_status_t bq_write_u16(uint8_t reg, uint16_t v_le);

bms_status_t bq_subcmd_exec(uint16_t subcmd);
bms_status_t bq_subcmd_read(uint16_t subcmd, uint8_t *out, uint8_t len);
bms_status_t bq_buf_write_and_commit(uint16_t subcmd, const uint8_t *data, uint8_t len);

bms_status_t bq_dm_read(uint16_t dm_addr, uint8_t *buf, uint8_t len);
bms_status_t bq_dm_write(uint16_t dm_addr, const uint8_t *buf, uint8_t len);

bms_status_t bq_dm_write_u1(uint16_t dm_addr, uint8_t v); // dm_addr like 0x9024 etc
bms_status_t bq_dm_write_u2(uint16_t dm_addr, uint16_t v);

bms_status_t bq_enter_cfgupdate(void);    // send subcmd 0x0090 (SET_CFGUPDATE)
bms_status_t bq_exit_cfgupdate(void);     // send subcmd 0x0092 (EXIT_CFGUPDATE)

bms_status_t bq_configure_2s_basic(void);


bms_status_t bq_shutdown(void);       // subcmd 0x0010 (must send twice within 4s)
bms_status_t bq_deepsleep(void);      // subcmd 0x000F (must send twice within 4s)
bms_status_t bq_exit_deepsleep(void); // subcmd 0x000E
bms_status_t bq_fet_enable(void);

void bq_alert_irq_flag_set(void); // ISR-safe: sets volatile flag only

bms_status_t bq_handle_alert_and_clear(bq_alert_snapshot_t *out);
bms_status_t bq_get_alarm_status(uint16_t *alarm);
bms_status_t bq_get_safety_alert_a(uint8_t *alert);
bms_status_t bq_get_safety_alert_b(uint8_t *alert);
bms_status_t bq_get_safety_status_a(uint8_t *status);
bms_status_t bq_get_safety_status_b(uint8_t *status);
bms_status_t bq_clear_alarm_status(uint16_t alarm);
bms_status_t bq_get_cell_minmax(uint16_t *min_mv, uint16_t *max_mv);
bms_status_t bq_get_battery_status(uint16_t *status);


int nucleo_send_ok_status(void);
int nucleo_send_warning(uint16_t code);
int nucleo_send_fault(uint16_t fault_code);
int bms_nucleo_send_status(void);
int bms_nucleo_send_event(uint16_t event_code);
int nucleo_send_low_batt_mode(void);
int nucleo_send_safe_req(uint16_t code, uint32_t timeout_ms);
int nucleo_send_scd_event(uint32_t timeout_ms);
int nucleo_send_low_batt_lockdown(uint32_t timeout_ms);
int nucleo_send_cur_latched(uint32_t timeout_ms);
int nucleo_send_low_batt_warn(uint8_t c1_pct, uint8_t c2_pct);
int nucleo_send_last_fault(uint16_t reason, uint32_t count);

bms_status_t bq_read_safety_alert_a(uint8_t *out);
bms_status_t bq_read_safety_status_a(uint8_t *out);
bms_status_t bq_read_safety_alert_b(uint8_t *out);
bms_status_t bq_read_safety_status_b(uint8_t *out);
bms_status_t bq_read_battery_status_u16(uint16_t *out_le);
bms_status_t bq_read_cell1_mv(uint16_t *out_le);
bms_status_t bq_read_cell2_mv(uint16_t *out_le);
bms_status_t bq_read_stack_mv(uint16_t *out_le);
bms_status_t bq_read_current_ua(uint16_t *out_le);
bms_status_t bq_read_alarm_status_u16(uint16_t *out_le);
bms_status_t bq_write_alarm_status_clear(uint16_t mask);
bms_status_t bq_read_alarm_enable_u16(uint16_t *out_le);
bms_status_t bq_write_alarm_enable_u16(uint16_t mask);
bms_status_t bq_get_cell_percents(uint8_t *c1_pct, uint8_t *c2_pct);

bms_status_t bms_init(void);
void bms_alert_isr_hook(void);
void bms_process(void);

#endif
