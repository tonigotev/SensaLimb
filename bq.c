// UART is still TX-only; no ACK RX yet.
//SCD/OCC thresholds/delays still need final tuning once hardware data is known.
#include "bq.h"
#include "stm32c0xx_hal.h"

static bq_ctx_t g_bq = {0};
static bool g_init = false;
static volatile bool g_alert_flag = false;
static bool g_bms_watchdog_warn = false;
static uint8_t g_alert_i2c_fail_count = 0;
static bool g_safe_req_backoff = false;
static uint16_t g_last_fault_reason = BQ_FAULT_NONE;
static uint32_t g_fault_count = 0;
static bool g_low_batt_warn_armed = true;
static uint32_t g_last_pct_poll_ms = 0;
static bool g_last_fault_pending = false;
__attribute__((weak)) UART_HandleTypeDef huart1;
static uint8_t g_nuc_seq = 0;
static int bq_hal_nucleo_send(const uint8_t *data, uint8_t len, uint32_t timeout_ms);
static const uint8_t BQ_NUC_FRAME_SAFE_REQ      = 0x10;
static const uint8_t BQ_NUC_FRAME_SCD_EVENT     = 0x11;
static const uint8_t BQ_NUC_FRAME_LOW_BATT_MODE = 0x12;
static const uint8_t BQ_NUC_FRAME_LOW_BATT_LOCK = 0x13;
static const uint8_t BQ_NUC_FRAME_CUR_LATCHED   = 0x14;
static const uint8_t BQ_NUC_FRAME_LOW_BATT_WARN = 0x15;
static const uint8_t BQ_NUC_FRAME_LAST_FAULT    = 0x16;

int nucleo_send_low_batt_mode(void) {
    return bq_send_frame(BQ_NUC_FRAME_LOW_BATT_MODE, NULL, 0, 50);
}
int nucleo_send_safe_req(uint16_t code, uint32_t timeout_ms) {
    uint8_t p[2] = { (uint8_t)(code & 0xFF), (uint8_t)(code >> 8) };
    return bq_send_frame(BQ_NUC_FRAME_SAFE_REQ, p, 2, timeout_ms);
}
int nucleo_send_scd_event(uint32_t timeout_ms) {
    return bq_send_frame(BQ_NUC_FRAME_SCD_EVENT, NULL, 0, timeout_ms);
}
int nucleo_send_low_batt_lockdown(uint32_t timeout_ms) {
    return bq_send_frame(BQ_NUC_FRAME_LOW_BATT_LOCK, NULL, 0, timeout_ms);
}
int nucleo_send_cur_latched(uint32_t timeout_ms) {
    return bq_send_frame(BQ_NUC_FRAME_CUR_LATCHED, NULL, 0, timeout_ms);
}
int nucleo_send_low_batt_warn(uint8_t c1_pct, uint8_t c2_pct) {
    uint8_t p[2] = { c1_pct, c2_pct };
    return bq_send_frame(BQ_NUC_FRAME_LOW_BATT_WARN, p, 2, 50);
}
int nucleo_send_last_fault(uint16_t reason, uint32_t count) {
    uint8_t p[6] = {
        (uint8_t)(reason & 0xFF), (uint8_t)(reason >> 8),
        (uint8_t)(count & 0xFF), (uint8_t)((count >> 8) & 0xFF),
        (uint8_t)((count >> 16) & 0xFF), (uint8_t)((count >> 24) & 0xFF)
    };
    return bq_send_frame(BQ_NUC_FRAME_LAST_FAULT, p, 6, 50);
}

static void bq_set_fault(uint16_t code) {
    g_last_fault_reason = code;
    g_fault_count++;
}

static int bq_send_frame(uint8_t id, const uint8_t *payload, uint8_t plen, uint32_t timeout_ms) {
    uint8_t buf[8];
    if (1 + plen > sizeof(buf)) plen = sizeof(buf) - 1;
    buf[0] = id;
    for (uint8_t i = 0; i < plen; i++) buf[1 + i] = payload[i];
    return bq_hal_nucleo_send(buf, (uint8_t)(1 + plen), timeout_ms);
}

static void bq_handle_alert_actions(uint16_t alarm, const bq_alert_snapshot_t *snap) {
    if (!snap) return;

    uint32_t safe_req_timeout = (alarm & BQ_ALARM_XDSG) ? 20 : 200;

    uint8_t sta = snap->sta;
    uint8_t stb = snap->stb;

    if (sta || stb) {
        if (sta & SS_SCD) {
            bq_set_fault(BQ_FAULT_SCD);
            nucleo_send_scd_event(50);
            return;
        }
        if ((sta & (SS_OCD1 | SS_OCD2)) ||
            (stb & (SSB_OTINT | SSB_VREF | SSB_VSS)) ||
            (stb & (SSB_OTD | SSB_OTC | SSB_UTD | SSB_UTC)) ||
            (sta & SS_REGOUT)) {
            bq_set_fault(BQ_FAULT_OCD);
            nucleo_send_safe_req(BQ_FAULT_OCD, safe_req_timeout);
            return;
        }
        if (sta & SS_CUV) {
            bq_set_fault(BQ_FAULT_CUV);
            nucleo_send_low_batt_lockdown(safe_req_timeout);
            return;
        }
        if (sta & SS_CURLATCH) {
            bq_set_fault(BQ_FAULT_OCC); 
            nucleo_send_cur_latched(50);
            return;
        }
        if (sta & SS_OCC) {
            bq_set_fault(BQ_FAULT_OCC);
            return;
        }
        if (sta & SS_COV) {
            bq_set_fault(BQ_FAULT_COV);
            return;
        }
        if (stb & SSB_HWD) {
            bq_set_fault(BQ_FAULT_HWD);
            g_bms_watchdog_warn = true;
            return;
        }
    } else {
        if (snap->sa & SA_SCD) {
            bq_set_fault(BQ_FAULT_SCD);
            nucleo_send_scd_event(50);
            return;
        }
        if ((snap->sa & (SA_OCD1 | SA_OCD2)) ||
            (snap->sb & (SB_OTINT | SB_VREF | SB_VSS)) ||
            (snap->sb & (SB_OTD | SB_OTC | SB_UTD | SB_UTC))) {
            bq_set_fault(BQ_FAULT_OCD);
            nucleo_send_safe_req(BQ_FAULT_OCD, safe_req_timeout);
            return;
        }
        if (snap->sa & SA_CUV) {
            bq_set_fault(BQ_FAULT_CUV);
            nucleo_send_low_batt_lockdown(safe_req_timeout);
            return;
        }
        if (snap->sa & SA_OCC) {
            bq_set_fault(BQ_FAULT_OCC);
            return;
        }
        if (snap->sa & SA_COV) {
            bq_set_fault(BQ_FAULT_COV);
            return;
        }
        if (snap->sb & SB_HWD) {
            bq_set_fault(BQ_FAULT_HWD);
            g_bms_watchdog_warn = true;
            return;
        }
    }
}

static inline bool bq_ready(void) {
    return g_init;
}

static inline bms_status_t bq_guard(void) {
    return bq_ready() ? BMS_OK : BMS_ERR_BAD_PARAM;
}

static uint8_t bq_crc8(const uint8_t *data, uint8_t len) {
    uint8_t crc = 0x00; // init
    for (uint8_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            if (crc & 0x80) crc = (uint8_t)((crc << 1) ^ 0x07);
            else crc <<= 1;
        }
    }
    return crc;
}

static uint8_t bq_checksum(const uint8_t *data, uint16_t len) {
    uint32_t sum = 0;
    for (uint16_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return (uint8_t)(~sum);
}

static uint8_t bq_delay_ms_to_adscan_counts(uint32_t target_ms) {
    uint32_t ad = BQ_ADSCAN_PERIOD_MS;
    uint32_t c = (target_ms + ad / 2) / ad;
    if (c < 1) c = 1;
    if (c > 255) c = 255;
    return (uint8_t)c;
}

static uint8_t bq_pct_from_mv(uint16_t mv) {
    int32_t delta = (int32_t)mv - 3200;
    if (delta <= 0) return 0;
    if (delta >= 1000) return 100;
    return (uint8_t)(delta / 10); // (mv-3200)/1000 * 100
}

static int bq_hal_nucleo_send(const uint8_t *data, uint8_t len, uint32_t timeout_ms) {
    if (!data || len == 0) return -1;
    if (!g_bq.nucleo_uart) return -1;

    uint8_t max_payload = g_bq.nucleo_max_payload ? g_bq.nucleo_max_payload : 16;
    if (len > max_payload) return -1;

    const uint8_t critical_id1 = BQ_NUC_FRAME_SAFE_REQ;
    const uint8_t critical_id2 = BQ_NUC_FRAME_SCD_EVENT;
    const uint8_t critical_id3 = BQ_NUC_FRAME_LOW_BATT_LOCK;
    bool critical = (data[0] == critical_id1) || (data[0] == critical_id2) || (data[0] == critical_id3);

    UART_HandleTypeDef *huart = (UART_HandleTypeDef *)g_bq.nucleo_uart;
    if (huart->gState != HAL_UART_STATE_READY && !critical) {
        return -1;
    }

    uint8_t frame[32];
    uint8_t seq = g_nuc_seq++;
    uint8_t frame_len = (uint8_t)(1 + len); // seq + payload
    if ((uint16_t)(frame_len + 3) > sizeof(frame)) return -1; // preamble + len + crc

    frame[0] = 0xAA;
    frame[1] = frame_len;
    frame[2] = seq;
    for (uint8_t i = 0; i < len; i++) frame[3 + i] = data[i];
    uint8_t crc = bq_crc8(&frame[1], (uint8_t)(frame_len + 1)); // len + seq+payload
    frame[3 + len] = crc;

    uint32_t to = timeout_ms ? timeout_ms : g_bq.nucleo_timeout_ms;
    HAL_StatusTypeDef st = HAL_UART_Transmit(huart, frame, (uint16_t)(frame_len + 3), to);
    return (st == HAL_OK) ? 0 : -1;
}

static int bq_hal_i2c_mem_read(uint8_t reg, uint8_t *buf, uint16_t len) {
    HAL_StatusTypeDef st = HAL_I2C_Mem_Read(
        (I2C_HandleTypeDef *)g_bq.i2c_handle,
        (uint16_t)(g_bq.i2c_addr_7b << 1),
        reg,
        I2C_MEMADD_SIZE_8BIT,
        buf,
        len,
        g_bq.xfer_timeout_ms);
    return (st == HAL_OK) ? 0 : -1;
}

static int bq_hal_i2c_mem_write(uint8_t reg, const uint8_t *buf, uint16_t len) {
    HAL_StatusTypeDef st = HAL_I2C_Mem_Write(
        (I2C_HandleTypeDef *)g_bq.i2c_handle,
        (uint16_t)(g_bq.i2c_addr_7b << 1),
        reg,
        I2C_MEMADD_SIZE_8BIT,
        (uint8_t *)buf,
        len,
        g_bq.xfer_timeout_ms);
    return (st == HAL_OK) ? 0 : -1;
}

bms_status_t bq_init(bq_ctx_t *ctx, void *i2c_handle, uint8_t addr_7b, bool crc_enabled, uint32_t timeout_ms) {
    if (!ctx || !i2c_handle) {
        return BMS_ERR_BAD_PARAM;
    }

    // Default address if caller passes 0
    if (addr_7b == 0) {
        addr_7b = BQ_I2C_ADDR_7B;
    }

    if (timeout_ms == 0) {
        timeout_ms = 50; // sane minimum default
    }

    ctx->i2c_handle = i2c_handle;
    ctx->i2c_addr_7b = addr_7b;
    ctx->crc_enabled = crc_enabled;
    ctx->xfer_timeout_ms = timeout_ms;
    if (ctx->nucleo_uart == NULL) {
        ctx->nucleo_uart = &huart1;
    }
    if (ctx->nucleo_max_payload == 0) {
        ctx->nucleo_max_payload = 16;
    }
    if (ctx->nucleo_timeout_ms == 0) {
        ctx->nucleo_timeout_ms = 50;
    }

    g_bq = *ctx;
    g_init = true;
    return BMS_OK;
}

bms_status_t bq_read_u8(uint8_t reg, uint8_t *out) {
    if (!out) return BMS_ERR_BAD_PARAM;
    if (bq_guard() != BMS_OK) return BMS_ERR_BAD_PARAM;
    return (bq_hal_i2c_mem_read(reg, out, 1) == 0) ? BMS_OK : BMS_ERR_I2C;
}

bms_status_t bq_read_u16(uint8_t reg, uint16_t *out_le) {
    if (!out_le) return BMS_ERR_BAD_PARAM;
    if (bq_guard() != BMS_OK) return BMS_ERR_BAD_PARAM;
    uint8_t buf[2];
    if (bq_hal_i2c_mem_read(reg, buf, 2) != 0) return BMS_ERR_I2C;
    *out_le = (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
    return BMS_OK;
}

bms_status_t bq_write_u8(uint8_t reg, uint8_t v) {
    if (bq_guard() != BMS_OK) return BMS_ERR_BAD_PARAM;
    return (bq_hal_i2c_mem_write(reg, &v, 1) == 0) ? BMS_OK : BMS_ERR_I2C;
}

bms_status_t bq_write_u16(uint8_t reg, uint16_t v_le) {
    if (bq_guard() != BMS_OK) return BMS_ERR_BAD_PARAM;
    uint8_t buf[2] = { (uint8_t)(v_le & 0xFF), (uint8_t)(v_le >> 8) };
    return (bq_hal_i2c_mem_write(reg, buf, 2) == 0) ? BMS_OK : BMS_ERR_I2C;
}

bms_status_t bq_subcmd_exec(uint16_t subcmd) {
    if (bq_guard() != BMS_OK) return BMS_ERR_BAD_PARAM;
    uint8_t cmd_le[2] = { (uint8_t)(subcmd & 0xFF), (uint8_t)(subcmd >> 8) };
    return (bq_hal_i2c_mem_write(BQ_SUBCMD_LO, cmd_le, sizeof(cmd_le)) == 0) ? BMS_OK : BMS_ERR_I2C;
}

bms_status_t bq_subcmd_read(uint16_t subcmd, uint8_t *out, uint8_t len) {
    if (!out || len == 0) return BMS_ERR_BAD_PARAM;
    bms_status_t st = bq_subcmd_exec(subcmd);
    if (st != BMS_OK) return st;
    return (bq_hal_i2c_mem_read(BQ_BUF_START, out, len) == 0) ? BMS_OK : BMS_ERR_I2C;
}

bms_status_t bq_buf_write_and_commit(uint16_t subcmd, const uint8_t *data, uint8_t len) {
    if ((len > (BQ_BUF_END - BQ_BUF_START + 1)) || (len == 0 && data != NULL)) {
        return BMS_ERR_BAD_PARAM;
    }
    bms_status_t st = bq_subcmd_exec(subcmd);
    if (st != BMS_OK) return st;

    if (len && data) {
        if (bq_hal_i2c_mem_write(BQ_BUF_START, data, len) != 0) return BMS_ERR_I2C;
    }

    uint8_t chk_data[2 + 64]; // 2 for subcmd + max payload (fits buffer)
    chk_data[0] = (uint8_t)(subcmd & 0xFF);
    chk_data[1] = (uint8_t)(subcmd >> 8);
    for (uint8_t i = 0; i < len; i++) chk_data[2 + i] = data ? data[i] : 0x00;
    uint8_t checksum = bq_checksum(chk_data, (uint16_t)(2 + len));
    uint8_t length = (uint8_t)(len + 4); // cmd(2) + checksum + length

    if (bq_hal_i2c_mem_write(BQ_BUF_CHECKSUM, &checksum, 1) != 0) return BMS_ERR_I2C;
    if (bq_hal_i2c_mem_write(BQ_BUF_LENGTH, &length, 1) != 0) return BMS_ERR_I2C;
    return BMS_OK;
}

bms_status_t bq_dm_read(uint16_t dm_addr, uint8_t *buf, uint8_t len) {
    return bq_subcmd_read(dm_addr, buf, len);
}

bms_status_t bq_dm_write(uint16_t dm_addr, const uint8_t *buf, uint8_t len) {
    return bq_buf_write_and_commit(dm_addr, buf, len);
}

bms_status_t bq_dm_write_u1(uint16_t dm_addr, uint8_t v) {
    return bq_dm_write(dm_addr, &v, 1);
}

bms_status_t bq_dm_write_u2(uint16_t dm_addr, uint16_t v) {
    uint8_t le[2] = { (uint8_t)(v & 0xFF), (uint8_t)(v >> 8) };
    return bq_dm_write(dm_addr, le, 2);
}

bms_status_t bq_enter_cfgupdate(void) {
    return bq_subcmd_exec(BQ_SUBCMD_SET_CFGUPDATE);
}

bms_status_t bq_exit_cfgupdate(void) {
    return bq_subcmd_exec(BQ_SUBCMD_EXIT_CFGUPDATE);
}

bms_status_t bq_shutdown(void) {
    bms_status_t st = bq_subcmd_exec(BQ_SUBCMD_SHUTDOWN);
    if (st != BMS_OK) return st;
    HAL_Delay(1);
    return bq_subcmd_exec(BQ_SUBCMD_SHUTDOWN);
}

bms_status_t bq_deepsleep(void) {
    bms_status_t st = bq_subcmd_exec(BQ_SUBCMD_DEEPSLEEP);
    if (st != BMS_OK) return st;
    HAL_Delay(1);
    return bq_subcmd_exec(BQ_SUBCMD_DEEPSLEEP);
}

bms_status_t bq_exit_deepsleep(void) {
    return bq_subcmd_exec(BQ_SUBCMD_EXIT_DEEPSLEEP);
}

bms_status_t bq_fet_enable(void) {
    return bq_subcmd_exec(BQ_SUBCMD_FET_ENABLE);
}

bms_status_t bq_read_safety_alert_a(uint8_t *out) { return bq_read_u8(BQ_SAFETY_ALERT_A, out); }
bms_status_t bq_read_safety_status_a(uint8_t *out) { return bq_read_u8(BQ_SAFETY_STATUS_A, out); }
bms_status_t bq_read_safety_alert_b(uint8_t *out) { return bq_read_u8(BQ_SAFETY_ALERT_B, out); }
bms_status_t bq_read_safety_status_b(uint8_t *out) { return bq_read_u8(BQ_SAFETY_STATUS_B, out); }
bms_status_t bq_read_battery_status_u16(uint16_t *out_le) { return bq_read_u16(BQ_BATTERY_STATUS, out_le); }
bms_status_t bq_read_cell1_mv(uint16_t *out_le) { return bq_read_u16(BQ_CELL1_VOLT_mV, out_le); }
bms_status_t bq_read_cell2_mv(uint16_t *out_le) { return bq_read_u16(BQ_CELL2_VOLT_mV, out_le); }
bms_status_t bq_read_stack_mv(uint16_t *out_le) { return bq_read_u16(BQ_STACK_VOLT_mV, out_le); }
bms_status_t bq_read_current_ua(uint16_t *out_le) { return bq_read_u16(BQ_CURR_USERA, out_le); }
bms_status_t bq_read_alarm_status_u16(uint16_t *out_le) { return bq_read_u16(BQ_ALARM_STATUS, out_le); }
bms_status_t bq_write_alarm_status_clear(uint16_t mask) { return bq_write_u16(BQ_ALARM_STATUS, mask); }
bms_status_t bq_read_alarm_enable_u16(uint16_t *out_le) { return bq_read_u16(BQ_ALARM_ENABLE, out_le); }
bms_status_t bq_write_alarm_enable_u16(uint16_t mask) { return bq_write_u16(BQ_ALARM_ENABLE, mask); }

void bq_alert_irq_flag_set(void) {
    g_alert_flag = true;
}

bms_status_t bq_get_alarm_status(uint16_t *alarm) {
    return bq_read_u16(BQ_ALARM_STATUS, alarm);
}

bms_status_t bq_get_safety_alert_a(uint8_t *alert) { return bq_read_safety_alert_a(alert); }
bms_status_t bq_get_safety_alert_b(uint8_t *alert) { return bq_read_safety_alert_b(alert); }
bms_status_t bq_get_safety_status_a(uint8_t *status) { return bq_read_safety_status_a(status); }
bms_status_t bq_get_safety_status_b(uint8_t *status) { return bq_read_safety_status_b(status); }

bms_status_t bq_clear_alarm_status(uint16_t alarm) {
    return bq_write_u16(BQ_ALARM_STATUS, alarm);
}

bms_status_t bq_get_cell_minmax(uint16_t *min_mv, uint16_t *max_mv) {
    if (!min_mv || !max_mv) return BMS_ERR_BAD_PARAM;
    uint16_t c1, c2;
    bms_status_t st = bq_read_cell1_mv(&c1);
    if (st != BMS_OK) return st;
    st = bq_read_cell2_mv(&c2);
    if (st != BMS_OK) return st;
    *min_mv = (c1 < c2) ? c1 : c2;
    *max_mv = (c1 > c2) ? c1 : c2;
    return BMS_OK;
}

bms_status_t bq_get_battery_status(uint16_t *status) {
    return bq_read_battery_status_u16(status);
}

bms_status_t bq_get_cell_percents(uint8_t *c1_pct, uint8_t *c2_pct) {
    if (!c1_pct || !c2_pct) return BMS_ERR_BAD_PARAM;
    uint16_t c1_mv, c2_mv;
    bms_status_t st = bq_read_cell1_mv(&c1_mv);
    if (st != BMS_OK) return st;
    st = bq_read_cell2_mv(&c2_mv);
    if (st != BMS_OK) return st;
    *c1_pct = bq_pct_from_mv(c1_mv);
    *c2_pct = bq_pct_from_mv(c2_mv);
    return BMS_OK;
}

bms_status_t bq_handle_alert_and_clear(bq_alert_snapshot_t *out) {
    if (!g_alert_flag) return BMS_OK;

    uint16_t alarm = 0;
    uint8_t sa = 0, sb = 0, sta = 0, stb = 0;
    bms_status_t st;
    bms_status_t first_err = BMS_OK;

    st = bq_get_alarm_status(&alarm);
    if (st != BMS_OK && first_err == BMS_OK) first_err = st;

    st = bq_get_safety_alert_a(&sa);
    if (st != BMS_OK && first_err == BMS_OK) first_err = st;
    st = bq_get_safety_alert_b(&sb);
    if (st != BMS_OK && first_err == BMS_OK) first_err = st;

    if (alarm & BQ_ALARM_SSA) {
        st = bq_get_safety_status_a(&sta);
        if (st != BMS_OK && first_err == BMS_OK) first_err = st;
    }
    if (alarm & BQ_ALARM_SSB) {
        st = bq_get_safety_status_b(&stb);
        if (st != BMS_OK && first_err == BMS_OK) first_err = st;
    }

    if (out) {
        out->alarm = alarm;
        out->sa = sa;
        out->sb = sb;
        out->sta = sta;
        out->stb = stb;
    }

    if (first_err == BMS_OK) {
        st = bq_clear_alarm_status(alarm);
        if (st == BMS_OK) {
            g_alert_flag = false;
        } else if (first_err == BMS_OK) {
            first_err = st;
        }
    }
    return first_err;
}

bms_status_t bq_configure_2s_basic(void) {
    bms_status_t st;

    // Alarm mask / options
    if (BQ_CFG_DEFAULT_ALARM_MASK != BQ_CFG_SKIP_U2) {
        st = bq_dm_write_u2(DM_DEFAULT_ALARM_MASK, BQ_CFG_DEFAULT_ALARM_MASK);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_DEFAULT_FET_OPTIONS != BQ_CFG_SKIP_U2) {
        st = bq_dm_write_u2(DM_FET_OPTIONS, BQ_CFG_DEFAULT_FET_OPTIONS);
        if (st != BMS_OK) return st;
    }

    // Protection enables / mapping
    if (BQ_CFG_ENABLED_PROT_A_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_ENABLED_PROT_A, BQ_CFG_ENABLED_PROT_A_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_ENABLED_PROT_B_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_ENABLED_PROT_B, BQ_CFG_ENABLED_PROT_B_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_DSG_FET_PROT_A_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_DSG_FET_PROT_A, BQ_CFG_DSG_FET_PROT_A_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_CHG_FET_PROT_A_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_CHG_FET_PROT_A, BQ_CFG_CHG_FET_PROT_A_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_BOTH_FET_PROT_B_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_BOTH_FET_PROT_B, BQ_CFG_BOTH_FET_PROT_B_VAL);
        if (st != BMS_OK) return st;
    }

    // Voltage thresholds/delays (CUV/COV delay fields are in ADSCAN counts)
    if (BQ_CFG_CUV_THRESH_mV != BQ_CFG_SKIP_U2) {
        st = bq_dm_write_u2(DM_CUV_THRESH_mV, BQ_CFG_CUV_THRESH_mV);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_CUV_DELAY_TARGET_MS != BQ_CFG_SKIP_U1 && BQ_CFG_CUV_DELAY_TARGET_MS != 0) {
        uint8_t cnt = bq_delay_ms_to_adscan_counts(BQ_CFG_CUV_DELAY_TARGET_MS);
        st = bq_dm_write_u1(DM_CUV_DELAY, cnt);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_CUV_RECOV_HYST_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_CUV_RECOV_HYST, BQ_CFG_CUV_RECOV_HYST_VAL);
        if (st != BMS_OK) return st;
    }

    if (BQ_CFG_COV_THRESH_mV != BQ_CFG_SKIP_U2) {
        st = bq_dm_write_u2(DM_COV_THRESH_mV, BQ_CFG_COV_THRESH_mV);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_COV_DELAY_TARGET_MS != BQ_CFG_SKIP_U1 && BQ_CFG_COV_DELAY_TARGET_MS != 0) {
        uint8_t cnt = bq_delay_ms_to_adscan_counts(BQ_CFG_COV_DELAY_TARGET_MS);
        st = bq_dm_write_u1(DM_COV_DELAY, cnt);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_COV_RECOV_HYST_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_COV_RECOV_HYST, BQ_CFG_COV_RECOV_HYST_VAL);
        if (st != BMS_OK) return st;
    }

    // Current thresholds/delays
    if (BQ_CFG_OCC_THRESH_2mV_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_OCC_THRESH_2mV, BQ_CFG_OCC_THRESH_2mV_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_OCC_DELAY_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_OCC_DELAY, BQ_CFG_OCC_DELAY_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_SCD_THRESH_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_SCD_THRESH, BQ_CFG_SCD_THRESH_VAL);
        if (st != BMS_OK) return st;
    }
    if (BQ_CFG_SCD_DELAY_VAL != BQ_CFG_SKIP_U1) {
        st = bq_dm_write_u1(DM_SCD_DELAY, BQ_CFG_SCD_DELAY_VAL);
        if (st != BMS_OK) return st;
    }

    return BMS_OK;
}

bms_status_t bms_init(void) {
    if (!g_init) return BMS_ERR_BAD_PARAM;

    bms_status_t st = bq_enter_cfgupdate();
    if (st != BMS_OK) return st;

    st = bq_configure_2s_basic();
    if (st != BMS_OK) {
        (void)bq_exit_cfgupdate();
        return st;
    }

    if (BQ_CFG_DEFAULT_ALARM_MASK != BQ_CFG_SKIP_U2) {
        bms_status_t ae = bq_write_alarm_enable_u16(BQ_CFG_DEFAULT_ALARM_MASK);
        if (ae != BMS_OK) {
            (void)bq_exit_cfgupdate();
            return ae;
        }
    }

    st = bq_exit_cfgupdate();
    if (st != BMS_OK) return st;

    (void)bq_fet_enable();

    uint16_t alarm = 0;
    if (bq_get_alarm_status(&alarm) == BMS_OK && alarm) {
        (void)bq_clear_alarm_status(alarm);
    }

    if (nucleo_send_last_fault(g_last_fault_reason, g_fault_count) != 0) {
        g_last_fault_pending = true;
    } else {
        g_last_fault_pending = false;
    }

    return BMS_OK;
}

void bms_alert_isr_hook(void) {
    bq_alert_irq_flag_set();
}

void bms_process(void) {
    if (g_alert_flag) {
        bq_alert_snapshot_t snap = {0};
        bms_status_t st = bq_handle_alert_and_clear(&snap);
        if (st == BMS_OK) {
            bq_handle_alert_actions(snap.alarm, &snap);
            g_alert_i2c_fail_count = 0;
            g_safe_req_backoff = false;
        } else {
            if (++g_alert_i2c_fail_count >= 3) {
                if (!g_safe_req_backoff) {
                    nucleo_send_safe_req(BQ_FAULT_OCD, 200);
                    g_safe_req_backoff = true;
                }
                g_alert_i2c_fail_count = 0;
            }
        }
    }

    if (g_last_fault_pending && g_bq.nucleo_uart) {
        UART_HandleTypeDef *huart = (UART_HandleTypeDef *)g_bq.nucleo_uart;
        if (huart->gState == HAL_UART_STATE_READY) {
            if (nucleo_send_last_fault(g_last_fault_reason, g_fault_count) == 0) {
                g_last_fault_pending = false;
            }
        }
    }

    uint32_t now = HAL_GetTick();
    if ((now - g_last_pct_poll_ms) >= 1000) {
        g_last_pct_poll_ms = now;
        uint8_t c1_pct = 0, c2_pct = 0;
        if (bq_get_cell_percents(&c1_pct, &c2_pct) == BMS_OK) {
            if ((c1_pct < BQ_LOW_BATT_WARN_PCT) || (c2_pct < BQ_LOW_BATT_WARN_PCT)) {
                if (g_low_batt_warn_armed) {
                    nucleo_send_low_batt_warn(c1_pct, c2_pct);
                    g_low_batt_warn_armed = false;
                }
            } else if ((c1_pct >= BQ_LOW_BATT_CLEAR_PCT) && (c2_pct >= BQ_LOW_BATT_CLEAR_PCT)) {
                g_low_batt_warn_armed = true;
            }
        }
    }
}

// =====================
// Nucleo publishing stubs
// =====================
int bms_nucleo_send_status(void) { return 0; }
int bms_nucleo_send_event(uint16_t event_code) { (void)event_code; return 0; }
