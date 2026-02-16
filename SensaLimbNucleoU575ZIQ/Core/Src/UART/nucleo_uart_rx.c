// Nucleo UART RX implementation for BMS frames.
#include "nucleo_uart_rx.h"

typedef struct {
    UART_HandleTypeDef *huart;
    uint8_t *ring;
    size_t ring_len;
    volatile size_t head;
    volatile size_t tail;
    bool overflow;
    bool initialized;
} nucleo_uart_ctx_t;

static nucleo_uart_ctx_t g_uart = {0};

static size_t rb_next(size_t v) {
    return (v + 1u) % g_uart.ring_len;
}

static size_t rb_count(void) {
    size_t head = g_uart.head;
    size_t tail = g_uart.tail;
    if (head >= tail) return head - tail;
    return g_uart.ring_len - (tail - head);
}

static void rb_clear(void) {
    g_uart.tail = g_uart.head;
}

static bool rb_peek(size_t offset, uint8_t *out) {
    if (!out) return false;
    size_t count = rb_count();
    if (offset >= count) return false;
    size_t idx = (g_uart.tail + offset) % g_uart.ring_len;
    *out = g_uart.ring[idx];
    return true;
}

static void rb_pop(size_t count) {
    if (count == 0 || g_uart.ring_len == 0) return;
    g_uart.tail = (g_uart.tail + count) % g_uart.ring_len;
}

static void rb_push(uint8_t b) {
    if (!g_uart.ring || g_uart.ring_len < 2u) return;
    size_t next = rb_next(g_uart.head);
    if (next == g_uart.tail) {
        g_uart.overflow = true;
        return;
    }
    g_uart.ring[g_uart.head] = b;
    g_uart.head = next;
}

uint8_t nucleo_uart_crc8(const uint8_t *data, uint8_t len) {
    uint8_t crc = NUCLEO_UART_CRC8_INIT;
    if (!data) return crc;
    for (uint8_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            if (crc & 0x80u) crc = (uint8_t)((crc << 1) ^ NUCLEO_UART_CRC8_POLY);
            else crc <<= 1;
        }
    }
    return crc;
}

void nucleo_uart_rx_init(UART_HandleTypeDef *huart,
                         uint8_t *rx_ring,
                         size_t rx_ring_len) {
    g_uart.huart = huart;
    g_uart.ring = rx_ring;
    g_uart.ring_len = rx_ring_len;
    g_uart.head = 0;
    g_uart.tail = 0;
    g_uart.overflow = false;
    g_uart.initialized = (rx_ring != NULL && rx_ring_len >= NUCLEO_UART_MAX_FRAME_LEN);
}

void nucleo_uart_rx_reset_parser(void) {
    g_uart.overflow = false;
    rb_clear();
}

void nucleo_uart_rx_push_byte(uint8_t byte) {
    if (!g_uart.initialized) return;
    rb_push(byte);
}

void nucleo_uart_rx_push_block(const uint8_t *data, size_t len) {
    if (!g_uart.initialized || !data || len == 0) return;
    for (size_t i = 0; i < len; i++) {
        rb_push(data[i]);
    }
}

static bool nucleo_uart_parse_frame(nucleo_uart_frame_t *out) {
    if (!out || !g_uart.initialized) return false;

    if (g_uart.overflow) {
        g_uart.overflow = false;
        rb_clear();
    }

    while (rb_count() >= 1u) {
        uint8_t b0 = 0;
        (void)rb_peek(0u, &b0);
        if (b0 != NUCLEO_UART_PREAMBLE) {
            rb_pop(1u);
            continue;
        }

        if (rb_count() < 2u) return false;

        uint8_t len = 0;
        (void)rb_peek(1u, &len);
        if (len < 1u || len > (uint8_t)(1u + NUCLEO_UART_MAX_PAYLOAD)) {
            rb_pop(1u);
            continue;
        }

        size_t total = (size_t)len + 4u; // preamble + len + seq + body(len) + crc
        if (rb_count() < total) return false;

        uint8_t frame[NUCLEO_UART_MAX_FRAME_LEN];
        for (size_t i = 0; i < total; i++) {
            (void)rb_peek(i, &frame[i]);
        }

        uint8_t crc = nucleo_uart_crc8(&frame[1], (uint8_t)(len + 2u)); // len + seq + body
        uint8_t crc_rx = frame[total - 1u];
        if (crc != crc_rx) {
            rb_pop(1u);
            continue;
        }

        out->seq = frame[2];
        out->id = frame[3];
        uint8_t payload_len = (uint8_t)(len - 1u);
        out->payload_len = payload_len;
        if (payload_len > 0u) {
            memcpy(out->payload, &frame[4], payload_len);
        }

        rb_pop(total);
        return true;
    }

    return false;
}

bool nucleo_uart_rx_pop_frame(nucleo_uart_frame_t *out) {
    return nucleo_uart_parse_frame(out);
}

void nucleo_uart_rx_process(void) {
    nucleo_uart_frame_t frame;
    while (nucleo_uart_parse_frame(&frame)) {
        switch (frame.id) {
            case NUCLEO_FRAME_SAFE_REQ:
                if (frame.payload_len == NUCLEO_PLEN_SAFE_REQ) {
                    uint16_t fault = (uint16_t)frame.payload[0] |
                                     ((uint16_t)frame.payload[1] << 8);
                    nucleo_on_safe_req(fault);
                }
                break;
            case NUCLEO_FRAME_SCD_EVENT:
                if (frame.payload_len == NUCLEO_PLEN_SCD_EVENT) {
                    nucleo_on_scd_event();
                }
                break;
            case NUCLEO_FRAME_LOW_BATT_MODE:
                if (frame.payload_len == NUCLEO_PLEN_LOW_BATT_MODE) {
                    nucleo_on_low_batt_mode();
                }
                break;
            case NUCLEO_FRAME_LOW_BATT_LOCK:
                if (frame.payload_len == NUCLEO_PLEN_LOW_BATT_LOCK) {
                    nucleo_on_low_batt_lock();
                }
                break;
            case NUCLEO_FRAME_CUR_LATCHED:
                if (frame.payload_len == NUCLEO_PLEN_CUR_LATCHED) {
                    nucleo_on_cur_latched();
                }
                break;
            case NUCLEO_FRAME_LOW_BATT_WARN:
                if (frame.payload_len == NUCLEO_PLEN_LOW_BATT_WARN) {
                    nucleo_on_low_batt_warn(frame.payload[0], frame.payload[1]);
                }
                break;
            case NUCLEO_FRAME_LAST_FAULT:
                if (frame.payload_len == NUCLEO_PLEN_LAST_FAULT) {
                    uint16_t reason = (uint16_t)frame.payload[0] |
                                      ((uint16_t)frame.payload[1] << 8);
                    uint32_t count = (uint32_t)frame.payload[2] |
                                     ((uint32_t)frame.payload[3] << 8) |
                                     ((uint32_t)frame.payload[4] << 16) |
                                     ((uint32_t)frame.payload[5] << 24);
                    nucleo_on_last_fault(reason, count);
                }
                break;
            default:
                break;
        }
    }
}
