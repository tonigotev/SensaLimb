// Nucleo UART RX interface for BMS frames (declarations only).
#ifndef NUCLEO_UART_RX_H
#define NUCLEO_UART_RX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Forward declaration to avoid HAL include in header.
typedef struct __UART_HandleTypeDef UART_HandleTypeDef;

// UART link defaults
#define NUCLEO_UART_BAUD_DEFAULT        115200u
#define NUCLEO_UART_PREAMBLE            0xAAu
#define NUCLEO_UART_CRC8_POLY           0x07u
#define NUCLEO_UART_CRC8_INIT           0x00u

// Maximum payload from BMS (payload bytes only, excludes frame ID).
#define NUCLEO_UART_MAX_PAYLOAD         16u

// Frame size bounds (on-wire).
// len field counts: seq + id + payload.
#define NUCLEO_UART_MIN_FRAME_LEN       5u  // preamble + len + seq + id + crc
#define NUCLEO_UART_MAX_FRAME_LEN       (NUCLEO_UART_MAX_PAYLOAD + 5u)

// Frame IDs (BMS -> Nucleo)
#define NUCLEO_FRAME_SAFE_REQ           0x10u
#define NUCLEO_FRAME_SCD_EVENT          0x11u
#define NUCLEO_FRAME_LOW_BATT_MODE      0x12u
#define NUCLEO_FRAME_LOW_BATT_LOCK      0x13u
#define NUCLEO_FRAME_CUR_LATCHED        0x14u
#define NUCLEO_FRAME_LOW_BATT_WARN      0x15u
#define NUCLEO_FRAME_LAST_FAULT         0x16u
// Optional Nucleo -> BMS ACK (reserved)
#define NUCLEO_FRAME_ACK                0x80u

// Payload lengths (bytes, excluding frame ID).
// Multi-byte fields are little-endian (LSB first).
#define NUCLEO_PLEN_SAFE_REQ            2u
#define NUCLEO_PLEN_SCD_EVENT           0u
#define NUCLEO_PLEN_LOW_BATT_MODE       0u
#define NUCLEO_PLEN_LOW_BATT_LOCK       0u
#define NUCLEO_PLEN_CUR_LATCHED         0u
#define NUCLEO_PLEN_LOW_BATT_WARN       2u
#define NUCLEO_PLEN_LAST_FAULT          6u
#define NUCLEO_PLEN_ACK                 2u  // {seq_acked, status}

// ACK status values (optional)
#define NUCLEO_ACK_STATUS_OK            0x00u
#define NUCLEO_ACK_STATUS_ERR           0x01u

// Fault codes (from BMS)
#define NUCLEO_FAULT_SCD                1u
#define NUCLEO_FAULT_OCD                2u
#define NUCLEO_FAULT_CUV                3u
#define NUCLEO_FAULT_COV                4u
#define NUCLEO_FAULT_OCC                5u
#define NUCLEO_FAULT_OTINT              6u
#define NUCLEO_FAULT_VREF_VSS           7u
#define NUCLEO_FAULT_HWD                8u

typedef struct {
    uint8_t seq;
    uint8_t id;
    uint8_t payload[NUCLEO_UART_MAX_PAYLOAD];
    uint8_t payload_len;
} nucleo_uart_frame_t;

// Application hooks (implement these in your app).
// fault_code is little-endian (LSB first) on the wire.
void nucleo_on_safe_req(uint16_t fault_code);
void nucleo_on_scd_event(void);
void nucleo_on_low_batt_mode(void);
void nucleo_on_low_batt_lock(void);
void nucleo_on_cur_latched(void);
void nucleo_on_low_batt_warn(uint8_t c1_pct, uint8_t c2_pct);
// reason/count are little-endian (LSB first) on the wire.
void nucleo_on_last_fault(uint16_t reason, uint32_t count);

// Initialization / configuration
void nucleo_uart_rx_init(UART_HandleTypeDef *huart,
                         uint8_t *rx_ring,
                         size_t rx_ring_len);
void nucleo_uart_rx_reset_parser(void);

// RX feeding (choose one based on IRQ/DMA strategy)
void nucleo_uart_rx_push_byte(uint8_t byte);                 // IRQ byte-by-byte
void nucleo_uart_rx_push_block(const uint8_t *data, size_t len); // DMA/IDLE block

// Parser / dispatch
void nucleo_uart_rx_process(void);                           // poll in main loop
bool nucleo_uart_rx_pop_frame(nucleo_uart_frame_t *out);      // optional manual dispatch

// Utility
uint8_t nucleo_uart_crc8(const uint8_t *data, uint8_t len);

#endif
