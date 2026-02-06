Nucleo UART Integration Notes (BMS STM32C0 ↔ Nucleo)
====================================================

Goal
----
Implement a small, robust RX path on the Nucleo that consumes the BMS UART frames, validates them, and executes the required actions immediately (e.g., SAFE_REQ).

Link & Electrical
-----------------
- UART, 3.3 V logic.
- Wiring: BMS TX → Nucleo RX, BMS RX → Nucleo TX (optional for future ACKs), common GND.
- No flow control.
- Suggested baud: 115200 8N1 (match whatever you configure on the BMS).

Framing (BMS → Nucleo)
----------------------
- Bytes: `[0]=0xAA | [1]=len | [2]=seq | [3..(len+1)]=payload | [last]=crc8`
- `len` counts `seq + payload` bytes (does NOT include crc).
- CRC8: poly 0x07, init 0x00, over bytes `[1 .. 1+len]` (len + seq + payload), MSB-first.
- Drop frame if:
  - Wrong preamble
  - Len exceeds buffer
  - CRC mismatch
- `seq` wraps 0–255. You may ignore duplicates or suppress by `seq` if needed.

Current Frame IDs and Payloads
------------------------------
- 0x10 SAFE_REQ        : payload 2 bytes `fault_code` (LSB first)
- 0x11 SCD_EVENT       : no payload
- 0x12 LOW_BATT_MODE   : no payload
- 0x13 LOW_BATT_LOCK   : no payload
- 0x14 CUR_LATCHED     : no payload
- 0x15 LOW_BATT_WARN   : payload 2 bytes `cell1_pct`, `cell2_pct` (0–100)
- 0x16 LAST_FAULT      : payload 6 bytes: `reason` (u16, LSB first), `count` (u32, LSB first)
- (IDs 0x10/0x20/0x21 for OK/WARNING/FAULT are reserved but unused in current BMS code.)

Fault Codes (from BMS side)
---------------------------
- 1: SCD
- 2: OCD (OCD1/2, OTINT/VREF/VSS, REGOUT grouped)
- 3: CUV
- 4: COV
- 5: OCC
- 6: OTINT (if used distinctly)
- 7: VREF_VSS
- 8: HWD

Expected Nucleo Actions
-----------------------
- SAFE_REQ (0x10): Immediately stop/park motors.
- SCD_EVENT (0x11): Stop motors; log event.
- LOW_BATT_MODE (0x12): Enter reduced-power/limited mode per your app.
- LOW_BATT_LOCK (0x13): Treat as critical low-battery; halt as needed.
- CUR_LATCHED (0x14): Treat as latched current fault; require controlled recovery.
- LOW_BATT_WARN (0x15): Warn UI/log; optionally slow down if either cell <15%.
- LAST_FAULT (0x16): Log/display last fault `reason` and running `count` on boot.

RX Implementation Sketch (Nucleo)
----------------------------------
- Configure UART RX with DMA or IRQ into a ring buffer.
- Parser loop:
  1) Search for `0xAA`.
  2) Ensure at least 2 more bytes (len + seq) available; read `len`.
  3) Require total bytes available: `len + 2` (seq..payload) + 1 crc.
  4) Compute CRC8 over `[len, seq, payload...]`; compare to crc byte.
  5) On success, dispatch by `payload[0]` (the first byte of the payload is the frame ID; seq precedes payload).
  6) Optional: drop duplicates by `seq` if desired; otherwise ignore.
- Keep RX non-blocking; no delays in IRQ/DMA callbacks.

TX / ACKs
---------
- The BMS does not require or parse ACKs today. If you add ACKs:
  - Mirror the framing above; include `seq` from the received frame in your ACK payload.
  - Decide which frames need ACK (e.g., SAFE_REQ, LOW_BATT_LOCK).
  - Keep ACK short and non-blocking on Nucleo TX.

Timeouts / Priorities
---------------------
- Treat SAFE_REQ and SCD_EVENT as critical: act immediately on receipt.
- LOW_BATT_WARN/STATUS-type frames are non-critical; they can be dropped if parsing fails.

Startup Behavior
----------------
- On every BMS init, a LAST_FAULT frame is sent once; consume it to show/log the most recent BMS fault.

Minimal Bring-up Checklist (Nucleo)
-----------------------------------
- Configure UART at agreed baud, 8N1, no flow control.
- Enable RX via DMA/IRQ into a ring.
- Implement the parser with CRC8 (poly 0x07, init 0x00) and the framing above.
- Wire TX/RX/GND to the BMS UART, common 3.3 V domain.
- On SAFE_REQ/SCD_EVENT, stop motors immediately.

Notes
-----
- All multi-byte fields are little-endian (LSB first).
- If you see unexpected frames, verify CRC calculation and len handling first.
