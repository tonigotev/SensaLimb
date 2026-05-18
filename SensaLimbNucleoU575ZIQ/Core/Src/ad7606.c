#include <stdbool.h>
#include <stdint.h>
#include "ad7606.h"
#include "ad7606_pins.h"
#include "core_cm33.h"

static bool adc_initialized = false;
// Init once. Access current cycle on DWT->CYCCNT
// Example cycle:
// while(DWT->CYCCNT - start < desired_cycles){}
static inline void dwt_init(void) {
    // Enable trace
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    // Reset cycle counter
    DWT->CYCCNT = 0;

    // Enable cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline void dwt_delay_cycles(uint32_t cycles) {
    uint32_t start = DWT->CYCCNT;
    while(DWT->CYCCNT - start < cycles){}
}

static inline uint32_t convert_us_to_cycles(uint32_t time_us) {
    // cycles = time_us * SystemCoreClock / 1000000
    return (uint32_t)(((uint64_t)time_us * (uint64_t)SystemCoreClock) / 1000000ULL);
}

static inline void delay_us(uint32_t us) {
    dwt_delay_cycles(convert_us_to_cycles(us));
}

static inline uint32_t convert_ns_to_cycles(uint32_t time_ns) {
    return (uint32_t)(((uint64_t)SystemCoreClock * (uint64_t)time_ns) / 1000000000ULL);
}

static inline void delay_ns(uint32_t ns) {
    dwt_delay_cycles(convert_ns_to_cycles(ns));
}

// These functions exist for clarity instead of using HAL_GPIO_WritePin which is ugly.
static inline void pin_low(GPIO_TypeDef *gpio, uint32_t pin) {
    HAL_GPIO_WritePin(gpio, pin, GPIO_PIN_RESET);
}

static inline void pin_high(GPIO_TypeDef *gpio, uint32_t pin) {
    HAL_GPIO_WritePin(gpio, pin, GPIO_PIN_SET);
}

// For convenience
static inline void pulse_low_ns(GPIO_TypeDef *gpio, uint32_t pin, uint32_t ns) {
    pin_low(gpio, pin);
    delay_ns(ns);
    pin_high(gpio, pin);
    delay_ns(ns);
}

static inline void pulse_high_ns(GPIO_TypeDef *gpio, uint32_t pin, uint32_t ns) {
    pin_high(gpio, pin);
    delay_ns(ns);
    pin_low(gpio, pin);
    delay_ns(ns);
}

static inline void ad7606_gpio_init(void) {
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOF_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();

    GPIO_InitTypeDef init_struct = {0};

    /* Output control pins */
    init_struct.Mode  = GPIO_MODE_OUTPUT_PP;
    init_struct.Pull  = GPIO_NOPULL;
    init_struct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    init_struct.Pin = ADC_CONVST_PIN; HAL_GPIO_Init(ADC_CONVST_GPIO, &init_struct);

    init_struct.Speed = GPIO_SPEED_FREQ_HIGH;
    init_struct.Pin = ADC_CS_PIN;     HAL_GPIO_Init(ADC_CS_GPIO,     &init_struct);
    init_struct.Pin = ADC_STBY_PIN;   HAL_GPIO_Init(ADC_STBY_GPIO,   &init_struct);
    init_struct.Pin = ADC_RESET_PIN;  HAL_GPIO_Init(ADC_RESET_GPIO,  &init_struct);

    /* RD is on GPIOH — must be explicitly initialized as output (resets to analog on STM32U5) */
    init_struct.Pin = ADC_RD_PIN;     HAL_GPIO_Init(ADC_RD_GPIO,     &init_struct);

    /* Input pins — BUSY and data bus */
    init_struct.Mode  = GPIO_MODE_INPUT;
    init_struct.Pull  = GPIO_PULLDOWN;
    init_struct.Speed = GPIO_SPEED_FREQ_HIGH;

    /* BUSY is on GPIOH — must be explicitly initialized as input (resets to analog on STM32U5) */
    init_struct.Pin = ADC_BUSY_PIN;  HAL_GPIO_Init(ADC_BUSY_GPIO,  &init_struct);

    init_struct.Pull  = GPIO_PULLDOWN;
    init_struct.Pin = ADC_DB0_PIN;   HAL_GPIO_Init(ADC_DB0_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB1_PIN;   HAL_GPIO_Init(ADC_DB1_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB2_PIN;   HAL_GPIO_Init(ADC_DB2_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB3_PIN;   HAL_GPIO_Init(ADC_DB3_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB4_PIN;   HAL_GPIO_Init(ADC_DB4_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB5_PIN;   HAL_GPIO_Init(ADC_DB5_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB6_PIN;   HAL_GPIO_Init(ADC_DB6_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB7_PIN;   HAL_GPIO_Init(ADC_DB7_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB8_PIN;   HAL_GPIO_Init(ADC_DB8_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB9_PIN;   HAL_GPIO_Init(ADC_DB9_GPIO,   &init_struct);
    init_struct.Pin = ADC_DB10_PIN;  HAL_GPIO_Init(ADC_DB10_GPIO,  &init_struct);
    init_struct.Pin = ADC_DB11_PIN;  HAL_GPIO_Init(ADC_DB11_GPIO,  &init_struct);
    init_struct.Pin = ADC_DB12_PIN;  HAL_GPIO_Init(ADC_DB12_GPIO,  &init_struct);
    init_struct.Pin = ADC_DB13_PIN;  HAL_GPIO_Init(ADC_DB13_GPIO,  &init_struct);
    init_struct.Pin = ADC_DB14_PIN;  HAL_GPIO_Init(ADC_DB14_GPIO,  &init_struct);
    init_struct.Pin = ADC_DB15_PIN;  HAL_GPIO_Init(ADC_DB15_GPIO,  &init_struct);

    /* Default idle states */
    HAL_GPIO_WritePin(ADC_CONVST_GPIO, ADC_CONVST_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_RD_GPIO,     ADC_RD_PIN,     GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_CS_GPIO,     ADC_CS_PIN,     GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_STBY_GPIO,   ADC_STBY_PIN,   GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_RESET_GPIO,  ADC_RESET_PIN,  GPIO_PIN_RESET);

#if AD7606_DEBUG_HOLD_CS_RD_LOW
    HAL_GPIO_WritePin(ADC_CS_GPIO, ADC_CS_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(ADC_RD_GPIO, ADC_RD_PIN, GPIO_PIN_RESET);
#endif
}

void ad7606_init(void) {
    if (!adc_initialized) {
        ad7606_gpio_init();
        dwt_init();
        pin_high(ADC_RESET_GPIO, ADC_RESET_PIN);
        delay_us(1);
        pin_low(ADC_RESET_GPIO, ADC_RESET_PIN);
        delay_us(1);
        adc_initialized = true;
    }
}

static inline void start_conversion(void) {
    pin_low(ADC_CONVST_GPIO, ADC_CONVST_PIN);
    delay_us(5);
    pin_high(ADC_CONVST_GPIO, ADC_CONVST_PIN);
}

static uint32_t g_busy_timeout_count = 0;
uint32_t ad7606_get_busy_timeout_count(void) { return g_busy_timeout_count; }

static inline void wait_conversion_complete(void) {
    /* Skip past BUSY rising edge (~25ns after CONVST), then poll for BUSY falling.
       10us timeout > 4us conversion time, so data is valid even on timeout.
       Timeout incremented for diagnostics; read always proceeds. */
    delay_ns(100);

    uint32_t timeout = convert_us_to_cycles(10);
    uint32_t start = DWT->CYCCNT;
    while (HAL_GPIO_ReadPin(ADC_BUSY_GPIO, ADC_BUSY_PIN) == GPIO_PIN_SET) {
        if ((DWT->CYCCNT - start) > timeout) {
            g_busy_timeout_count++;
            return;
        }
    }
}

static inline uint16_t read_current_channel(void) {
    uint16_t v = 0;
    if (HAL_GPIO_ReadPin(ADC_DB0_GPIO,  ADC_DB0_PIN))  v |= (1u << 0);
    if (HAL_GPIO_ReadPin(ADC_DB1_GPIO,  ADC_DB1_PIN))  v |= (1u << 1);
    if (HAL_GPIO_ReadPin(ADC_DB2_GPIO,  ADC_DB2_PIN))  v |= (1u << 2);
    if (HAL_GPIO_ReadPin(ADC_DB3_GPIO,  ADC_DB3_PIN))  v |= (1u << 3);
    if (HAL_GPIO_ReadPin(ADC_DB4_GPIO,  ADC_DB4_PIN))  v |= (1u << 4);
    if (HAL_GPIO_ReadPin(ADC_DB5_GPIO,  ADC_DB5_PIN))  v |= (1u << 5);
    if (HAL_GPIO_ReadPin(ADC_DB6_GPIO,  ADC_DB6_PIN))  v |= (1u << 6);
    if (HAL_GPIO_ReadPin(ADC_DB7_GPIO,  ADC_DB7_PIN))  v |= (1u << 7);
    if (HAL_GPIO_ReadPin(ADC_DB8_GPIO,  ADC_DB8_PIN))  v |= (1u << 8);
    if (HAL_GPIO_ReadPin(ADC_DB9_GPIO,  ADC_DB9_PIN))  v |= (1u << 9);
    if (HAL_GPIO_ReadPin(ADC_DB10_GPIO, ADC_DB10_PIN)) v |= (1u << 10);
    if (HAL_GPIO_ReadPin(ADC_DB11_GPIO, ADC_DB11_PIN)) v |= (1u << 11);
    if (HAL_GPIO_ReadPin(ADC_DB12_GPIO, ADC_DB12_PIN)) v |= (1u << 12);
    if (HAL_GPIO_ReadPin(ADC_DB13_GPIO, ADC_DB13_PIN)) v |= (1u << 13);
    if (HAL_GPIO_ReadPin(ADC_DB14_GPIO, ADC_DB14_PIN)) v |= (1u << 14);
    if (HAL_GPIO_ReadPin(ADC_DB15_GPIO, ADC_DB15_PIN)) v |= (1u << 15);
    return v;
}

static inline uint16_t read_channel_sample(void) {
    pin_low(ADC_RD_GPIO, ADC_RD_PIN);
    delay_ns(200);
    uint16_t v = read_current_channel();
    pin_high(ADC_RD_GPIO, ADC_RD_PIN);
    delay_ns(200);
    return v;
}

static inline void read_all_channels(uint16_t *channels, uint8_t num_channels) {
    /* CS stays low for entire read sequence — data bus only driven while CS+RD both low */
    pin_low(ADC_CS_GPIO, ADC_CS_PIN);
    delay_ns(100);

    for (uint8_t i = 0; i < num_channels; i++) {
        pin_low(ADC_RD_GPIO, ADC_RD_PIN);
        delay_ns(200);
        channels[i] = read_current_channel();
        pin_high(ADC_RD_GPIO, ADC_RD_PIN);
        delay_ns(200);
    }

    pin_high(ADC_CS_GPIO, ADC_CS_PIN);
}

void ad7606_read_all_channels(uint16_t *channels, uint8_t num_channels) {
#if AD7606_FORCE_FAKE
    #include "adc_fake_data.h"
    static size_t frame_idx = 0;
    if (!channels || num_channels == 0) return;
    if (g_fake_frame_count > 0u && g_fake_channels > 0u) {
        size_t ch_max = (num_channels < g_fake_channels) ? num_channels : g_fake_channels;
        for (size_t ch = 0; ch < ch_max; ch++) {
            size_t idx = (frame_idx * g_fake_channels + ch) % (g_fake_frame_count * g_fake_channels);
            channels[ch] = g_fake_samples[idx];
        }
        for (size_t ch = ch_max; ch < num_channels; ch++) {
            channels[ch] = 0;
        }
        frame_idx++;
        if (frame_idx >= g_fake_frame_count) frame_idx = 0;
        return;
    }
#endif
    start_conversion();
    wait_conversion_complete();
    read_all_channels(channels, num_channels);
}

void ad7606_normalize_channels(const uint16_t *channels,
                               float *out,
                               uint8_t num_channels) {
    for (uint8_t i = 0; i < num_channels; i++) {
        // Convert raw unsigned sample to normalized float in [-1.0, 1.0) range.
        out[i] = ((float)channels[i] - 16384.0f) / 8192.0f;
    }
}
