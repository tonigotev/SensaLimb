#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "ad7606_pins.h"
#include "core_cm33.h"   // or stm32u5xx.h (which includes it)
#include <math.h>

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

static inline uint32_t convert_us_to_cycles(uint32_t time_us){
    // cycles = time_us * CLOCK_FREQUENCY / 1000000
    return (time_us * CLOCK_FREQUENCY) / 1000000;
}

static inline void delay_us(uint32_t us) {
    dwt_delay_cycles(convert_us_to_cycles(us));
}

static inline uint32_t convert_ns_to_cycles(uint32_t time_ns) {
    return (uint32_t)(((uint64_t)CLOCK_FREQUENCY * time_ns) / 1000000000ULL);
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

static inline void ad7606_gpio_init(void){
    // Enable clock for GPIOA and GPIOB
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // Initialize control pins
    GPIO_InitTypeDef init_struct = {0};
    init_struct.pin = ADC_CONVST_PIN | ADC_RD_PIN | ADC_CS_PIN | ADC_OS0_PIN | ADC_OS1_PIN | ADC_OS2_PIN | ADC_STBY_PIN;
    init_struct.mode = GPIO_MODE_OUTPUT_PP;
    init_struct.pull = GPIO_NOPULL;
    init_struct.speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(ADC_CONVST_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_RD_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_CS_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_OS0_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_OS1_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_OS2_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_STBY_GPIO, &init_struct);
    
    // Initialize data bus pins
    init_struct.pin = ADC_DB0_PIN | ADC_DB1_PIN | ADC_DB2_PIN | ADC_DB3_PIN | ADC_DB4_PIN | ADC_DB5_PIN | ADC_DB6_PIN | ADC_DB7_PIN | ADC_DB8_PIN | ADC_DB9_PIN | ADC_DB10_PIN | ADC_DB11_PIN | ADC_DB12_PIN | ADC_DB13_PIN | ADC_DB14_PIN | ADC_DB15_PIN | ADC_BUSY_PIN;
    init_struct.mode = GPIO_MODE_INPUT;
    init_struct.pull = GPIO_NOPULL;
    init_struct.speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(ADC_DB0_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB1_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB2_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB3_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB4_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB5_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB6_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB7_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB8_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB9_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB10_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB11_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB12_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB13_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB14_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_DB15_GPIO, &init_struct);
    HAL_GPIO_Init(ADC_BUSY_GPIO, &init_struct);

    // Init them high
    HAL_GPIO_WritePin(ADC_CONVST_GPIO, ADC_CONVST_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_RD_GPIO, ADC_RD_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_CS_GPIO, ADC_CS_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADC_STBY_GPIO, ADC_STBY_PIN, GPIO_PIN_SET);

    // Init them low
    HAL_GPIO_WritePin(ADC_OS0_GPIO, ADC_OS0_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(ADC_OS1_GPIO, ADC_OS1_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(ADC_OS2_GPIO, ADC_OS2_PIN, GPIO_PIN_RESET);

    adc_initialized = true;
}

static inline void ad7606_init(void){
    if(!adc_initialized){
        ad7606_gpio_init();
        dwt_init();
        adc_initialized = true;
    }
}

static inline void start_conversion(void){
    pulse_low_ns(ADC_CONVST_GPIO, ADC_CONVST_PIN, 30);
}

static inline void wait_conversion_complete(void){
    pin_high(ADC_RD_GPIO, ADC_RD_PIN);
    while(HAL_GPIO_ReadPin(ADC_BUSY_GPIO, ADC_BUSY_PIN) == GPIO_PIN_RESET){}
    while(HAL_GPIO_ReadPin(ADC_BUSY_GPIO, ADC_BUSY_PIN) == GPIO_PIN_SET){}
}

static inline uint16_t read_current_channel(void){
    return (uint16_t)(ADC_DB_GPIO->IDR & 0xFFFFu);
}

static inline uint16_t read_channel_sample(void){
    pin_low(ADC_RD_GPIO, ADC_RD_PIN);
    delay_ns(30);
    uint16_t v = read_current_channel();
    pin_high(ADC_RD_GPIO, ADC_RD_PIN);
    delay_ns(20);
    return v;
}

static inline uint16_t* read_all_channels(uint16_t* channels, uint8_t num_channels){
    pin_low(ADC_CS_GPIO, ADC_CS_PIN);
    pin_high(ADC_RD_GPIO, ADC_RD_PIN);

    for(int i = 0; i < num_channels; i++){
        channels[i] = read_channel_sample();
    }
}

static inline void finish_conversion(void){
    pin_high(ADC_CS_GPIO, ADC_CS_PIN);
    delay_ns(25);
}

void ad7606_read_all_channels(uint16_t* channels, uint8_t num_channels){
    ad7606_init();
    start_conversion();
    wait_conversion_complete();
    read_all_channels(channels, num_channels);
    finish_conversion();
}

main(void){
    uint16_t channels[2000][DEFINED_CHANNELS];
    // 2khz sampling rate on 180mhz clock
    // 180mhz / 2khz = 90k cycles per sample
    // 1/180mhz = 5.56ns
    // 90k * 5.56ns = 500.4us
    for(int i = 0; i < 2000; i++){
        ad7606_read_all_channels(channels[i], DEFINED_CHANNELS);
        delay_us(500);
    }

    // We downsample by 10 to get 200 samples
    // We average the samples to get the RMS
    float sum[DEFINED_CHANNELS] = {0};
    float rms[200][DEFINED_CHANNELS] = {0};
    for(int g = 0; g < 200; g++){
        for(int i = 0; i < DEFINED_CHANNELS; i++){
            for(int j = 0; j < 10; j++){
                sum[i] += (float)channels[g*10 + j][i] * (float)channels[g*10 + j][i];
            }
            rms[g][i] = sqrtf(sum[i] / 10.0f);
            sum[i] = 0.0f;
        }
    }

}