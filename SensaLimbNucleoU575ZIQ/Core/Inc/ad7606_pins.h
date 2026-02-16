#pragma once
#include "stm32u5xx_hal.h"

// DWT timing uses SystemCoreClock (set by CubeMX).
#define DEFINED_CHANNELS 5

#define ADC_CONVST_GPIO   GPIOA
#define ADC_CONVST_PIN    GPIO_PIN_0

#define ADC_BUSY_GPIO     GPIOA
#define ADC_BUSY_PIN      GPIO_PIN_1

#define ADC_RD_GPIO       GPIOA
#define ADC_RD_PIN        GPIO_PIN_2

#define ADC_CS_GPIO       GPIOA
#define ADC_CS_PIN        GPIO_PIN_3

#define ADC_OS0_GPIO      GPIOA
#define ADC_OS0_PIN       GPIO_PIN_4
#define ADC_OS1_GPIO      GPIOA
#define ADC_OS1_PIN       GPIO_PIN_5
#define ADC_OS2_GPIO      GPIOA
#define ADC_OS2_PIN       GPIO_PIN_6

#define ADC_STBY_GPIO     GPIOA
#define ADC_STBY_PIN      GPIO_PIN_7

// Data bus (DB0..DB15) - placeholders
#define ADC_DA_GPIO   GPIOA
#define ADC_DB_GPIO   GPIOB

#define ADC_DB0_GPIO      GPIOB
#define ADC_DB0_PIN       GPIO_PIN_0

#define ADC_DB1_GPIO      GPIOB
#define ADC_DB1_PIN       GPIO_PIN_1

#define ADC_DB2_GPIO      GPIOB
#define ADC_DB2_PIN       GPIO_PIN_2

#define ADC_DB3_GPIO      GPIOB
#define ADC_DB3_PIN       GPIO_PIN_3

#define ADC_DB4_GPIO      GPIOB
#define ADC_DB4_PIN       GPIO_PIN_4

#define ADC_DB5_GPIO      GPIOB
#define ADC_DB5_PIN       GPIO_PIN_5

#define ADC_DB6_GPIO      GPIOB
#define ADC_DB6_PIN       GPIO_PIN_6

#define ADC_DB7_GPIO      GPIOB
#define ADC_DB7_PIN       GPIO_PIN_7

#define ADC_DB8_GPIO      GPIOB
#define ADC_DB8_PIN       GPIO_PIN_8

#define ADC_DB9_GPIO      GPIOB
#define ADC_DB9_PIN       GPIO_PIN_9

#define ADC_DB10_GPIO     GPIOB
#define ADC_DB10_PIN      GPIO_PIN_10

#define ADC_DB11_GPIO     GPIOB
#define ADC_DB11_PIN      GPIO_PIN_11

#define ADC_DB12_GPIO     GPIOB
#define ADC_DB12_PIN      GPIO_PIN_12

#define ADC_DB13_GPIO     GPIOB
#define ADC_DB13_PIN      GPIO_PIN_13

#define ADC_DB14_GPIO     GPIOB
#define ADC_DB14_PIN      GPIO_PIN_14

#define ADC_DB15_GPIO     GPIOB
#define ADC_DB15_PIN      GPIO_PIN_15
