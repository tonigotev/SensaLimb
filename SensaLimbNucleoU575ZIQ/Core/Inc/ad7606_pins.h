#pragma once
#include "stm32u5xx_hal.h"

#define DEFINED_CHANNELS 5

/* Control pins */
#define ADC_CONVST_GPIO   GPIOC
#define ADC_CONVST_PIN    GPIO_PIN_13

#define ADC_BUSY_GPIO     GPIOH
#define ADC_BUSY_PIN      GPIO_PIN_0

#define ADC_RD_GPIO       GPIOH
#define ADC_RD_PIN        GPIO_PIN_1

#define ADC_CS_GPIO       GPIOC
#define ADC_CS_PIN        GPIO_PIN_2

#define ADC_STBY_GPIO     GPIOB
#define ADC_STBY_PIN      GPIO_PIN_7

/* Data bus — DB0..DB15 across multiple ports */
/* DB0  */ #define ADC_DB0_GPIO   GPIOG
/* DB0  */ #define ADC_DB0_PIN    GPIO_PIN_1
/* DB1  */ #define ADC_DB1_GPIO   GPIOF
/* DB1  */ #define ADC_DB1_PIN    GPIO_PIN_9
/* DB2  */ #define ADC_DB2_GPIO   GPIOF
/* DB2  */ #define ADC_DB2_PIN    GPIO_PIN_8
/* DB3  */ #define ADC_DB3_GPIO   GPIOF
/* DB3  */ #define ADC_DB3_PIN    GPIO_PIN_2
/* DB4  */ #define ADC_DB4_GPIO   GPIOE
/* DB4  */ #define ADC_DB4_PIN    GPIO_PIN_5
/* DB5  */ #define ADC_DB5_GPIO   GPIOE
/* DB5  */ #define ADC_DB5_PIN    GPIO_PIN_4
/* DB6  */ #define ADC_DB6_GPIO   GPIOE
/* DB6  */ #define ADC_DB6_PIN    GPIO_PIN_2
/* DB7  */ #define ADC_DB7_GPIO   GPIOG
/* DB7  */ #define ADC_DB7_PIN    GPIO_PIN_3
/* DB8  */ #define ADC_DB8_GPIO   GPIOG
/* DB8  */ #define ADC_DB8_PIN    GPIO_PIN_2
/* DB9  */ #define ADC_DB9_GPIO   GPIOD
/* DB9  */ #define ADC_DB9_PIN    GPIO_PIN_3
/* DB10 */ #define ADC_DB10_GPIO  GPIOC
/* DB10 */ #define ADC_DB10_PIN   GPIO_PIN_0
/* DB11 */ #define ADC_DB11_GPIO  GPIOC
/* DB11 */ #define ADC_DB11_PIN   GPIO_PIN_1
/* DB12 */ #define ADC_DB12_GPIO  GPIOB
/* DB12 */ #define ADC_DB12_PIN   GPIO_PIN_0
/* DB13 */ #define ADC_DB13_GPIO  GPIOA
/* DB13 */ #define ADC_DB13_PIN   GPIO_PIN_4
/* DB14 */ #define ADC_DB14_GPIO  GPIOA
/* DB14 */ #define ADC_DB14_PIN   GPIO_PIN_1
/* DB15 */ #define ADC_DB15_GPIO  GPIOA
/* DB15 */ #define ADC_DB15_PIN   GPIO_PIN_0
