#ifndef STM32C0XX_HAL_CONF_H
#define STM32C0XX_HAL_CONF_H

#ifndef USE_HAL_DRIVER
#define USE_HAL_DRIVER
#endif

#ifndef STM32C011xx
#define STM32C011xx
#endif

#include "stm32c0xx.h"

#define HAL_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_EXTI_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_I2C_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED

#define HSE_VALUE    8000000U
#define HSE_STARTUP_TIMEOUT    100U
#define HSI_VALUE    48000000U
#define HSI14_VALUE  14000000U
#define LSI_VALUE    32000U
#define LSE_VALUE    32768U
#define LSE_STARTUP_TIMEOUT    5000U
#define EXTERNAL_CLOCK_VALUE   12288000U
#define EXTERNAL_I2S1_CLOCK_VALUE  EXTERNAL_CLOCK_VALUE

#define VDD_VALUE                    3300U
#define TICK_INT_PRIORITY            0U
#define USE_RTOS                     0U
#define PREFETCH_ENABLE              1U

#define USE_HAL_I2C_REGISTER_CALLBACKS    0U
#define USE_HAL_UART_REGISTER_CALLBACKS   0U

#include "stm32c0xx_hal_cortex.h"
#include "stm32c0xx_hal_dma.h"
#include "stm32c0xx_hal_exti.h"
#include "stm32c0xx_hal_flash.h"
#include "stm32c0xx_hal_gpio.h"
#include "stm32c0xx_hal_i2c.h"
#include "stm32c0xx_hal_pwr.h"
#include "stm32c0xx_hal_rcc.h"
#include "stm32c0xx_hal_uart.h"

#define assert_param(expr) ((void)0U)

#endif /* STM32C0XX_HAL_CONF_H */
