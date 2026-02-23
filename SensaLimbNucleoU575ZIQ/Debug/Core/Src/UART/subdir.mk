################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/UART/nucleo_uart_actions.c \
../Core/Src/UART/nucleo_uart_rx.c \
../Core/Src/UART/nucleo_uart_rx_port.c 

OBJS += \
./Core/Src/UART/nucleo_uart_actions.o \
./Core/Src/UART/nucleo_uart_rx.o \
./Core/Src/UART/nucleo_uart_rx_port.o 

C_DEPS += \
./Core/Src/UART/nucleo_uart_actions.d \
./Core/Src/UART/nucleo_uart_rx.d \
./Core/Src/UART/nucleo_uart_rx_port.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/UART/%.o Core/Src/UART/%.su Core/Src/UART/%.cyclo: ../Core/Src/UART/%.c Core/Src/UART/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32U575xx -c -I../Core/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32U5xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/ST/AI/Inc -I../X-CUBE-AI/App -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-UART

clean-Core-2f-Src-2f-UART:
	-$(RM) ./Core/Src/UART/nucleo_uart_actions.cyclo ./Core/Src/UART/nucleo_uart_actions.d ./Core/Src/UART/nucleo_uart_actions.o ./Core/Src/UART/nucleo_uart_actions.su ./Core/Src/UART/nucleo_uart_rx.cyclo ./Core/Src/UART/nucleo_uart_rx.d ./Core/Src/UART/nucleo_uart_rx.o ./Core/Src/UART/nucleo_uart_rx.su ./Core/Src/UART/nucleo_uart_rx_port.cyclo ./Core/Src/UART/nucleo_uart_rx_port.d ./Core/Src/UART/nucleo_uart_rx_port.o ./Core/Src/UART/nucleo_uart_rx_port.su

.PHONY: clean-Core-2f-Src-2f-UART

