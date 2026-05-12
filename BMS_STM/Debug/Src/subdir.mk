################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Src/bq.c \
../Src/main.c \
../Src/stm32c0xx_hal_msp.c \
../Src/stm32c0xx_it.c \
../Src/syscalls.c \
../Src/sysmem.c \
../Src/system_stm32c0xx.c 

OBJS += \
./Src/bq.o \
./Src/main.o \
./Src/stm32c0xx_hal_msp.o \
./Src/stm32c0xx_it.o \
./Src/syscalls.o \
./Src/sysmem.o \
./Src/system_stm32c0xx.o 

C_DEPS += \
./Src/bq.d \
./Src/main.d \
./Src/stm32c0xx_hal_msp.d \
./Src/stm32c0xx_it.d \
./Src/syscalls.d \
./Src/sysmem.d \
./Src/system_stm32c0xx.d 


# Each subdirectory must supply rules for building sources it contributes
Src/%.o Src/%.su Src/%.cyclo: ../Src/%.c Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m0plus -std=gnu11 -g3 -DDEBUG -DSTM32 -DSTM32C011F6Px -DSTM32C0 -DSTM32C011xx -DUSE_HAL_DRIVER -c -I../Inc -I../Drivers/STM32C0xx_HAL_Driver/Inc -I../Drivers/CMSIS/Device/ST/STM32C0xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfloat-abi=soft -mthumb -o "$@"

clean: clean-Src

clean-Src:
	-$(RM) ./Src/bq.cyclo ./Src/bq.d ./Src/bq.o ./Src/bq.su ./Src/main.cyclo ./Src/main.d ./Src/main.o ./Src/main.su ./Src/stm32c0xx_hal_msp.cyclo ./Src/stm32c0xx_hal_msp.d ./Src/stm32c0xx_hal_msp.o ./Src/stm32c0xx_hal_msp.su ./Src/stm32c0xx_it.cyclo ./Src/stm32c0xx_it.d ./Src/stm32c0xx_it.o ./Src/stm32c0xx_it.su ./Src/syscalls.cyclo ./Src/syscalls.d ./Src/syscalls.o ./Src/syscalls.su ./Src/sysmem.cyclo ./Src/sysmem.d ./Src/sysmem.o ./Src/sysmem.su ./Src/system_stm32c0xx.cyclo ./Src/system_stm32c0xx.d ./Src/system_stm32c0xx.o ./Src/system_stm32c0xx.su

.PHONY: clean-Src

