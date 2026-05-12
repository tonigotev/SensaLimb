################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c011xx.s \
../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c031xx.s \
../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c051xx.s \
../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c071xx.s \
../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c091xx.s \
../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c092xx.s 

OBJS += \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c011xx.o \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c031xx.o \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c051xx.o \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c071xx.o \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c091xx.o \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c092xx.o 

S_DEPS += \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c011xx.d \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c031xx.d \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c051xx.d \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c071xx.d \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c091xx.d \
./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c092xx.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/%.o: ../Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/%.s Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m0plus -g3 -DDEBUG -DUSE_HAL_DRIVER -c -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfloat-abi=soft -mthumb -o "$@" "$<"

clean: clean-Drivers-2f-CMSIS-2f-Device-2f-ST-2f-STM32C0xx-2f-Source-2f-Templates-2f-gcc

clean-Drivers-2f-CMSIS-2f-Device-2f-ST-2f-STM32C0xx-2f-Source-2f-Templates-2f-gcc:
	-$(RM) ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c011xx.d ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c011xx.o ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c031xx.d ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c031xx.o ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c051xx.d ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c051xx.o ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c071xx.d ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c071xx.o ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c091xx.d ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c091xx.o ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c092xx.d ./Drivers/CMSIS/Device/ST/STM32C0xx/Source/Templates/gcc/startup_stm32c092xx.o

.PHONY: clean-Drivers-2f-CMSIS-2f-Device-2f-ST-2f-STM32C0xx-2f-Source-2f-Templates-2f-gcc

