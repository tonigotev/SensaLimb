// System call stubs for newlib (UART1 stdout).
#include "main.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

extern UART_HandleTypeDef huart1;

int _write(int fd, char *ptr, int len) {
    if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
        HAL_UART_Transmit(&huart1, (uint8_t *)ptr, (uint16_t)len, HAL_MAX_DELAY);
        return len;
    }
    errno = EBADF;
    return -1;
}

int _read(int fd, char *ptr, int len) {
    (void)fd;
    (void)ptr;
    (void)len;
    errno = EBADF;
    return -1;
}

int _close(int fd) {
    (void)fd;
    return -1;
}

int _lseek(int fd, int ptr, int dir) {
    (void)fd;
    (void)ptr;
    (void)dir;
    return -1;
}

int _fstat(int fd, struct stat *st) {
    (void)fd;
    if (!st) return -1;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int fd) {
    (void)fd;
    return 1;
}
