#pragma once

#include <cstdio>
#include <cstdlib>

/* Useful macros */
#define EXIT(status)                                                           \
  do {                                                                         \
    MPI_Finalize();                                                            \
    exit(status);                                                              \
  } while (0)

#define PRINTF_WITH_RANK(fmt, ...)                                             \
  do {                                                                         \
    printf("[rank %d] " fmt "\n", mpi_rank, ##__VA_ARGS__);                    \
  } while (0)

#define PRINTF_ROOT(fmt, ...)                                                  \
  do {                                                                         \
    if (mpi_rank == 0)                                                         \
      printf("[rank %d] " fmt "\n", mpi_rank, ##__VA_ARGS__);                  \
  } while (0)

#define CHECK_ERROR(cond, fmt, ...)                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      PRINTF_WITH_RANK("[%s:%d] " fmt "\n", __FILE__, __LINE__,                \
                       ##__VA_ARGS__);                                         \
      EXIT(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)

double get_time();
void *read_binary(const char *filename, size_t *size);