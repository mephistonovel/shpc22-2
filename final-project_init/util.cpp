#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

// Defined in main.cpp
extern int mpi_rank, mpi_size;

void *read_binary(const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  CHECK_ERROR(f != NULL, "Failed to read %s", filename);
  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  fclose(f);
  CHECK_ERROR(size_ == ret, "Failed to read %ld bytes from %s", size_,
              filename);
  if (size != NULL)
    *size = size_;
  return buf;
}

void WriteFile(const char *filename, size_t size, void *buf) {
  FILE *f = fopen(filename, "wb");
  CHECK_ERROR(f != NULL, "Failed to write %s", filename);
  size_t ret = fwrite(buf, 1, size, f);
  fclose(f);
  CHECK_ERROR(size == ret, "Failed to write %ld bytes to %s", size, filename);
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}
