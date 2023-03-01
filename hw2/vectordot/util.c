#include "util.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

static double start_time[8];

void timer_init() { srand(time(NULL)); }

static double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void timer_start(int i) { start_time[i] = get_time(); }

double timer_stop(int i) { return get_time() - start_time[i]; }

void alloc_array(float **m, int N) {
  *m = (float *)aligned_alloc(32, sizeof(float) * N);
  if (*m == NULL) {
    printf("Failed to allocate memory for array.\n");
    exit(0);
  }
}

void rand_array(float *m, int N) {
  for (int j = 0; j < N; j++) {
    m[j] = (float)rand() / RAND_MAX - 0.5;
  }
}

void zero_array(float *m, int N) { memset(m, 0, sizeof(float) * N); }

void print_vec(float *m, int N) {
  for (int i = 0; i < N; ++i) {
    printf("%+.3f ", m[i]);
  }
  printf("\n");
}

void print_mat(float *m, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%+.3f ", m[i * N + j]);
    }
    printf("\n");
  }
}

void check_vectordot(float *A, float *B, float candidate, int N) {
  printf("Validating...\n");

  float answer = 0.f;
  for (int i = 0; i < N; ++i) {
    answer += A[i] * B[i];
  }

  bool is_valid = true;
  float eps = 1e-3;
  if (fabsf(candidate - answer) > eps &&
      (answer == 0 || fabsf((candidate - answer) / answer) > eps)) {
    printf("correct_value = %f, your_value = %f\n", answer, candidate);
    is_valid = false;
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}
