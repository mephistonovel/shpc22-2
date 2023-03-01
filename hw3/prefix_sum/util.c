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

void alloc_array(double **m, int N) {
  *m = (double *)aligned_alloc(32, sizeof(double) * N);
  if (*m == NULL) {
    printf("Failed to allocate memory for array.\n");
    exit(0);
  }
}

void rand_array(double *m, int N) {
  for (int j = 0; j < N; j++) {
    m[j] = (double)rand() / RAND_MAX - 0.5;
  }
}

void copy_array(double *a, double *b, int N) {
  for (int j = 0; j < N; j++) {
    a[j] = b[j];
  }
}

void zero_array(double *m, int N) { memset(m, 0, sizeof(double) * N); }

void print_vec(double *m, int N) {
  for (int i = 0; i < N; ++i) {
    printf("%+.3f ", m[i]);
  }
  printf("\n");
}

void print_mat(double *m, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%+.3f ", m[i * N + j]);
    }
    printf("\n");
  }
}

void check_prefix_sum(const double *out, const double *in, int N) {
  printf("Validating...\n");

  double answer = 0.f;
  bool is_valid = true;
  double eps = 1e-3;

  for (int i = 0; i < N; ++i) {
    answer += in[i];
    double candidate = out[i];

    if (fabs((candidate - answer) / answer) > eps) {
      printf("out[%d]: correct_value = %f, your_value = %f\n", i, answer,
             candidate);
      is_valid = false;
      break;
    } else {
      answer = out[i];
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}
