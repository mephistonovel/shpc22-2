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

// Assume f is a black-box operation
double f(double x) { return 4.0 / (1.0 + x * x); }

void check_riemannsum(int num_intervals, double parallel_result) {
  double eps = 1e-5;

  double sum = 0.0;
  double h = 1.0 / (double)num_intervals;
  for (int i = 1; i <= num_intervals; i++) {
    double x = h * ((double)i - 0.5);
    sum += h * f(x);
  }

  if (fabsf(parallel_result - sum) > eps &&
      (sum == 0 || fabsf((parallel_result - sum) / sum) > eps)) {

    printf("FAIL\n");
    printf("Correct value: %f, Your value: %f\n", sum, parallel_result);
  } else {
    printf("PASS\n");
  }
}