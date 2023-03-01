#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  // TODO: FILL_IN_HERE
  omp_set_num_threads(num_threads); 
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}
