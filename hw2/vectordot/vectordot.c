#include <immintrin.h>
#include <math.h>

float vectordot_naive(float *A, float *B, int N) {
  float c = 0.f;
  for (int i = 0; i < N; ++i) {
    c += A[i] * B[i];
  }
  return c;
}

float vectordot_fma(float *A, float *B, int N) {
  float c = 0.f;
  /*
  TODO: FILL IN HERE
  */
  __m256 a256;
  __m256 b256;
  __m256 c256;

  c256 = _mm256_setzero_ps();
  int N2 = N+(8-(N%8));

  for (int i=0;i<N2/8;i++){
    a256 = _mm256_loadu_ps(A+i*8);
    b256 = _mm256_loadu_ps(B+i*8);
    c256 = _mm256_fmadd_ps(a256,b256,c256);
  }

  float *f = (float*)&c256;

  for (int i=0;i<8;i++){
    c+= f[i];
  }
  return c;
}
