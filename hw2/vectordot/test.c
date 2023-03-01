#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2048


float *matrix_a;
float *matrix_b;
float result[N][N];

void chunked_mm(int chunk, int n_chunks) {
    __m256 va, vb, vc;
    for (int i = chunk*(N/n_chunks); i < (chunk+1)*(N/n_chunks); i++) {
        for (int j = 0; j < N; j++) {
            float buffer[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            vc = _mm256_loadu_ps(buffer);
            for (int k = 0; k < N; k += 8) {
                // load
                va = _mm256_loadu_ps(matrix_a+(i*N)+k); // matrix_a[i][k]
                vb = _mm256_loadu_ps(matrix_b+(j*N)+k); // matrix_b[j][k]

                // fused multiply and add
                vc = _mm256_fmadd_ps(va, vb, vc);
            }
            //vc = _mm256_hadd_ps(vc, vc);
            _mm256_storeu_ps(buffer, vc);
            result[i][j] = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
            //result[i][j] = buffer[0] + buffer[2] + buffer[4] + buffer[6];
        }
    }
}

 // __m256 a_tmp; 
  // __m256 b_tmp; 
  // __m256 c_tmp; 
  // for (int i = start; i < end; ++i){
  //     float * c = C + i * N;
  //     int Jnew = N+(8-N%8)
  //     for (int j = 0; j < Jnew; ++j){
  //         c[j] = 0;
  //     }
  //     int Knew = K+(8-K%8);
  //     for (int k = 0; k < Knew; ++k){
  //       a_tmp = _mm256_loadu_ps(A+k*8)
  //       const float * b = B + k * N;
  //       float a = A[i*K + k];
  //       for (int j = 0; j < N; ++j)
  //         c[j] += a * b[j];
  //     }
  // }

//  for (int i = start; i < end; ++i)
//     {
//         float * c = C + i * N;
//         for (int j = 0; j < N; j += 8)
//             _mm256_storeu_ps(c + j + 0, _mm256_setzero_ps());
//         for (int k = 0; k < K; ++k)
//         {
//             const float * b = B + k * N;
//             __m256 a = _mm256_set1_ps(A[i*K + k]);
//             for (int j = 0; j < N; j += 16)
//             {
//                 _mm256_storeu_ps(c + j + 0, _mm256_fmadd_ps(a,_mm256_loadu_ps(b + j + 0), _mm256_loadu_ps(c + j + 0)));
//                 _mm256_storeu_ps(c + j + 8, _mm256_fmadd_ps(a,_mm256_loadu_ps(b + j + 8), _mm256_loadu_ps(c + j + 8)));
//             }
//         }
//     }
  // __m256 va, vb;
  // __m256 vc = _mm256_setzero_ps();

  // for (int i = start; i < end; ++i) {
  //   for (int j = 0; j < N; ++j){
  //     vc = _mm256_setzero_ps();
  //     float tmpb[K];
  //     float tmpa[K];
  //     for (int tmk = 0; tmk<K;tmk++){
  //       tmpb[tmk] = B[tmk*N+j];
  //       tmpa[tmk] = A[i*K+tmk];
  //     }
  //     for (int k = 0; k < K; k+=8){
  //       va = _mm256_loadu_ps(tmpa+k); // matrix_a[i][k]
  //       vb = _mm256_loadu_ps(tmpb+k); // matrix_b[j][k]
  //       vc = _mm256_fmadd_ps(va, vb, vc);
  //     }
  //   float* f = (float*)&vc;
  //   C[i*N + j] = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7];
  //   }
  //}

  //59.826175....