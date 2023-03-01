#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[256];
static pthread_t threads[256];

static void *matmul_kernel(void *arg) {
  /*
  TODO: FILL IN HERE
  */
  int min(int a,int b){
    return (a>b)? b: a;
  }
  struct thread_arg *argk = (struct thread_arg *) arg;
  int M = argk->M;
  int N = argk->N;
  int K = argk->K;
  const float (*A)= argk->A;
  const float (*B) = argk->B;
  float (*C)= argk->C;
  int rank = argk->rank;
  int num_threads = argk->num_threads;

  int start = rank * (M / num_threads);
  int end = rank == num_threads - 1 ? M : (rank + 1) * (M / num_threads);

  float A_ik;
  int size = 1024;
  for (int k2 = 0; k2 < K; k2 += size) {
    for (int j2 = 0; j2 < N; j2 += size) {
      for (int i = start; i < end; ++i) {
        for (int k = k2; k < min(k2 + size, K); ++k) {
          A_ik = A[i * K + k];
          for (int j = j2; j < min(j2 + size, N); ++j) {
            C[i * N + j] += A_ik * B[k * N + j];
          }
        }
      }
    }
  }

 
  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
    if (err) {
      printf("pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      printf("pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
}