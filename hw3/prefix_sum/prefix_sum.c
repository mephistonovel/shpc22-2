#include <immintrin.h>
#include <math.h>
#include <omp.h>

void prefix_sum_sequential(double *out, const double *in, int N) {
  out[0] = in[0];
  for (int i = 1; i < N; ++i) {
    out[i] = in[i] + out[i - 1];
  }
}

void prefix_sum_parallel(double *out, const double *in, int N) {
  omp_set_num_threads(256);  
  //TODO: FILL_IN_HERE
  double *part;  
  #pragma omp parallel
  {
      int id = omp_get_thread_num();  
      int num_th = omp_get_num_threads();  
      #pragma omp single
      part = malloc(sizeof *part * (num_th+1)), part[0] = 0;

      double s = 0;
      #pragma omp for schedule(static) nowait
      for (int i=0; i<N; i++){
        s += in[i];
        out[i] = s;
      }
      part[id+1] = s;

      #pragma omp barrier

      double setof = 0;
      for(int i=0; i<(id+1); i++) setof += part[i];

      #pragma omp for schedule(static)
      for (int i=0; i<N; i++) out[i] += setof;
  }
  free(part);
}
