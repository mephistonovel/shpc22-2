#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define MAX_NUM_GPU 4
#define TS 16 // tile size
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
                                
  
  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];


  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int row = bx*TS + tx;
  int col = by*TS + ty;

  float Pvalue = 0; // phase value;
  for (int ph = 0; ph < ceil(K/(float)TS); ++ph){ // phase
    if((row<M) && (ph*TS + tx < K))
      Asub[tx][ty] = A[row*K + ph*TS + ty];
    else
      Asub[tx][ty] = 0;
    if((col<N) && (ph*TS + ty < K))
      Bsub[tx][ty] = B[(ph*TS + tx)*N + col];
    else
      Bsub[tx][ty] = 0;

    __syncthreads();

    for(int k = 0; k < TS; k++)
      Pvalue += Asub[tx][k] * Bsub[k][ty];

    __syncthreads();
  }
  
  if((row<M) && (col < N))
    C[row*K+col] = Pvalue;

}

static int mpi_rank, mpi_world_size;

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {


  // Upload A and B matrix to every GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(a_d[i], A + Mbegin[i] * K,
                         (Mend[i] - Mbegin[i]) * K * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(TS, TS);
    dim3 gridDim(ceil(Mend[i] - Mbegin[i])/TS, N/TS);

    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(C + Mbegin[i] * N, c_d[i],
                         (Mend[i] - Mbegin[i]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  // Only root process do something
  if (mpi_rank == 0) {
    CUDA_CALL(cudaGetDeviceCount(&num_devices));

    printf("Using %d devices\n", num_devices);
    for (int i = 0; i < num_devices; i++) {
      cudaDeviceProp prop;
      CUDA_CALL(cudaGetDeviceProperties(&prop, i));

      // Try printing more detailed information here
      printf("GPU %d: %s\n", i, prop.name);
    }

    if (num_devices <= 0) {
      printf("No CUDA device found. Aborting\n");
      exit(1);
    }

    // Setup problem size for each GPU
    for (int i = 0; i < num_devices; i++) {
      Mbegin[i] = (M / num_devices) * i;
      Mend[i] = (M / num_devices) * (i + 1);
    }
    Mend[num_devices - 1] = M;

    // Allocate device memory for each GPU
    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaSetDevice(i));
      CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
      CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
      CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    }
  }
}

void matmul_finalize() {

  // Only root process do something
  if (mpi_rank == 0) {
    // Free all GPU memory
    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaFree(a_d[i]));
      CUDA_CALL(cudaFree(b_d[i]));
      CUDA_CALL(cudaFree(c_d[i]));
    }
  }
}