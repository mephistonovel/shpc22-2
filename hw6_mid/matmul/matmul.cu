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

#define width 4
#define NUM_WORK_ITEM 16
#define tile_size 16
#define MAX_NUM_GPU 4
int num_devices = 4; 
// #define transposex 32
// #define transposey 32 


static int mpi_rank, mpi_world_size;

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

// node 4개에 대해 A local_row 나누기
// GPU각각에 대해 local_row potion 또 나누기
// memcpy로 GPU to node1
// mpi 로 node각각 to rank 0 

// __global__ void transpose_kernel(const int P, const int Q, const __global float* input, __global float*output){
//   const int local_row = threadIdx.x;
//   const int local_col = threadIdx.y;
//   const int ID0 = blockIdx.x*transposex + local_row;
//   const int ID1 = blockIdx.y*transposey + local_col;

//   __shared__ float buffer[transposex][transposey];

//   if (ID0<P && ID1<Q){
//     buffer[local_col][local_row]=input[ID1*P+ID0];
//   }  
  
//   __syncthreads();

//   const int newID0 = blockIdx.y*transposey+local_row;
//   const int newID1 = blockIdx.x*transposey;

//   if (newID0<Q && newID1<P){
//     output[newID1*Q+newID0]=buffer[local_row][local_col];
//   }

// }

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {

// float4 : make_float4(0.0f,0.0f,0.0f,0.0f)

  __shared__ float Asub[tile_size][tile_size];
  __shared__ float Bsub[tile_size][tile_size];


  int bx = blockIdx.x, by = blockIdx.y;
  int local_row = threadIdx.y, local_col = threadIdx.x;

  int row = by*tile_size + local_row;
  int col = bx*tile_size + local_col;

  if (row >= M +tile_size || col >= N +tile_size) {return;}

  float Pvalue = 0; // phase value;
  for (int ph = 0; ph < ceil(K/(float)tile_size); ++ph){ // phase
    
    int t_col=tile_size*ph+local_col;
    int t_loc = tile_size*ph+local_row;
    if((row<M) && (t_col < K))
      Asub[local_row][local_col] = A[row*K + t_col];
    else
      Asub[local_row][local_col] = 0;
    if((col<N) && (t_loc < K))
      Bsub[local_row][local_col] = B[t_loc*N + col];
    else
      Bsub[local_row][local_col] = 0;

    __syncthreads();

    for(int k = 0; k < tile_size; k++)
      Pvalue += Asub[local_row][k] * Bsub[k][local_col];

    __syncthreads();
  }
  
  if((row<M) && (col < N))
    C[row*N+col] = Pvalue;

}


void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU

  // Launch kernel on every GPU

  // mpi bcast B to all nodes
  // mpi scatter A to all nodes

  MPI_Bcast((float *)&B[0], K*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // A 보내고 C에서 받을 거 미리 크기 설정
  int sendcounts[mpi_world_size];
  int recvcounts_c[mpi_world_size];
  

  int a_displ[mpi_world_size];
  int c_displ[mpi_world_size];


  for (int i=0;i<(mpi_world_size-1);i++){
    sendcounts[i] = (M/mpi_world_size)*K;
    recvcounts_c[i] = (M/mpi_world_size)*N;
    // node_part[i] = M/mpi_world_size;
  }
  // node_part[mpi_world_size-1] = M-(M/mpi_world_size)*(mpi_world_size-1);

  sendcounts[mpi_world_size-1] = M*K-((M/mpi_world_size)*K)*(mpi_world_size-1);
  recvcounts_c[mpi_world_size-1]=M*N-((M/mpi_world_size)*N)*(mpi_world_size-1);
  
  a_displ[0] = 0;
  c_displ[0] = 0;
  for (int j=1;j<mpi_world_size;j++){
    a_displ[j] = a_displ[j-1]+sendcounts[j-1];
    c_displ[j] = c_displ[j-1]+recvcounts_c[j-1];
  }

  // float* recvbuf_a;
  // recvbuf_a = (float *)malloc(sizeof(float)*sendcounts[mpi_rank]);

  int Ainit = (M/mpi_world_size)*mpi_rank*K;
  // MPI_Scatterv((void *)A,sendcounts, a_displ, MPI_FLOAT, recvbuf_a, sendcounts[mpi_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv((void *)A,sendcounts, a_displ, MPI_FLOAT, (void*)(A+Ainit), sendcounts[mpi_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(a_d[i], A+Ainit + Mbegin[i] * K,
                         (Mend[i] - Mbegin[i]) * K * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  }

  int Cinit = (M/mpi_world_size) * mpi_rank * N;
  for (int i = 0; i < num_devices; i++) {
    // dim3 blockDim(1, 1, 1);
    // dim3 gridDim(Mend[i] - Mbegin[i], N, 1);
    dim3 blockDim(tile_size, tile_size,1);
    dim3 gridDim((N+tile_size-1)/tile_size,(Mend[i] - Mbegin[i]+tile_size-1)/tile_size,1);

    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], Mend[i]-Mbegin[i], N, K);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  // float* sendbuf;
  // sendbuf = (float*)malloc(sizeof(float)*recvcounts_c[mpi_rank]);

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(C+Cinit + Mbegin[i] * N, c_d[i],
                         (Mend[i] - Mbegin[i]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
  }


  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaDeviceSynchronize());
  } 

  MPI_Gatherv((void*)(C+Cinit),recvcounts_c[mpi_rank],MPI_FLOAT,C,recvcounts_c,c_displ,MPI_FLOAT,0,MPI_COMM_WORLD);
}

// A-> M/n_node 
// B -> 다뿌림 
//initialize: 데이터 뿌리기 to other nodes
void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  // Only root process do something
  // B 는 전체로 다 보내기
  

  if (mpi_rank == 0) {
    // CUDA_CALL(cudaGetDeviceCount(&num_devices));

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
  }
    // Setup problem size for each GPU
  int node_part[mpi_world_size];
  for (int i=0;i<(mpi_world_size-1);i++){
    node_part[i]=M/mpi_world_size;
  }
  node_part[mpi_world_size-1]= M-(M/mpi_world_size)*(mpi_world_size-1);

  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (node_part[mpi_rank] / num_devices) * i; // (각 rank마다 나눠준 크기 / gpu개수) * gpu_id
    Mend[i] = (node_part[mpi_rank] / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = node_part[mpi_rank];

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
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
