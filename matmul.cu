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
#define NUM_WORK_ITEM 32
#define tile_size 32
#define MAX_NUM_GPU 4
int num_devices = 4; 


// node 4개에 대해 A row 나누기
// GPU각각에 대해 row potion 또 나누기
// memcpy로 GPU to node1
// mpi 로 node각각 to rank 0 

__global__ void matmul_kernel(float4 *A, float4 *B, float4 *C, int M, int N,
                              int K) {

// float4 : make_float4(0.0f,0.0f,0.0f,0.0f)

  int row=threadIdx.x, col=threadIdx.y;

  int glob_row = tile_size * blockIdx.x + row;
  int glob_col = (tile_size/width) * blockIdx.y + col;
  if (row >= M || col >= N)
    return;
    
  __shared__ float4 asub[tile_size][tile_size/width];
  __shared__ float4 bsub[tile_size][tile_size/width];

  float4 mediate_val = make_float4(0.0f,0.0f,0.0f,0.0f);

  const int num_tiles = K/tile_size;

  for (int t=0;t<num_tiles;t++){
    const int t_row = tile_size*t+row;
    const int t_col = (tile_size/width)*t +col;

    asub[row][col] = A[glob_row*(K/width)+t_col];
    bsub[row][col] = B[t_row*(N/width)+glob_col];

    __syncthreads();

    float4 veca,vecb;
    float vala;
    for (int k=0;k<tile_size/width;k++){
      veca=asub[row][k];
      for (int w=0;w<width;w++){
        vecb=bsub[width*k+w][col];

        switch(w) {
          case 0: vala = veca.x; break;
          case 1: vala = veca.y; break;
          case 2: vala = veca.z; break;
          case 3: vala = veca.w; break;
        }
        mediate_val.x += vecb.x*vala;
        mediate_val.y += vecb.y*vala;
        mediate_val.z += vecb.z*vala;
        mediate_val.w += vecb.w*vala;
      }
    }
    __syncthreads();
  }
  C[glob_row*(N/width)+glob_col] = mediate_val;
}

static int mpi_rank, mpi_world_size;

// Array of device (GPU) pointers
static float4 *a_d[MAX_NUM_GPU];
static float4 *b_d[MAX_NUM_GPU];
static float4 *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

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
    sendcounts[i] = (M*K)/mpi_world_size;
    recvcounts_c[i] = (M*N)/mpi_world_size;
  }

  sendcounts[mpi_world_size-1] = M*K-((M*K)/mpi_world_size)*(mpi_world_size-1);
  recvcounts_c[mpi_world_size-1]=M*N-((M*N)/mpi_world_size)*(mpi_world_size-1);
  
  a_displ[0] = 0;
  c_displ[0] = 0;
  for (int j=1;j<mpi_world_size;j++){
    a_displ[j] = a_displ[j-1]+sendcounts[j-1];
    c_displ[j] = c_displ[j-1]+recvcounts_c[j-1];
  }

  float* recvbuf_a;
  recvbuf_a = (float *)malloc(sizeof(float)*sendcounts[mpi_rank]);

  MPI_Scatterv((void *)A,sendcounts, a_displ, MPI_FLOAT, recvbuf_a, sendcounts[mpi_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(a_d[i], recvbuf_a + Mbegin[i] * K,
                         (Mend[i] - Mbegin[i]) * K * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  }


  for (int i = 0; i < num_devices; i++) {
    // dim3 blockDim(1, 1, 1);
    // dim3 gridDim(Mend[i] - Mbegin[i], N, 1);
    dim3 blockDim(tile_size, tile_size/width,1);
    dim3 gridDim(((Mend[i]-Mbegin[i])+tile_size-1)/tile_size, (N/width + tile_size/width-1)/(tile_size/width),1);

    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  float* sendbuf;
  sendbuf = (float*)malloc(sizeof(float)*recvcounts_c[mpi_rank]);

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(sendbuf + Mbegin[i] * N, c_d[i],
                         (Mend[i] - Mbegin[i]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
  }


  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  MPI_Gatherv(sendbuf,recvcounts_c[mpi_rank],MPI_FLOAT,C,recvcounts_c,c_displ,MPI_FLOAT,0,MPI_COMM_WORLD);
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
