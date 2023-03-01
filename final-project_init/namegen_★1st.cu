#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

#include <cuda_runtime.h> //cuda 추가
#include <mpi.h> //MPI 포함
#define MYTAG 29202 //통신에 사용한다.
#define MAX_NUM_NODE 4 //최대 NODE수
#define MAX_NUM_GPU 4 // 최대 GPU 수
#define MAX_BS 1024 // 최대 Batch Size(메모리 한계 고려)
#define TS 16 // tile 사이즈 (SM 내에서 사용할 local memory size에 대응) 실험결과 16이 제일 빠름
#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }


// Defined in main.cpp
extern int mpi_rank, mpi_size;

int num_devices = 0; //GPU개수 노드별로 파악해둘 예정
static int ele_num_per_node = 0; //노드별 담당할 단어 개수(init에서 계산한다.)
static int ele_num_per_gpu = 0; // GPU별 담당할 단어 개수(init에서 계산한다.)


/* device GPU 올릴 때 사용할 버퍼 */
static float *d_input[MAX_NUM_GPU];
static float *d_character_embedding[MAX_NUM_GPU];
static float *d_nx512[MAX_NUM_GPU];
static float *d_1024x512[MAX_NUM_GPU];
static float *d_nx1024[MAX_NUM_GPU];
static float *d_nx1024_2[MAX_NUM_GPU];
static float *d_1024x1024[MAX_NUM_GPU];
static float *d_1024[MAX_NUM_GPU];
static float *d_256x1024[MAX_NUM_GPU];
static float *d_nx256[MAX_NUM_GPU];
static float *d_nx256_2[MAX_NUM_GPU];
static float *d_256[MAX_NUM_GPU];
static float *d_n[MAX_NUM_GPU];
static float *d_n_2[MAX_NUM_GPU];



// You can modify the data structure as you want
struct Tensor {

  /* Alloc memory */
  Tensor(std::vector<int> shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
    memcpy(buf, buf_, n * sizeof(float));
  }

  ~Tensor() {
    if (buf != nullptr)
      free(buf);
  }

  void set_zero() {
    size_t n = num_elem();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++)
      buf[i] = 0.0;
    #pragma omp barrier
  }

  void set_sos() {
    size_t n = num_elem();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++)
      buf[i] = SOS;
    #pragma omp barrier
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz;
  }

  // Pointer to data
  float *buf = nullptr;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  size_t ndim = 0;
  size_t shape[4];
};


__global__ void embedding_kernel(float *input, float *character_embedding, float *emb_out, int emb_dim) {

  int emb_col = threadIdx.x;
  int emb_row = blockIdx.x;
  
  __shared__ int ref_row; 

  if (emb_col ==0) {
    ref_row = (int)(input[emb_row]);
  }
  __syncthreads(); 
  
  emb_out[emb_row * emb_dim + emb_col] = character_embedding[ref_row * emb_dim + emb_col];
  __syncthreads();
}

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  

  //위치식별                              
  int lRow = threadIdx.y; 
  int lCol = threadIdx.x;
  int gRow = TS * blockIdx.y + lRow; 
  int gCol = TS * blockIdx.x + lCol; 

  //local variable을 이용한 tiling 사용
  __shared__ float A_tiled[TS][TS];
  __shared__ float B_tiled[TS][TS];

  //결과값 저장할 local variable
  float tmp_sum = 0.0;

  //tile 수 : [K보다 크거나 같은 TS배수]를 TS로 나눈 값으로 한다
  // int n_tiles = ((K + TS - 1) / TS * TS) / TS ;
  int n_tiles = (K + TS - 1) / TS;

  //tile에 data 옮기기
  // local-memory 에 있는 변수로 global memory에 있는 정보를 옮긴다. 
  // (이때 여러 work-item이 동시에 이를 수행하여 속도를 높인다.)
  for (int t = 0; t < n_tiles; t++)
  {
    int t_row = TS * t + lRow;
    int t_col = TS * t + lCol;
   
    if (gRow < M && t_col < K){
      A_tiled[lRow][lCol] = A[gRow * K + t_col];
    }
    else {
      A_tiled[lRow][lCol] = 0.0;
    }
      
    if (gCol < N && t_row < K) {
      // B_tiled[lRow][lCol] = B[t_row * N + gCol];
      B_tiled[lRow][lCol] = B[t_row + gCol*K];
    }
    else {
      B_tiled[lRow][lCol] = 0.0;
    }

    // 여러 work-item이 동시에 local memory상의 변수 안에 값을 옮기고 있으므로 
    // 작업이 완료된 후 다음 단계가 이루어지도록 sync 맞추어준다.
    __syncthreads(); 

    // A_tile, B_tiled로부터 담당하고 있는 부분을 연산한다.
    for (int k = 0; k < TS; k++)
    {
      tmp_sum += A_tiled[lRow][k] * B_tiled[k][lCol];
    }

    // 다음 loop에서 load하는 작업이 시작하기 전에 sync를 맞추어준다.(계산 종료 확인)   
    __syncthreads();
  }

  // 결과값 저장
  if (gRow < M && gCol < N)
    C[gRow * N + gCol] = tmp_sum;

}


__global__ void elewise_add_broadcast_kernel(float *input_mat, float *input_vec, float *out, int M, int K) {
  int col = threadIdx.x;
  int row = blockIdx.x;
  // float mat_val = input_mat[row * K + col];
  // float vec_val = input_vec[col];
  // out[row * K + col] = mat_val + vec_val;
  out[row * K + col] = input_mat[row * K + col] + input_vec[col];
}

__global__ void softmax_kernel(float *input, float *output, int M, int K) {
  extern __shared__ float sdata[];
  int col = threadIdx.x;
  int row = blockIdx.x;
  
  __shared__ float sum ; sum = 0.f;
  float tmp = expf(input[row*K+col]);
  sdata[col] = tmp;
  __syncthreads(); 

  for (unsigned int s=K/2; s>0; s/=2) {
    if (col < s) {
      sdata[col]+=sdata[col+s];
    }
    __syncthreads();
  }

  if (col==0) {sum = sdata[0];}
  __syncthreads(); 

  output[row*K+col]=tmp/sum;
}


__global__ void random_select_and_set_kernel(float *char_prob, float *rfloats, float *output, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float psum = 0.0;
  float ref = rfloats[row];
  int flag =0;
  for (int j=0; j<K; j++){
    psum += char_prob[row*K+j];
    if (psum > ref) {
      output[row]=(int)j;
      flag = 1;
      break;
    }
  }
  if (flag==0) {
    output[row]=(int)(K-1);
  }
}



/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

/* Operations */

/*
 * Embedding
 * input: [ele_num_per_node] 
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [ele_num_per_node x EMBEDDING_DIM]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_input[i], input->buf+ele_num_per_gpu*i, ele_num_per_gpu*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_character_embedding[i], weight->buf, NUM_CHAR*EMBEDDING_DIM*sizeof(float), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(EMBEDDING_DIM, 1, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(ele_num_per_gpu, 1, 1); //to-do : 시간 재서 효율적인지 체크 ★★
    CUDA_CALL(cudaSetDevice(i));
    embedding_kernel<<<gridDim, blockDim>>>(d_input[i], d_character_embedding[i], d_nx512[i], EMBEDDING_DIM);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*EMBEDDING_DIM*i, d_nx512[i],
                         ele_num_per_gpu*EMBEDDING_DIM*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

/*
 * Elementwise addition_broadcast
 * input1: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * input2: [K]=[HIDDEN_DIM(1024)]
 * output: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 */
void elemwise_add_broadcast(Tensor *input1, Tensor *input2, Tensor *output) {
  // int M_ = (int)(input1->shape[0]); //ele_num_per_node
  int K_ = (int)(input1->shape[1]); //HIDDEN_DIM1024
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx1024[i], input1->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_1024[i], input2->buf, K_*sizeof(float), cudaMemcpyHostToDevice));
  }
  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(K_, 1, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(ele_num_per_gpu, 1, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    elewise_add_broadcast_kernel<<<gridDim, blockDim>>>(d_nx1024[i], d_1024[i], d_nx1024_2[i], ele_num_per_gpu, K_);
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*K_*i, d_nx1024_2[i],
                         ele_num_per_gpu*K_*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

/*
 * Elementwise addition_broadcast2
 * input1: [M x K]=[ele_num_per_node x NUM_CHAR(256)]
 * input2: [K]=[NUM_CHAR(256)]
 * output: [M x K]=[ele_num_per_node x NUM_CHAR(256)]
 */
void elemwise_add_broadcast2(Tensor *input1, Tensor *input2, Tensor *output) {
  // int M_ = (int)(input1->shape[0]); //ele_num_per_node
  int K_ = (int)(input1->shape[1]); //NUM_CHAR(256)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx256[i], input1->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_256[i], input2->buf, K_*sizeof(float), cudaMemcpyHostToDevice));
  }
  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(K_, 1, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(ele_num_per_gpu, 1, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    elewise_add_broadcast_kernel<<<gridDim, blockDim>>>(d_nx256[i], d_256[i], d_nx256_2[i], ele_num_per_gpu, K_);
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*K_*i, d_nx256_2[i],
                         ele_num_per_gpu*K_*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

/*
 * Elementwise addition
 * input1: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * input2: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * output: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t n = input1->shape[0];
  size_t hidden_dim = input1->shape[1];
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < hidden_dim; j++){
      output->buf[i*hidden_dim + j] = input1->buf[i*hidden_dim + j]+input2->buf[i*hidden_dim + j];
    }
  }
  #pragma omp barrier
}



/*
 * Elementwise (1-x)
 * input: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * output: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)] (same shape as input)
 */
void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->shape[0];
  size_t hidden_dim = input->shape[1];
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < hidden_dim; j++){
      output->buf[i*hidden_dim + j] = 1.0 - input->buf[i*hidden_dim + j];
    }
  }
  #pragma omp barrier
}

/*
 * Elementwise multiplication
 * input1: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * input2: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)](same shape as input1)
 * output: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)](same shape as input1)
  */
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t n = input1->shape[0];
  size_t hidden_dim = input1->shape[1];
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < hidden_dim; j++){
      output->buf[i*hidden_dim + j] = input1->buf[i*hidden_dim + j]*input2->buf[i*hidden_dim + j];
    }
  }
  #pragma omp barrier
}

/*
 * Elementwise tanh(x)
 * input: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * output: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)] (same shape as input)
 */
void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->shape[0];
  size_t hidden_dim = input->shape[1];
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < hidden_dim; j++){
      output->buf[i*hidden_dim + j] = tanhf(input->buf[i*hidden_dim + j]);
    }
  }
  #pragma omp barrier
  
}

/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * output: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)] (same shape as input)
 */
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->shape[0];
  size_t hidden_dim = input->shape[1];
  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < hidden_dim; j++){
      output->buf[i*hidden_dim + j] = 1.0 / (1.0 + expf(-input->buf[i*hidden_dim + j]));
    }
  }
  #pragma omp barrier
}

/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  for (size_t i = 0; i < N_; i++) {
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1->buf[i * K_ + j] * input2->buf[j];
    }
    output->buf[i] = c;
  }
}

/*
 * SGEMM
 * input1: [M x K]=[ele_num_per_node x EMB_DIM(512)]
 * input2: [N x K]=[HIDDEN_DIM(1024) x EMB_DIM(512)] (주의 행열바뀜)
 * output: [M x N]=[ele_num_per_node x HIDDEN_DIM(1024)]
 */
void matmul_1024x512(Tensor *input1, Tensor *input2, Tensor *output) {
  // int M_ = (int)(input1->shape[0]); //ele_num_per_node
  int K_ = (int)(input1->shape[1]); //EMB_DIM=512
  int N_ = (int)(input2->shape[0]); //HIDDEN_DIM=1024

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx512[i], input1->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_1024x512[i], input2->buf, N_*K_*sizeof(float), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(TS, TS, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(N_/TS, (ele_num_per_gpu+TS-1)/TS, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(d_nx512[i], d_1024x512[i], d_nx1024[i], ele_num_per_gpu, N_, K_);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*N_*i, d_nx1024[i],
                         ele_num_per_gpu*N_*sizeof(float), cudaMemcpyDeviceToHost));
  }
}


/*
 * SGEMM
 * input1: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * input2: [N x K]=[HIDDEN_DIM(1024) x HIDDEN_DIM(1024)] (주의 행열바뀜)
 * output: [M x N]=[ele_num_per_node x HIDDEN_DIM(1024)]
 */
void matmul_1024x1024(Tensor *input1, Tensor *input2, Tensor *output) {
  // int M_ = (int)(input1->shape[0]); //ele_num_per_node
  int K_ = (int)(input1->shape[1]); //HIDDEN_DIM=1024
  int N_ = (int)(input2->shape[0]); //HIDDEN_DIM=1024

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx1024[i], input1->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_1024x1024[i], input2->buf, N_*K_*sizeof(float), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(TS, TS, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(N_/TS, (ele_num_per_gpu+TS-1)/TS, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(d_nx1024[i], d_1024x1024[i], d_nx1024_2[i], ele_num_per_gpu, N_, K_);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*N_*i, d_nx1024_2[i],
                         ele_num_per_gpu*N_*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

/*
 * SGEMM
 * input1: [M x K]=[ele_num_per_node x HIDDEN_DIM(1024)]
 * input2: [N x K]=[NUM_CHAR(256) x HIDDEN_DIM(1024)] (주의 행열바뀜)
 * output: [M x N]=[ele_num_per_node x NUM_CHAR(256)]
 */
void matmul_256x1024(Tensor *input1, Tensor *input2, Tensor *output) {
  // int M_ = (int)(input1->shape[0]); //ele_num_per_node
  int K_ = (int)(input1->shape[1]); //HIDDEN_DIM=1024
  int N_ = (int)(input2->shape[0]); //NUM_CHAR=256

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx1024[i], input1->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_256x1024[i], input2->buf, N_*K_*sizeof(float), cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(TS, TS, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(N_/TS, (ele_num_per_gpu+TS-1)/TS, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(d_nx1024[i], d_256x1024[i], d_nx256[i], ele_num_per_gpu, N_, K_);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*N_*i, d_nx256[i],
                         ele_num_per_gpu*N_*sizeof(float), cudaMemcpyDeviceToHost));
  }
}


/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [M x K]=[ele_num_per_node x NUM_CHAR(256)]
 * output: [M x K]=[ele_num_per_node x NUM_CHAR(256)] (same shape as input)
 */
void softmax(Tensor *input, Tensor *output) {
  // int M_ = (int)(input->shape[0]); //ele_num_per_node
  int K_ = (int)(input->shape[1]); //NUM_CHAR(256)
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx256[i], input->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
  }
  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(K_, 1, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(ele_num_per_gpu, 1, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    softmax_kernel<<<gridDim, blockDim, K_*sizeof(float)>>>(d_nx256[i], d_nx256_2[i], ele_num_per_gpu, K_);
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + ele_num_per_gpu*K_*i, d_nx256_2[i],
                         ele_num_per_gpu*K_*sizeof(float), cudaMemcpyDeviceToHost));
  }
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->num_elem(); // 그냥 256인듯?
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * rng_seq: [N*MAX_LEN],

 * char_prob : [M x K] = [ele_num_per_node x NUM_CHAR(256)]
 * rfloat : [M x L] = [ele_num_per_node x MAX_LEN(10)]
 * l : 0 ~ 9
 * output : ★char★ [M x (L+1)] = [ele_num_per_node x (MAX_LEN + 1)]
 * input [ele_num_per_node] 

 */
void random_select_and_set(Tensor *char_prob_v, Tensor *rfloats_v, int l_v, char *output_v, Tensor *input_v) {
  int M_ = (int)(char_prob_v->shape[0]); //ele_num_per_node
  int K_ = (int)(char_prob_v->shape[1]); //NUM_CHAR(256)
  // int L_ = (int)(rfloats_v->shape[1]); //MAX_LEN(10)
  // float ref[M_] ;
  float *ref;
  ref = (float *)malloc(M_ * sizeof(float));
  // #pragma omp parallel for schedule(dynamic)
  for (int i=0; i<M_; i++){
    ref[i] = rfloats_v->buf[i * MAX_LEN + l_v];
    // printf("i : %d, ref[i] : %f\n", i, ref[i]);
  }
  // #pragma omp barrier

  // for (int i=0; i<M_;i++){
  //   float psum = 0.0;
  //   float ref_regi = ref[i];
  //   int flag=0;
  //   for (int j=0; j<K_;j++){
  //     psum += char_prob_v->buf[i*K_+j];
  //     // if (i==0) {
  //     //   printf("[PSUM] i : %d, j: %d, psum : %f\n", i, j, psum);
  //     // }
      
  //     if (psum>ref_regi) {
  //       output_v[i*(MAX_LEN + 1) + l_v]=(int)((size_t)j);
  //       input_v->buf[i]=(int)((size_t)j);
  //       flag=1;
  //       break;
  //     }
  //   }
  //   if (flag==0){
  //     output_v[i*(MAX_LEN + 1) + l_v]=(int)((size_t)(K_-1));
  //     input_v->buf[i]=(int)((size_t)(K_-1));
  //   }
  // }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(d_nx256[i], char_prob_v->buf+ele_num_per_gpu*K_*i, ele_num_per_gpu*K_*sizeof(float), cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_n[i], ref+ele_num_per_gpu*i, ele_num_per_gpu*sizeof(float), cudaMemcpyHostToDevice));
  }
  int num_thread = ele_num_per_gpu;
  int block_size = 1;
  if (num_thread > 1024) {
    num_thread = 1024;
    block_size = ele_num_per_gpu/1024;
  }
  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(num_thread, 1, 1); //to-do : 시간 재서 효율적인지 체크
    dim3 gridDim(block_size, 1, 1); //to-do : 시간 재서 효율적인지 체크
    CUDA_CALL(cudaSetDevice(i));
    random_select_and_set_kernel<<<gridDim, blockDim>>>(d_nx256[i], d_n[i], d_n_2[i], K_);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(ref + ele_num_per_gpu*i, d_n_2[i],
                         ele_num_per_gpu*sizeof(float), cudaMemcpyDeviceToHost));
  }
  #pragma omp parallel for schedule(dynamic)
  for (int i=0; i<M_; i++){
    int tmp = (int)(ref[i]);
    output_v[i*(MAX_LEN + 1) + l_v] = tmp;
    input_v->buf[i] = tmp;
  }
  #pragma omp barrier
}





/* Initialize the model. */
void namegen_initialize(int N, int rng_seed, char *parameter_fname) {

  //MPI준비
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  //각 노드가 device(GPU) 가져오기
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  /* N = 1024 즉, 1024개의 이름을 생성할 경우 node별로 256개를 담당하고, 각 GPU가 64개를 담당하여 계산한다*/
  ele_num_per_node = N / MAX_NUM_NODE; //노드별 담당할 단어 개수(init에서 계산한다.)
  ele_num_per_gpu = ele_num_per_node / num_devices; //노드별 담당할 단어 개수(init에서 계산한다.)

  // Only root process do something /* 단순 프린트만 하도록 한다.*/
  if (mpi_rank == 0) {
    /* num_device는 각 node가 모두 가져오도록 위로 배치한다. */
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


  // 모든 노드에서 parameter 읽고 준비하기
  size_t parameter_binary_size = 0;
  float *parameter = (float *)read_binary(parameter_fname, &parameter_binary_size);


  /* 노드 내 변수 생성 */
  /* Network parameters */
  
  character_embedding = new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

  W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
  W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
  W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
  W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
  W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
  W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

  W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
  W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
  W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
  W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
  W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
  W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

  b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);
  b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);
  b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
  b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);
  b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);
  b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);

  b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);
  b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);
  b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
  b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);
  b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);
  b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);

  W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
  b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);


  /* input, activations, output, etc. */
  
 
  input = new Tensor({ele_num_per_node});
  emb_out = new Tensor({ele_num_per_node, EMBEDDING_DIM});
  hidden0 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  hidden1 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  input->set_sos();
  hidden0->set_zero();
  hidden1->set_zero();    
  
  // printf("CHECK INPUT : %f\n", input->buf[0] );
  // printf("CHECK INPUT : %f\n", input->buf[1] );
  // printf("CHECK INPUT : %f\n", input->buf[2] );
  // printf("CHECK INPUT : %lu\n", input->shape[0] );
  r0 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  r1 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  z0 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  z1 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  n0 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  n1 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  f = new Tensor({ele_num_per_node, NUM_CHAR});

  rtmp00 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp01 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp02 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp03 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp04 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp10 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp11 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp12 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp13 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  rtmp14 = new Tensor({ele_num_per_node, HIDDEN_DIM});

  ztmp00 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp01 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp02 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp03 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp04 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp10 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp11 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp12 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp13 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ztmp14 = new Tensor({ele_num_per_node, HIDDEN_DIM});

  ntmp00 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp01 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp02 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp03 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp04 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp05 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp10 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp11 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp12 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp13 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp14 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  ntmp15 = new Tensor({ele_num_per_node, HIDDEN_DIM});

  htmp00 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  htmp01 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  htmp02 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  htmp10 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  htmp11 = new Tensor({ele_num_per_node, HIDDEN_DIM});
  htmp12 = new Tensor({ele_num_per_node, HIDDEN_DIM});

  rfloats = new Tensor({ele_num_per_node*MAX_LEN});
  ftmp0 = new Tensor({ele_num_per_node, NUM_CHAR});
  char_prob = new Tensor({ele_num_per_node, NUM_CHAR});


  

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    
    /* 노드 내 GPU별 사용할 버퍼 생성*/
    CUDA_CALL(cudaMalloc(&d_input[i], ele_num_per_gpu*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_character_embedding[i], NUM_CHAR*EMBEDDING_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_nx512[i], ele_num_per_gpu*EMBEDDING_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_1024x512[i], HIDDEN_DIM*EMBEDDING_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_nx1024[i], ele_num_per_gpu*HIDDEN_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_nx1024_2[i], ele_num_per_gpu*HIDDEN_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_1024x1024[i], HIDDEN_DIM*HIDDEN_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_1024[i], HIDDEN_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_256x1024[i], NUM_CHAR*HIDDEN_DIM*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_nx256[i], ele_num_per_gpu*NUM_CHAR*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_nx256_2[i], ele_num_per_gpu*NUM_CHAR*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_256[i], NUM_CHAR*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_n[i], ele_num_per_gpu*sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_n_2[i], ele_num_per_gpu*sizeof(float)));
  }

}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {

  // 1~3노드 random float 및 output 생성하기
  if (mpi_rank !=0 ){
    random_floats = (float *)malloc(ele_num_per_node * MAX_LEN * sizeof(float)); //MAX_LEN=10 (최대이름길이)
    output = (char *)malloc(ele_num_per_node * (MAX_LEN + 1) * sizeof(char));  
  }
  
  // random float 통신으로 받기
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Send((float *)(random_floats + i*ele_num_per_node*MAX_LEN), ele_num_per_node*MAX_LEN, MPI_FLOAT, i, MYTAG, MPI_COMM_WORLD);
    }
  }
  else {
    MPI_Recv((float *)(random_floats), ele_num_per_node*MAX_LEN, MPI_FLOAT, 0, MYTAG, MPI_COMM_WORLD, NULL);
  }
  
  //전송 sink 맞추기
  MPI_Barrier(MPI_COMM_WORLD);

  //rfloat으로 옮기기
  memcpy(rfloats->buf, random_floats, ele_num_per_node * MAX_LEN * sizeof(float)); 
  //output 초기화해주기
  if (mpi_rank == 0)
    memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));//rank=0 node에서 0으로 만들어주기
  else
    memset(output, 0, ele_num_per_node * (MAX_LEN + 1) * sizeof(char));
  
  // printf("RANK : %d, CHECK output : %d\n", mpi_rank, output[0] );
  // printf("RANK : %d, CHECK output : %c\n", mpi_rank, output[0]);


  /* Generate N names */

  for (int l = 0; l < MAX_LEN; l++) {
    
    // if (mpi_rank==0 && l==3) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK hidden0[0] : %f\n", mpi_rank, 0, hidden0->buf[0*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[1] : %f\n", mpi_rank, 0, hidden0->buf[0*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[2] : %f\n", mpi_rank, 0, hidden0->buf[0*HIDDEN_DIM+2] );
    // }
    // if (mpi_rank==0 && l==3) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK hidden0[0] : %f\n", mpi_rank, 1, hidden0->buf[1*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[1] : %f\n", mpi_rank, 1, hidden0->buf[1*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[2] : %f\n", mpi_rank, 1, hidden0->buf[1*HIDDEN_DIM+2] );
    // }
    embedding(input, character_embedding, emb_out);
    // if (mpi_rank==0 && l==3) {
    //     printf("\n\n");
    //     printf("RANK : %d, CHECK emb_out[0] : %f\n", mpi_rank, emb_out->buf[0*EMBEDDING_DIM +0] );
    //     printf("RANK : %d, CHECK emb_out[1] : %f\n", mpi_rank, emb_out->buf[0*EMBEDDING_DIM +1] );
    //     printf("RANK : %d, CHECK emb_out[2] : %f\n", mpi_rank, emb_out->buf[0*EMBEDDING_DIM +2] );
    // }
    // if (mpi_rank==0 && l==3) {
    //     printf("\n\n");
    //     printf("RANK : %d, CHECK emb_out[0] : %f\n", mpi_rank, emb_out->buf[1*EMBEDDING_DIM +0] );
    //     printf("RANK : %d, CHECK emb_out[1] : %f\n", mpi_rank, emb_out->buf[1*EMBEDDING_DIM +1] );
    //     printf("RANK : %d, CHECK emb_out[2] : %f\n", mpi_rank, emb_out->buf[1*EMBEDDING_DIM +2] );
    // }
    /* GRU1 - r */
    matmul_1024x512(emb_out, W_ir0, rtmp00);
    matmul_1024x1024(hidden0, W_hr0, rtmp01);
    elemwise_add_broadcast(rtmp00, b_ir0, rtmp02);
    elemwise_add(rtmp02, rtmp01, rtmp03); //
    elemwise_add_broadcast(rtmp03, b_hr0, rtmp04);
    elemwise_sigmoid(rtmp04, r0);
    
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK r0[0] : %f\n", mpi_rank, 0, r0->buf[0*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK r0[1] : %f\n", mpi_rank, 0, r0->buf[0*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK r0[2] : %f\n", mpi_rank, 0, r0->buf[0*HIDDEN_DIM+2] );
    // }
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK r0[0] : %f\n", mpi_rank, 1, r0->buf[1*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK r0[1] : %f\n", mpi_rank, 1, r0->buf[1*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK r0[2] : %f\n", mpi_rank, 1, r0->buf[1*HIDDEN_DIM+2] );
    // }
    /* GRU1 - z */
    matmul_1024x512(emb_out, W_iz0, ztmp00);
    matmul_1024x1024(hidden0, W_hz0, ztmp01);
    elemwise_add_broadcast(ztmp00, b_iz0, ztmp02);
    elemwise_add(ztmp02, ztmp01, ztmp03);
    elemwise_add_broadcast(ztmp03, b_hz0, ztmp04);
    elemwise_sigmoid(ztmp04, z0);
    
    
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK z0[0] : %f\n", mpi_rank, 0, z0->buf[0*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK z0[1] : %f\n", mpi_rank, 0, z0->buf[0*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK z0[2] : %f\n", mpi_rank, 0, z0->buf[0*HIDDEN_DIM+2] );
    // }
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK z0[0] : %f\n", mpi_rank, 1, z0->buf[1*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK z0[1] : %f\n", mpi_rank, 1, z0->buf[1*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK z0[2] : %f\n", mpi_rank, 1, z0->buf[1*HIDDEN_DIM+2] );
    // }
    /* GRU1 - n */
    matmul_1024x512(emb_out, W_in0, ntmp00);
    elemwise_add_broadcast(ntmp00, b_in0, ntmp01);
    matmul_1024x1024(hidden0, W_hn0, ntmp02);
    elemwise_add_broadcast(ntmp02, b_hn0, ntmp03);
    elemwise_mul(r0, ntmp03, ntmp04);
    elemwise_add(ntmp01, ntmp04, ntmp05);
    elemwise_tanh(ntmp05, n0);
    
    // if (mpi_rank==0 && l==3) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK n0[0] : %f\n", mpi_rank, 0, n0->buf[0*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK n0[1] : %f\n", mpi_rank, 0, n0->buf[0*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK n0[2] : %f\n", mpi_rank, 0, n0->buf[0*HIDDEN_DIM+2] );
    // }
    // if (mpi_rank==0 && l==3) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK n0[0] : %f\n", mpi_rank, 1, n0->buf[1*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK n0[1] : %f\n", mpi_rank, 1, n0->buf[1*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK n0[2] : %f\n", mpi_rank, 1, n0->buf[1*HIDDEN_DIM+2] );
    // }
    /* GRU1 - h (hidden) */
    elemwise_oneminus(z0, htmp00);
    elemwise_mul(htmp00, n0, htmp01);
    elemwise_mul(z0, hidden0, htmp02);
    elemwise_add(htmp01, htmp02, hidden0);
    
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK hidden0[0] : %f\n", mpi_rank, 0, hidden0->buf[0*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[1] : %f\n", mpi_rank, 0, hidden0->buf[0*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[2] : %f\n", mpi_rank, 0, hidden0->buf[0*HIDDEN_DIM+2] );
    // }
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK hidden0[0] : %f\n", mpi_rank, 1, hidden0->buf[1*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[1] : %f\n", mpi_rank, 1, hidden0->buf[1*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK hidden0[2] : %f\n", mpi_rank, 1, hidden0->buf[1*HIDDEN_DIM+2] );
    // }


    /* GRU2 - r */
    matmul_1024x1024(hidden0, W_ir1, rtmp10);
    matmul_1024x1024(hidden1, W_hr1, rtmp11);
    elemwise_add_broadcast(rtmp10, b_ir1, rtmp12);
    elemwise_add(rtmp12, rtmp11, rtmp13); 
    elemwise_add_broadcast(rtmp13, b_hr1, rtmp14);
    elemwise_sigmoid(rtmp14, r1);
    /* GRU2 - z */
    matmul_1024x1024(hidden0, W_iz1, ztmp10);
    matmul_1024x1024(hidden1, W_hz1, ztmp11);
    elemwise_add_broadcast(ztmp10, b_iz1, ztmp12);
    elemwise_add(ztmp12, ztmp11, ztmp13);
    elemwise_add_broadcast(ztmp13, b_hz1, ztmp14);
    elemwise_sigmoid(ztmp14, z1);
    /* GRU2 - n */
    matmul_1024x1024(hidden0, W_in1, ntmp10);
    elemwise_add_broadcast(ntmp10, b_in1, ntmp11);
    matmul_1024x1024(hidden1, W_hn1, ntmp12);
    elemwise_add_broadcast(ntmp12, b_hn1, ntmp13);
    elemwise_mul(r1, ntmp13, ntmp14);
    elemwise_add(ntmp11, ntmp14, ntmp15);
    elemwise_tanh(ntmp15, n1);
    /* GRU2 - h (hidden) */
    elemwise_oneminus(z1, htmp10);
    elemwise_mul(htmp10, n1, htmp11);
    elemwise_mul(z1, hidden1, htmp12);
    elemwise_add(htmp11, htmp12, hidden1);
    
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK hidden1[0] : %f\n", mpi_rank, 0, hidden1->buf[0*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK hidden1[1] : %f\n", mpi_rank, 0, hidden1->buf[0*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK hidden1[2] : %f\n", mpi_rank, 0, hidden1->buf[0*HIDDEN_DIM+2] );
    // }
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK hidden1[0] : %f\n", mpi_rank, 1, hidden1->buf[1*HIDDEN_DIM+0] );
    //   printf("RANK : %d, n : %d, CHECK hidden1[1] : %f\n", mpi_rank, 1, hidden1->buf[1*HIDDEN_DIM+1] );
    //   printf("RANK : %d, n : %d, CHECK hidden1[2] : %f\n", mpi_rank, 1, hidden1->buf[1*HIDDEN_DIM+2] );
    // }

    /* Fully connected layer */
    matmul_256x1024(hidden1, W_fc, ftmp0);
    
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK ftmp0[0] : %f\n", mpi_rank, 0, ftmp0->buf[0*NUM_CHAR+0] );
    //   printf("RANK : %d, n : %d, CHECK ftmp0[1] : %f\n", mpi_rank, 0, ftmp0->buf[0*NUM_CHAR+1] );
    //   printf("RANK : %d, n : %d, CHECK ftmp0[2] : %f\n", mpi_rank, 0, ftmp0->buf[0*NUM_CHAR+2] );
    // }
    // if (mpi_rank==0 && l==0) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK ftmp0[0] : %f\n", mpi_rank, 1, ftmp0->buf[1*NUM_CHAR+0] );
    //   printf("RANK : %d, n : %d, CHECK ftmp0[1] : %f\n", mpi_rank, 1, ftmp0->buf[1*NUM_CHAR+1] );
    //   printf("RANK : %d, n : %d, CHECK ftmp0[2] : %f\n", mpi_rank, 1, ftmp0->buf[1*NUM_CHAR+2] );
    // }
    elemwise_add_broadcast2(ftmp0, b_fc, f);
    // if (mpi_rank==0 && l==3) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK f[0] : %f\n", mpi_rank, 0, f->buf[0*NUM_CHAR+0] );
    //   printf("RANK : %d, n : %d, CHECK f[1] : %f\n", mpi_rank, 0, f->buf[0*NUM_CHAR+1] );
    //   printf("RANK : %d, n : %d, CHECK f[2] : %f\n", mpi_rank, 0, f->buf[0*NUM_CHAR+2] );
    // }
    // if (mpi_rank==0 && l==3) {
    //   printf("\n\n");
    //   printf("RANK : %d, n : %d, CHECK f[0] : %f\n", mpi_rank, 1, f->buf[1*NUM_CHAR+0] );
    //   printf("RANK : %d, n : %d, CHECK f[1] : %f\n", mpi_rank, 1, f->buf[1*NUM_CHAR+1] );
    //   printf("RANK : %d, n : %d, CHECK f[2] : %f\n", mpi_rank, 1, f->buf[1*NUM_CHAR+2] );
    // }

    /* Softmax */
    softmax(f, char_prob);
  //   if (mpi_rank==0 && l==3) {
  //     printf("\n\n");
  //     for (int i=0; i<256; i++) {
  //       printf("RANK : %d, n : %d, CHECK char_prob[%d] : %f\n", mpi_rank, 0, i, char_prob->buf[0*NUM_CHAR+i] );
  //     }
      
  //   }
  //   if (mpi_rank==0 && l==3) {
  //     printf("\n\n");
  //     for (int i=0; i<256; i++) {
  //       printf("RANK : %d, n : %d, CHECK char_prob[%d] : %f\n", mpi_rank, 1, i, char_prob->buf[1*NUM_CHAR+i] );
  //     }
  //   }
  //   /* Random select and prepare for next iteration */
    random_select_and_set(char_prob, rfloats, l, output, input);
  //   if (mpi_rank==0 && l==3) {
  //     printf("\n\n");
  //     printf("RANK : %d, n : %d, l : %d, CHECK rfloat : %f\n", mpi_rank, 0, l, rfloats->buf[0*MAX_LEN+l] );
  //     // printf("RANK : %d, n : %d, l : %d, CHECK selected_char : %d\n", mpi_rank, 0, l, selected_char );
  //     printf("RANK : %d, n : %d, l : %d, CHECK output[n * (MAX_LEN + 1) + l] : %c\n", mpi_rank, 0, l, output[0 * (MAX_LEN + 1) + l]);
  //     printf("RANK : %d, n : %d, l : %d, CHECK input->buf[0] : %f\n", mpi_rank, 0, l, input->buf[0] );
  //   }
  //   if (mpi_rank==0 && l==3) {
  //     printf("\n\n");
  //     printf("RANK : %d, n : %d, l : %d, CHECK rfloat : %f\n", mpi_rank, 1, l, rfloats->buf[1*MAX_LEN+l] );
  //     // printf("RANK : %d, n : %d, l : %d, CHECK selected_char : %d\n", mpi_rank, 0, l, selected_char );
  //     printf("RANK : %d, n : %d, l : %d, CHECK output[n * (MAX_LEN + 1) + l] : %c\n", mpi_rank, 1, l, output[1 * (MAX_LEN + 1) + l]);
  //     printf("RANK : %d, n : %d, l : %d, CHECK input->buf[0] : %f\n", mpi_rank, 1, l, input->buf[1] );
  //   }
    
  }

  
  //rank 0에 담기
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Recv(output + i*ele_num_per_node*(MAX_LEN+1), ele_num_per_node*(MAX_LEN+1), MPI_FLOAT, i, MYTAG, MPI_COMM_WORLD, NULL);
    }
  } 
  else {
    MPI_Send(output, ele_num_per_node*(MAX_LEN+1), MPI_FLOAT, 0, MYTAG, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {

  delete character_embedding;
  delete W_ir0;
  delete W_iz0;
  delete W_in0;
  delete W_ir1;
  delete W_iz1;
  delete W_in1;
  delete W_hr0;
  delete W_hz0;
  delete W_hn0;
  delete W_hr1;
  delete W_hz1;
  delete W_hn1;
  delete b_ir0;
  delete b_iz0;
  delete b_in0;
  delete b_ir1;
  delete b_iz1;
  delete b_in1;
  delete b_hr0;
  delete b_hz0;
  delete b_hn0;
  delete b_hr1;
  delete b_hz1;
  delete b_hn1;
  delete W_fc;
  delete b_fc;
  delete rfloats;

  delete input;
  delete emb_out;
  delete hidden0;
  delete hidden1;
  delete r0;
  delete r1;
  delete z0;
  delete z1;
  delete n0;
  delete n1;
  delete f;
  delete char_prob;
  delete rtmp00;
  delete rtmp01;
  delete rtmp02;
  delete rtmp03;
  delete rtmp04;
  delete rtmp10;
  delete rtmp11;
  delete rtmp12;
  delete rtmp13;
  delete rtmp14;
  delete ztmp00;
  delete ztmp01;
  delete ztmp02;
  delete ztmp03;
  delete ztmp04;
  delete ztmp10;
  delete ztmp11;
  delete ztmp12;
  delete ztmp13;
  delete ztmp14;
  delete ntmp00;
  delete ntmp01;
  delete ntmp02;
  delete ntmp03;
  delete ntmp04;
  delete ntmp05;
  delete ntmp10;
  delete ntmp11;
  delete ntmp12;
  delete ntmp13;
  delete ntmp14;
  delete ntmp15;
  delete htmp00;
  delete htmp01;
  delete htmp02;
  delete htmp10;
  delete htmp11;
  delete htmp12;
  delete ftmp0;

}