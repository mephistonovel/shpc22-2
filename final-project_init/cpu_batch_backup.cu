#include "namegen.h"
#include "util.h"
#include "mpi.h"
#include <cassert>
#include <math.h>
#include <vector>
#include <immintrin.h>
#include "omp.h"


// Defined in main.cpp
extern int mpi_rank, mpi_size;
int num_devices;

static int ghstart[4], ghend[4];
static MPI_Status status;

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

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
    for (size_t i = 0; i < n; i++)
      buf[i] = 0.0;
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

/* Network parameters */
Tensor *character_embedding;

Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
static Tensor *W_ir0_c = new Tensor({HIDDEN_DIM, EMBEDDING_DIM});// h * emb_dim
static Tensor *W_iz0_c = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}); //h*emb_dim
static Tensor *W_in0_c = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}); //h*emb_dim 
static Tensor *W_ir1_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM}); //h*hidden_dim
static Tensor *W_iz1_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM}); //h*h
static Tensor *W_in1_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM}); 

Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
static Tensor *W_hr0_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
static Tensor *W_hz0_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
static Tensor *W_hn0_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
static Tensor *W_hr1_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
static Tensor *W_hz1_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
static Tensor *W_hn1_c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});



Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
static Tensor *b_ir0_c = new Tensor({HIDDEN_DIM});
static Tensor *b_iz0_c = new Tensor({HIDDEN_DIM});
static Tensor *b_in0_c = new Tensor({HIDDEN_DIM});
static Tensor *b_ir1_c = new Tensor({HIDDEN_DIM});
static Tensor *b_iz1_c = new Tensor({HIDDEN_DIM});
static Tensor *b_in1_c = new Tensor({HIDDEN_DIM});

Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
static Tensor *b_hr0_c = new Tensor({HIDDEN_DIM});
static Tensor *b_hz0_c = new Tensor({HIDDEN_DIM});
static Tensor *b_hn0_c = new Tensor({HIDDEN_DIM});
static Tensor *b_hr1_c = new Tensor({HIDDEN_DIM});
static Tensor *b_hz1_c = new Tensor({HIDDEN_DIM});
static Tensor *b_hn1_c = new Tensor({HIDDEN_DIM});


Tensor *W_fc, *b_fc;
static Tensor *W_fc_c = new Tensor({NUM_CHAR, HIDDEN_DIM});
static Tensor *b_fc_c = new Tensor({NUM_CHAR});

Tensor *rfloats;

/* input, activations, output */
Tensor *input_test,*emb_out_test;

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
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */

__global__ void embedding_kernel(Tensor *A, Tensor *weight, Tensor *C) {
  
//tensor A : 1 scalar
//tensor weight: numchar * embdim
//tensor C: embdim
  size_t n=weight->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int x = (int)A->buf[0];
  C->buf[i] = weight->buf[x*n+i];
}

void embedding_batch(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1]; //embedding_dim
  size_t batch = input->shape[0]; //BATCH
  for (size_t j=0;j<batch;j++){
    int x = (int)input->buf[j];
    for (size_t i = 0; i < n; i++) {
      output->buf[i*batch+j] = weight->buf[x * n + i];
    }
  }
}



void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1];
  for (size_t i = 0; i < n; i++) {
    int x = (int)input->buf[0];
    output->buf[i] = weight->buf[x * n + i];
  }
}

/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */

//elementwise bias+matrix row
__global__ void biasadd_kernel(Tensor *A, Tensor *B, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid 
//tensor C: hid*batch
  size_t bs= A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  C->buf[i*bs+j] = A->buf[i*bs+j]+B->buf[i];
}

//elementwise add with matrix and matrix (naive)
__global__ void elemwise_add_kernel_m(Tensor *A, Tensor *B, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  size_t bs= A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  C->buf[i*bs+j] = A->buf[i*bs+j]+B->buf[i*bs+j];
}

__global__ void elemwise_add_kernel(Tensor *A, Tensor *B, Tensor *C) {
  
//tensor A : hid *
//tensor B: hid *
//tensor C: hid*
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  C->buf[i] = A->buf[i]+B->buf[i];
}

//input1 hid*b
//input2 hid
void matrow_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t h= input1->shape[0]; //hid
  size_t bs = input1->shape[1]; //batchsize
  for (size_t i = 0; i < h;i++){
    for (size_t j=0;j<bs;j++){
      float x=input1->buf[i*bs+j];
      output->buf[i*bs+j]= x+input2->buf[i];
    }
  }
}
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }

  // size_t sn = input1->num_elem();
  // int new_n = sn%8 == 0 ? sn : sn + (8-(sn%8));
  // __m256 vec3 = _mm256_setzero_ps();
  // float* pt = (float*) &vec3;
  
  // for (int i=0; i<new_n/8; i++){
  //   __m256 vec1 = _mm256_loadu_ps((input1->buf)+8*i);
  //   __m256 vec2 = _mm256_loadu_ps((input2->buf)+8*i);
  //   vec3 = _mm256_add_ps(vec1, vec2);
  //   memcpy((output->buf)+i*8, pt, sizeof(float)*8);
  //   vec3 = _mm256_setzero_ps();
  // };
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */

 __global__ void elemwise_oneminus_kernel_m(Tensor *A, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  size_t bs= A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  C->buf[i*bs+j] = 1-A->buf[i*bs+j];
}

 __global__ void elemwise_oneminus_kernel(Tensor *A, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  C->buf[i] = 1-A->buf[i];
}

void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();

  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 - x;
  }
}

/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
__global__ void elemwise_mul_kernel_m(Tensor *A, Tensor *B, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  size_t bs= A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  C->buf[i*bs+j] = A->buf[i*bs+j]*B->buf[i*bs+j];
}
__global__ void elemwise_mul_kernel(Tensor *A, Tensor *B, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  C->buf[i] = A->buf[i]*B->buf[i];
}

void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
  // size_t sn = input1->num_elem();
  // int new_n = sn%8 == 0 ? sn : sn + (8-(sn%8));
  // __m256 vec3 = _mm256_setzero_ps();
  // float* pt = (float*) &vec3;
  
  // for (int i=0; i<new_n/8; i++){
  //   __m256 vec1 = _mm256_loadu_ps((input1->buf)+8*i);
  //   __m256 vec2 = _mm256_loadu_ps((input2->buf)+8*i);
  //   vec3 = _mm256_mul_ps(vec1, vec2);
  //   memcpy((output->buf)+i*8, pt, sizeof(float)*8);
  //   vec3 = _mm256_setzero_ps();
  // };
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */

__global__ void elemwise_tanh_kernel_m(Tensor *A, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  size_t bs= A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  C->buf[i*bs+j] = tanhf(A->buf[i*bs+j]);
}

__global__ void elemwise_tanh_kernel(Tensor *A, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  C->buf[i] = tanhf(A->buf[i]);
}

void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();

  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
  }
}

/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
 __global__ void elemwise_sigmoid_kernel_m(Tensor *A, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  size_t bs= A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  C->buf[i*bs+j] = 1/(1+expf(-(A->buf[i*bs+j])));
}
__global__ void elemwise_sigmoid_kernel(Tensor *A, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  C->buf[i] = 1/(1+expf(-(A->buf[i])));
}

void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();

  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
  }
}

/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
 __global__ void elemwise_matvec_kernel(Tensor *A, Tensor *B, Tensor *C) {
  
//tensor A : hid * batch
//tensor B: hid * batch
//tensor C: hid*batch
  // size_t N_ = input1->shape[0];
  size_t K_ = A->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // int j= blockDim.y * blockIdx.y + threadIdx.y;
  float c=0.0;
  for (size_t j=0;j<K_;j++)
    c+= A->buf[i*K_+j]*B->buf[j];
  C->buf[i] = c;

}
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[0];
  size_t K_ = input1->shape[1];

  omp_set_num_threads(32);
  #pragma omp parallel for schedule(static)
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
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */

 __global__ void matmul_kernel(Tensor *A, Tensor *B, Tensor *C) {
  
  size_t M_ = A->shape[0];
  size_t K_ = A->shape[1];
  size_t N_ = B->shape[1];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M_ || j >= N_)
    return;

  C->buf[i * N_ + j] = 0;
  for (int k = 0; k < K_; ++k) {
    C->buf[i * N_ + j] += A->buf[i * K_ + k] * B->buf[k * N_ + j];
  }
}
void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];

  // omp_set_num_threads(32);
  // #pragma omp parallel for schedule(static)
  // for (size_t ii=0;ii<M_;ii+=B){
  //   for (size_t jj=0;jj<N_;jj+=B){
  //     for (size_t kk=0;kk<K_;kk+=B){
  //       for (size_t i=ii;i<min(ii+B,M_);i++){
  //         for (size_t j=jj;j<min(jj+B,N_);j++){
  //           for (size_t k=kk;k<min(kk+B,K_);k++){
  //             output->buf[i*N_+j] += input1->buf[i*K_+k]*input2->buf[k*N_+j];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }


  omp_set_num_threads(32);
  #pragma omp parallel for
  for (size_t i = 0; i < M_; i++) {
    for (size_t j = 0; j < N_; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K_; k++) {
        c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
      }
      output->buf[i * N_ + j] = c;
    }
  }
}




/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
 //input: numchar*bs
 //output: numchar*bs 
void softmax(Tensor *input, Tensor *output) {
  size_t numchar = input->shape[0];
  size_t bs = input->shape[1];
  
  for (size_t j=0;j<bs;j++){
    float sum=0.0;
    for (size_t i = 0; i < numchar; i++) {
      float x = input->buf[i*bs+j];
      sum += expf(x);
    }
    for (size_t k=0;k<numchar;k++){
      float y = input->buf[k*bs+j];
      output->buf[k*bs+j] = expf(y)/sum;
    }
  }
  // for (size_t i = 0; i < n; i++) {
  //   float x = input->buf[i];
  //   output->buf[i] = expf(x) / sum;
  // }
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
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}
//포루프용
//input: numchar*bs
//rng_seq: rfloats: N*maxlen
//rng_offset; n*numlen + l 
int random_select_for(Tensor *input, Tensor *rng_seq, int rng_offset,int bs_num) {
  float r = rng_seq->buf[rng_offset];
  size_t numchar = input->shape[0];
  size_t bs = input->shape[1];

  float psum = 0.0;
  for (size_t i = 0; i < numchar; i++) {
    psum += input->buf[i*bs+bs_num];
    if (psum > r) {
      return i;
    }
  }
  return numchar - 1;
}


/*
 * Initialize the model.
 * Do input-independent job here.
 */

void namegen_initialize(int N, int rng_seed, char *parameter_fname) {
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  // int local_rank = mpi_rank % num_devices;

  int bulk = N/mpi_size;
  for (int i=0; i<mpi_size; i++){
    ghstart[i] = bulk*i;
    ghend[i] = bulk*(i+1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

 

  int bs = bulk;

  /* Only the root process reads the parameter */
  //if (mpi_rank == 0) {

  

  size_t parameter_binary_size = 0;
  float *parameter =
      (float *)read_binary(parameter_fname, &parameter_binary_size);

  // Tensor* k = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
  // Tensor* p = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
  // Tensor* c = new Tensor({HIDDEN_DIM, HIDDEN_DIM});
  //   for (int i=0; i<num_devices; i++){
  //   CUDA_CALL(cudaSetDevice(i));
  //   CUDA_CALL(cudaMalloc(&W_ir1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
  //   CUDA_CALL(cudaMemcpy(W_ir1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})),cudaMemcpyHostToDevice))
  // }

  /* Network parameters */
  for (int i=0; i<num_devices; i++){
    CUDA_CALL(cudaSetDevice(i));
    // character_embedding = new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);
    CUDA_CALL(cudaMalloc(&character_embedding, sizeof(Tensor({NUM_CHAR, EMBEDDING_DIM}))));
    CUDA_CALL(cudaMemcpy(character_embedding, new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0), sizeof(Tensor({NUM_CHAR, EMBEDDING_DIM})), cudaMemcpyHostToDevice));
  
    //W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
    CUDA_CALL(cudaMalloc(&W_ir0, sizeof(Tensor({HIDDEN_DIM, EMBEDDING_DIM}))));
    CUDA_CALL(cudaMemcpy(W_ir0, new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1), sizeof(Tensor({HIDDEN_DIM, EMBEDDING_DIM})), cudaMemcpyHostToDevice));
    
    //W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
    CUDA_CALL(cudaMalloc(&W_iz0, sizeof(Tensor({HIDDEN_DIM, EMBEDDING_DIM}))));
    CUDA_CALL(cudaMemcpy(W_iz0, new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2), sizeof(Tensor({HIDDEN_DIM, EMBEDDING_DIM})), cudaMemcpyHostToDevice));
    
    //W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
    CUDA_CALL(cudaMalloc(&W_in0, sizeof(Tensor({HIDDEN_DIM, EMBEDDING_DIM}))));
    CUDA_CALL(cudaMemcpy(W_in0, new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3), sizeof(Tensor({HIDDEN_DIM, EMBEDDING_DIM})), cudaMemcpyHostToDevice));
    
    //W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
    CUDA_CALL(cudaMalloc(&W_ir1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_ir1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
    CUDA_CALL(cudaMalloc(&W_iz1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_iz1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));
  
    //W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);
    CUDA_CALL(cudaMalloc(&W_in1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_in1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
    CUDA_CALL(cudaMalloc(&W_hr0, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_hr0, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
    CUDA_CALL(cudaMalloc(&W_hz0, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_hz0, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));

    //W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
    CUDA_CALL(cudaMalloc(&W_hn0, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_hn0, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));

    //W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
    CUDA_CALL(cudaMalloc(&W_hr1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_hr1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
    CUDA_CALL(cudaMalloc(&W_hz1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_hz1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);
    CUDA_CALL(cudaMalloc(&W_hn1, sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_hn1, new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12), sizeof(Tensor({HIDDEN_DIM, HIDDEN_DIM})), cudaMemcpyHostToDevice));

    //b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);
    CUDA_CALL(cudaMalloc(&b_ir0, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_ir0, new Tensor({HIDDEN_DIM}, parameter + OFFSET13), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);
    CUDA_CALL(cudaMalloc(&b_iz0, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_iz0, new Tensor({HIDDEN_DIM}, parameter + OFFSET14), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
    CUDA_CALL(cudaMalloc(&b_in0, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_in0, new Tensor({HIDDEN_DIM}, parameter + OFFSET15), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);
    CUDA_CALL(cudaMalloc(&b_ir1, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_ir1, new Tensor({HIDDEN_DIM}, parameter + OFFSET16), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);
    CUDA_CALL(cudaMalloc(&b_iz1, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_iz1, new Tensor({HIDDEN_DIM}, parameter + OFFSET17), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);
    CUDA_CALL(cudaMalloc(&b_in1, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_in1, new Tensor({HIDDEN_DIM}, parameter + OFFSET18), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    

    //b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);
    CUDA_CALL(cudaMalloc(&b_hr0, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_hr0, new Tensor({HIDDEN_DIM}, parameter + OFFSET19), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);
    CUDA_CALL(cudaMalloc(&b_hz0, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_hz0, new Tensor({HIDDEN_DIM}, parameter + OFFSET20), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
    CUDA_CALL(cudaMalloc(&b_hn0, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_hn0, new Tensor({HIDDEN_DIM}, parameter + OFFSET21), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);
    CUDA_CALL(cudaMalloc(&b_hr1, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_hr1, new Tensor({HIDDEN_DIM}, parameter + OFFSET22), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);
    CUDA_CALL(cudaMalloc(&b_hz1, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_hz1, new Tensor({HIDDEN_DIM}, parameter + OFFSET23), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);
    CUDA_CALL(cudaMalloc(&b_hn1, sizeof(Tensor({HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(b_hn1, new Tensor({HIDDEN_DIM}, parameter + OFFSET24), sizeof(Tensor({HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
    CUDA_CALL(cudaMalloc(&W_fc, sizeof(Tensor({NUM_CHAR, HIDDEN_DIM}))));
    CUDA_CALL(cudaMemcpy(W_fc, new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25), sizeof(Tensor({NUM_CHAR, HIDDEN_DIM})), cudaMemcpyHostToDevice));
    
    //b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);
    CUDA_CALL(cudaMalloc(&b_fc, sizeof(Tensor({NUM_CHAR}))));
    CUDA_CALL(cudaMemcpy(b_fc, new Tensor({NUM_CHAR}, parameter + OFFSET26), sizeof(Tensor({NUM_CHAR})), cudaMemcpyHostToDevice));



//고쳐야돼
    //input = new Tensor({1});
    CUDA_CALL(cudaMalloc(&input, sizeof(Tensor({1}))));

    //emb_out = new Tensor({EMBEDDING_DIM});
    CUDA_CALL(cudaMalloc(&emb_out, sizeof(Tensor({EMBEDDING_DIM}))));

    //hidden0 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&hidden0, sizeof(Tensor({HIDDEN_DIM})))); //hid*batch

    //hidden1 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&hidden1, sizeof(Tensor({HIDDEN_DIM}))));
    
    //r0 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&r0, sizeof(Tensor({HIDDEN_DIM}))));
    
    //r1 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&r1, sizeof(Tensor({HIDDEN_DIM}))));
    
    //z0 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&z0, sizeof(Tensor({HIDDEN_DIM}))));
    
    //z1 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&z1, sizeof(Tensor({HIDDEN_DIM}))));
    
    //n0 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&n0, sizeof(Tensor({HIDDEN_DIM}))));
    
    //n1 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&n1, sizeof(Tensor({HIDDEN_DIM}))));
    
    //f = new Tensor({NUM_CHAR});
    CUDA_CALL(cudaMalloc(&f, sizeof(Tensor({NUM_CHAR}))));
    

    //rtmp00 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp00, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp01 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp01, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp02 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp02, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp03 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp03, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp04 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp04, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp10 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp10, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp11 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp11, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp12 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp12, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp13 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp13, sizeof(Tensor({HIDDEN_DIM}))));
    
    //rtmp14 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&rtmp14, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp00 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp00, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp01 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp01, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp02 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp02, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp03 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp03, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp04 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp04, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp10 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp10, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp11 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp11, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp12 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp12, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp13 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp13, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ztmp14 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ztmp14, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp00 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp00, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp01 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp01, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp02 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp02, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp03 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp03, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp04 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp04, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp05 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp05, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp10 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp10, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp11 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp11, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp12 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp12, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp13 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp13, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp14 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp14, sizeof(Tensor({HIDDEN_DIM}))));
    
    //ntmp15 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&ntmp15, sizeof(Tensor({HIDDEN_DIM}))));
    
    //htmp00 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&htmp00, sizeof(Tensor({HIDDEN_DIM}))));
    
    //htmp01 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&htmp01, sizeof(Tensor({HIDDEN_DIM}))));
    
    //htmp02 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&htmp02, sizeof(Tensor({HIDDEN_DIM}))));
    
    //htmp10 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&htmp10, sizeof(Tensor({HIDDEN_DIM}))));
    
    //htmp11 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&htmp11, sizeof(Tensor({HIDDEN_DIM}))));
    
    //htmp12 = new Tensor({HIDDEN_DIM});
    CUDA_CALL(cudaMalloc(&htmp12, sizeof(Tensor({HIDDEN_DIM}))));
    

    //rfloats = new Tensor({N * MAX_LEN});
    CUDA_CALL(cudaMalloc(&rfloats, sizeof(Tensor({N*MAX_LEN}))));
    
    //ftmp0 = new Tensor({NUM_CHAR});
    CUDA_CALL(cudaMalloc(&ftmp0, sizeof(Tensor({NUM_CHAR}))));
    
    //char_prob = new Tensor({NUM_CHAR});
    CUDA_CALL(cudaMalloc(&char_prob, sizeof(Tensor({NUM_CHAR}))));
  }

    character_embedding =
        new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

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
    input = new Tensor({bs});
    emb_out = new Tensor({EMBEDDING_DIM,bs});

    hidden0 = new Tensor({HIDDEN_DIM,bs});
    hidden1 = new Tensor({HIDDEN_DIM,bs});

    r0 = new Tensor({HIDDEN_DIM,bs});
    r1 = new Tensor({HIDDEN_DIM,bs});
    z0 = new Tensor({HIDDEN_DIM,bs});
    z1 = new Tensor({HIDDEN_DIM,bs});
    n0 = new Tensor({HIDDEN_DIM,bs});
    n1 = new Tensor({HIDDEN_DIM,bs});
    f = new Tensor({NUM_CHAR,bs});

    rtmp00 = new Tensor({HIDDEN_DIM,bs});
    rtmp01 = new Tensor({HIDDEN_DIM,bs});
    rtmp02 = new Tensor({HIDDEN_DIM,bs});
    rtmp03 = new Tensor({HIDDEN_DIM,bs});
    rtmp04 = new Tensor({HIDDEN_DIM,bs});
    rtmp10 = new Tensor({HIDDEN_DIM,bs});
    rtmp11 = new Tensor({HIDDEN_DIM,bs});
    rtmp12 = new Tensor({HIDDEN_DIM,bs});
    rtmp13 = new Tensor({HIDDEN_DIM,bs});
    rtmp14 = new Tensor({HIDDEN_DIM,bs});

    ztmp00 = new Tensor({HIDDEN_DIM,bs});
    ztmp01 = new Tensor({HIDDEN_DIM,bs});
    ztmp02 = new Tensor({HIDDEN_DIM,bs});
    ztmp03 = new Tensor({HIDDEN_DIM,bs});
    ztmp04 = new Tensor({HIDDEN_DIM,bs});
    ztmp10 = new Tensor({HIDDEN_DIM,bs});
    ztmp11 = new Tensor({HIDDEN_DIM,bs});
    ztmp12 = new Tensor({HIDDEN_DIM,bs});
    ztmp13 = new Tensor({HIDDEN_DIM,bs});
    ztmp14 = new Tensor({HIDDEN_DIM,bs});

    ntmp00 = new Tensor({HIDDEN_DIM,bs});
    ntmp01 = new Tensor({HIDDEN_DIM,bs});
    ntmp02 = new Tensor({HIDDEN_DIM,bs});
    ntmp03 = new Tensor({HIDDEN_DIM,bs});
    ntmp04 = new Tensor({HIDDEN_DIM,bs});
    ntmp05 = new Tensor({HIDDEN_DIM,bs});
    ntmp10 = new Tensor({HIDDEN_DIM,bs});
    ntmp11 = new Tensor({HIDDEN_DIM,bs});
    ntmp12 = new Tensor({HIDDEN_DIM,bs});
    ntmp13 = new Tensor({HIDDEN_DIM,bs});
    ntmp14 = new Tensor({HIDDEN_DIM,bs});
    ntmp15 = new Tensor({HIDDEN_DIM,bs});

    htmp00 = new Tensor({HIDDEN_DIM,bs});
    htmp01 = new Tensor({HIDDEN_DIM,bs});
    htmp02 = new Tensor({HIDDEN_DIM,bs});
    htmp10 = new Tensor({HIDDEN_DIM,bs});
    htmp11 = new Tensor({HIDDEN_DIM,bs});
    htmp12 = new Tensor({HIDDEN_DIM,bs});

    rfloats = new Tensor({N * MAX_LEN});
    ftmp0 = new Tensor({NUM_CHAR,bs});
    char_prob = new Tensor({NUM_CHAR,bs});
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {

  /* Only root process does the job, for now... */
  // if (mpi_rank != 0)
  //   return;
  if(mpi_rank != 0){
    random_floats = (float *)malloc(N * MAX_LEN * sizeof(float)); //malloc 없애기
    output = (char *)malloc(N * (MAX_LEN + 1) * sizeof(char)); //
  }
  // random_float 송수신 (host0 -> host 1, 2, 3)
  if (mpi_rank==0) {
    for (int i=1; i< mpi_size; i++) {
      MPI_Send((float*) random_floats, N*MAX_LEN, MPI_FLOAT, i, 3, MPI_COMM_WORLD);
    }
  } else {
      MPI_Recv((float*) random_floats, N*MAX_LEN, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &status);
  }

  // memcpy(rfloats->buf, random_floats + (N/mpi_size)*MAX_LEN*mpi_rank, (N/mpi_size) * MAX_LEN * sizeof(float));
  // memset(output, 0, (N/mpi_size) * (MAX_LEN + 1) * sizeof(char));
  memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));

  
  int bs = ghend[mpi_rank]-ghstart[mpi_rank];

  /* Generate N names */
  for (int n = 0; n < bs; n++) {  
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[n] = SOS;
    hidden0->set_zero();//차원 수리 필요 1 : hidden0 hid*bs
    hidden1->set_zero();//차원 수리 필요 2 : hidden1 hid*bsß
    }
  // printf("input_test shape: %lu,%lu,%lu,%lu\n",input_test->shape[0],input_test->shape[1],input_test->shape[2],input_test->shape[3]);
  // printf("emb_out_test shape: %lu,%lu,%lu,%lu\n",emb_out_test->shape[0],emb_out_test->shape[1],emb_out_test->shape[2],emb_out_test->shape[3]);
//   // printf("emb_out_test res 1x16+2: %f\n",emb_out_test->buf[1*16+2]); // 2번째 sos의 1번째 embedding element

// for (int n = ghstart[mpi_rank]; n < ghend[mpi_rank]; n++) {  
//     /* Initialize input and hidden vector. */
//     /* One hidden vector for each GRU layer */
//     input->buf[0] = SOS;
//     hidden0->set_zero();
//     hidden1->set_zero();
//     // printf(" sos확인: %f\n",input->buf[0]);

  for (int l = 0; l < MAX_LEN; l++) {
    /* Embedding */
    
    // printf("emb_out_test shape: %lu,%lu,%lu,%lu\n",emb_out_test->shape[0],emb_out_test->shape[1],emb_out_test->shape[2],emb_out_test->shape[3]);
    // printf("emb_out_test res %d*16+%d: %f\n", n,l,emb_out_test->buf[n*16+l]);
    embedding_batch(input, character_embedding, emb_out); //embdim*bs

    /* First layer r */
    matmul(W_ir0, emb_out, rtmp00); //hid*bs
    matmul(W_hr0, hidden0, rtmp01); //hid*bs 
    matrow_add(rtmp00, b_ir0, rtmp02); // hid*bs
    elemwise_add(rtmp02, rtmp01, rtmp03); //hid*bs -> 
    matrow_add(rtmp03, b_hr0, rtmp04); //hid*bs
    elemwise_sigmoid(rtmp04, r0); //hid*bs

    // matmul(W_ir0, emb_out, rtmp00); //hid*bs
    // matmul(W_hr0, hidden0, rtmp01); //hid*bs (hidden0사이즈 바꿔야함)
    // elemwise_add(rtmp00, rtmp01, rtmp02); //wx + wh -> hid*bs
    // elemwise_add(b_ir0, b_hr0, rtmp03); // bir + bhr -> hid 
    // matrow_add(rtmp02, rtmp03, rtmp04); // hid*bs
    // elemwise_sigmoid(rtmp04, r0);      //hid*bs

    /* First layer z */
    matmul(W_iz0, emb_out, ztmp00); //hid*bs
    matmul(W_hz0, hidden0, ztmp01);//hid*bs (hidden0사이즈 바꿔야함)
    matrow_add(ztmp00, b_iz0, ztmp02);//wx +b_iz0 -> hid*bs
    elemwise_add(ztmp02, ztmp01, ztmp03);// bir + bhr -> hid *bs
    matrow_add(ztmp03, b_hz0, ztmp04);// hid*bs
    elemwise_sigmoid(ztmp04, z0); //hid*bs

    // matmul(W_iz0, emb_out, ztmp00); //hid*bs
    // matvec(W_hz0, hidden0, ztmp01);//hid*bs (hidden0사이즈 바꿔야함)
    // elemwise_add(ztmp00, ztmp01, ztmp02);//wx + wh -> hid*bs
    // elemwise_add(b_iz0, b_hz0, ztmp03);// biz + bhz -> hid 
    // matrow_add(ztmp02,ztmp03, ztmp04);// hid*bs
    // elemwise_sigmoid(ztmp04, z0);      //hid*bs


    /* First layer n */
    // matvec(W_in0, emb_out, ntmp00);
    // elemwise_add(ntmp00, b_in0, ntmp01);
    // matvec(W_hn0, hidden0, ntmp02);
    // elemwise_add(ntmp02, b_hn0, ntmp03);
    // elemwise_mul(r0, ntmp03, ntmp04);
    // elemwise_add(ntmp01, ntmp04, ntmp05);
    // elemwise_tanh(ntmp05, n0);

    matmul(W_in0, emb_out, ntmp00);//win * x = hid*bs
    matrow_add(ntmp00, b_in0, ntmp01);//hid*bs
    matmul(W_hn0, hidden0, ntmp02); //hid*bs
    matrow_add(ntmp02, b_hn0, ntmp03); // hid*bs
    elemwise_mul(r0, ntmp03, ntmp04); //hid*bs
    elemwise_add(ntmp01, ntmp04, ntmp05); //hid*bs
    elemwise_tanh(ntmp05, n0); //hid*bs


    /* First layer h (hidden) */
    elemwise_oneminus(z0, htmp00); //hid*bs 
    elemwise_mul(htmp00, n0, htmp01); //hid*bs
    elemwise_mul(z0, hidden0, htmp02); //hid*bs
    elemwise_add(htmp01, htmp02, hidden0); //hid*bs

    /* Second layer r */
    matmul(W_ir1, hidden0, rtmp10); //hid*bs
    matmul(W_hr1, hidden1, rtmp11); //hid*bs
    matrow_add(rtmp10, b_ir1, rtmp12); //hid*bs
    elemwise_add(rtmp12, rtmp11, rtmp13); //hid*bs
    matrow_add(rtmp13, b_hr1, rtmp14); //hid*bs
    elemwise_sigmoid(rtmp14, r1); //hid*bs

    // matmul(W_ir1, hidden0, rtmp10); //hid*bs
    // matmul(W_hr1, hidden1, rtmp11); //hid*bs
    // elemwise_add(rtmp10,rtmp11, rtmp12); //hid*bs
    // elemwise_add(b_ir1,b_hr1, rtmp13); //hid
    // matrow_add(rtmp12, rtmp13, rtmp14); //hid*bs
    // elemwise_sigmoid(rtmp14, r1); //hid*bs

    /* Second layer z */
    matmul(W_iz1, hidden0, ztmp10); //hid*bs
    matmul(W_hz1, hidden1, ztmp11); //hid*bs
    matrow_add(ztmp10, b_iz1, ztmp12); //hid*bs
    elemwise_add(ztmp12, ztmp11, ztmp13); //hid*bs
    matrow_add(ztmp13, b_hz1, ztmp14); //hid*bs
    elemwise_sigmoid(ztmp14, z1); //hid*bs

    // matmul(W_iz1, hidden0, ztmp10); //hid*bs
    // matmul(W_hz1, hidden1, ztmp11); //hid*bs
    // elemwise_add(ztmp10, ztmp11, ztmp12); //hid*bs
    // elemwise_add(b_iz1, b_hz1, ztmp13); //hid
    // matrow_add(ztmp12, ztmp13, ztmp14); //hid*bs
    // elemwise_sigmoid(ztmp14, z1); //hid*bs

    /* Second layer n */
    // matvec(W_in1, hidden0, ntmp10);
    // elemwise_add(ntmp10, b_in1, ntmp11);
    // matvec(W_hn1, hidden1, ntmp12);
    // elemwise_add(ntmp12, b_hn1, ntmp13);
    // elemwise_mul(r1, ntmp13, ntmp14);
    // elemwise_add(ntmp11, ntmp14, ntmp15);
    // elemwise_tanh(ntmp15, n1);

    matmul(W_in1, hidden0, ntmp10); //hid*bs
    matrow_add(ntmp10, b_in1, ntmp11); //hid*bs
    matmul(W_hn1, hidden1, ntmp12); //hid*bs
    matrow_add(ntmp12, b_hn1, ntmp13); //hid*bs
    elemwise_mul(r1, ntmp13, ntmp14); //hid*bs
    elemwise_add(ntmp11, ntmp14, ntmp15); //hid*bs
    elemwise_tanh(ntmp15, n1); //hid*bs
    

    /* Second layer h (hidden) */
    // elemwise_oneminus(z1, htmp10);
    // elemwise_mul(htmp10, n1, htmp11);
    // elemwise_mul(z1, hidden1, htmp12);
    // elemwise_add(htmp11, htmp12, hidden1); //hid*bs
  
    elemwise_oneminus(z1, htmp10); //hid*bs
    elemwise_mul(htmp10, n1, htmp11); //hid*bs
    elemwise_mul(z1, hidden1, htmp12); //hid*bs
    elemwise_add(htmp11, htmp12, hidden1); //hid*bs


    /* Fully connected layer */
    matmul(W_fc, hidden1, ftmp0); // numchar, hid * hid,bs = numchar*bs
    matrow_add(ftmp0, b_fc, f); //f => numchar*bs

    /* Softmax */
    softmax(f, char_prob);//numchar*bs

    /* Random select */
    // int selected_char = random_select(char_prob, rfloats, n * MAX_LEN + l); //numchar*bs
    //printf("rflaots[%d*%d+%d]: %f\n",n,MAX_LEN,l,rfloats->buf[n*MAX_LEN+l]);

    //bohum1. 포루프 원래 함수 그대로 쓰기 
    //adventure: random select 병렬화용 
    // int bs = char_prob->shape[1];

    int selected_char;
    for (int i=0;i<bs;i++){
      // selected_chars[i]=random_select_for(char_prob,rfloats,i*MAX_LEN+l,i);
      // int n = ghstart[mpi_rank] + i; 
      int n = ghstart[mpi_rank] + i; 
      selected_char=random_select_for(char_prob,rfloats,n*MAX_LEN+l,i);
      //printf("char:%d ,",selected_chars[i]);
      output[n*(MAX_LEN+1)+l] = selected_char; //절대주소로 고쳐야함
      input->buf[i] = selected_char;
    }


    // output[n * (MAX_LEN + 1) + l] = selected_char; // 
    // input->buf[0] = selected_char;
    // // if (selected_char == EOS)
    //   break;
  }
  // printf("emb_out_test %f\n",emb_out_test->buf[1*16+3]);
  // printf("emb_out")


// mpi 송수신

if (mpi_rank == 0) {
  for (int i=1; i < mpi_size; i++) {
    MPI_Recv(output+ghstart[i]*(MAX_LEN+1), (ghend[i]-ghstart[i])*(MAX_LEN+1), MPI_CHAR, i, 4, MPI_COMM_WORLD, &status);
  }
} else {
  MPI_Send(output+ghstart[mpi_rank]*(MAX_LEN+1), (ghend[mpi_rank]-ghstart[mpi_rank])*(MAX_LEN+1), MPI_CHAR, 0, 4, MPI_COMM_WORLD);
}


}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {
  if (mpi_rank == 0) {
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
}