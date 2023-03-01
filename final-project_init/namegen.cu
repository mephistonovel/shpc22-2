#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>
#include <omp.h>
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

#define MAX_NUM_NODE 4 
#define MAX_NUM_GPU 4
#define TS 64 // Tile Size
#define width 4 // the length of vector used in CUDA

// Parameters for MPI communication
extern int mpi_rank, mpi_size;
static MPI_Status status;
static int node_size_start[MAX_NUM_NODE], node_size_end[MAX_NUM_NODE];

// Parameters for CUDA
int num_devices = 0;
static int cuinit[MAX_NUM_GPU], cuend[MAX_NUM_GPU];
static float4 *a_dev[MAX_NUM_GPU];
static float4 *b_dev[MAX_NUM_GPU];
static float4 *c_dev[MAX_NUM_GPU];

// Defined in main.cpp
extern int mpi_rank, mpi_size;

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
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
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


void batch_embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1];
  size_t batch_size = input->shape[0];

  #pragma omp parallel for
  for (size_t j=0;j<batch_size;j++){
    int x = (int)input->buf[j];
    for (size_t i = 0; i < n; i++) {
      output->buf[i*batch_size+j] = weight->buf[x * n + i];
    }
  }
}


/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */


void add_broadcast(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t h= input1->shape[0]; 
  size_t batch_size = input1->shape[1]; 

  #pragma omp parallel for
  for (size_t i = 0; i < h;i++){
    for (size_t j=0;j<batch_size;j++){
      float x=input1->buf[i*batch_size+j];
      output->buf[i*batch_size+j]= x+input2->buf[i];
    }
  }
}


void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();

  #pragma omp parallel for
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */


void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();

  #pragma omp parallel for
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


void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();

  #pragma omp parallel for
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */


void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();


  #pragma omp parallel for
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

void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();

  #pragma omp parallel for
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
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */

__global__ void matmul_kernel(float4 *A, float4 *B, float4 *C, int M, int N, int K) {
  int row=threadIdx.x;
  int col=threadIdx.y;

  int global_row = TS * blockIdx.x + row;
  int global_col = (TS/width) * blockIdx.y + col;

  if (row >= M || col >= N)
    return;
    
  __shared__ float4 a_tile[TS][TS/width];
  __shared__ float4 b_tile[TS][TS/width];

  float4 mediate = make_float4(0.0f,0.0f,0.0f,0.0f);

  const int num_tiles = K/TS;

  for (int t=0;t<num_tiles;t++){
    const int t_row = TS*t+row;
    const int t_col = (TS/width)*t +col;

    a_tile[row][col] = A[global_row*(K/width)+t_col];
    b_tile[row][col] = B[t_row*(N/width)+global_col];

    __syncthreads();

    float4 vector_a,vector_b;
    float value_a;

    for (int k = 0; k < TS/width; k++){
      vector_a = a_tile[row][k];
      for (int w = 0; w < width; w++){
        vector_b=b_tile[width*k+w][col];

        switch(w) {
          case 0: value_a = vector_a.x; break;
          case 1: value_a = vector_a.y; break;
          case 2: value_a = vector_a.z; break;
          case 3: value_a = vector_a.w; break;
        }
        mediate.x += vector_b.x*value_a;
        mediate.y += vector_b.y*value_a;
        mediate.z += vector_b.z*value_a;
        mediate.w += vector_b.w*value_a;
      }
    }
    __syncthreads();
  }
  C[global_row*(N/width)+global_col] = mediate;

}


void matmul(Tensor *input1, Tensor *input2, Tensor *output) 
{
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];
  
  for (int i = 0; i < num_devices; i++)
  {
    cuinit[i] = M_/4 * i ;
    cuend[i] = M_/4 * (i+1);
  }
  
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_dev[i], (cuend[i] - cuinit[i]) * K_ * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_dev[i], K_ * N_ * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_dev[i], (cuend[i] - cuinit[i]) * N_ * sizeof(float)));
  } 


  for (int i = 0; i < num_devices; i++) { 
    CUDA_CALL(cudaMemcpy(a_dev[i], input1->buf + cuinit[i] * K_,
                         (cuend[i] - cuinit[i]) * K_ * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_dev[i], input2->buf, K_ * N_ * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    dim3 gridDim((cuend[i] - cuinit[i] +TS- 1)/TS, (N_/width + TS/width-1)/(TS/width),1);
    dim3 blockDim(TS, TS/width, 1);
    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_dev[i], b_dev[i], c_dev[i], M_, N_, K_);
  }
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(output->buf + cuinit[i] * N_, c_dev[i],
                        (cuend[i] - cuinit[i]) * N_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */

void softmax(Tensor *input, Tensor *output) {
  size_t numchar = input->shape[0];
  size_t batch_size = input->shape[1];
  
  for (size_t j=0; j<batch_size; j++){
    float sum=0.0;
    for (size_t i = 0; i < numchar; i++) {
      float x = input->buf[i*batch_size+j];
      sum += expf(x);
    }
    for (size_t k=0; k<numchar; k++){
      float y = input->buf[k*batch_size+j];
      output->buf[k*batch_size+j] = expf(y)/sum;
    }
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

int random_select_for(Tensor *input, Tensor *rng_seq, int rng_offset,int bs_index) {
  float r = rng_seq->buf[rng_offset];
  size_t numchar = input->shape[0];
  size_t batch_size = input->shape[1];

  float psum = 0.0;
  for (size_t i = 0; i < numchar; i++) {
    psum += input->buf[i*batch_size+bs_index];
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
  for (int i = 0; i < mpi_size; ++i)
  {
    node_size_start[i] = (N / mpi_size) * i;
    node_size_end[i] = (N / mpi_size) * (i + 1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  int bs = 4096;
  size_t parameter_binary_size = 0;
  float *parameter =
      (float *)read_binary(parameter_fname, &parameter_binary_size);

  /* Network parameters */
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
  if(mpi_rank != 0){
    random_floats = (float *)malloc(N * MAX_LEN * sizeof(float)); 
    output = (char *)malloc(N * (MAX_LEN + 1) * sizeof(char));
  }

  if (mpi_rank==0) {
    for (int i=1; i< mpi_size; i++) {
      MPI_Send((float*) random_floats, N*MAX_LEN, MPI_FLOAT, i, 3, MPI_COMM_WORLD);
    }
  } else {
      MPI_Recv((float*) random_floats, N*MAX_LEN, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &status);
  }


  memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));

  int workload = node_size_end[mpi_rank]-node_size_start[mpi_rank];
  int bs = 4096;
  int total_round = workload/bs; 

  for (int batch_loop = 0; batch_loop < total_round ; batch_loop++){
    for (int n = 0; n < bs; n++) {  
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[n] = SOS; 
    }
    hidden0->set_zero();
    hidden1->set_zero();

    for (int l = 0; l < MAX_LEN; l++) {
      /* Embedding */
      batch_embedding(input, character_embedding, emb_out); 

      /* First layer r */
      matmul(W_ir0, emb_out, rtmp00); 
      matmul(W_hr0, hidden0, rtmp01); 
      add_broadcast(rtmp00, b_ir0, rtmp02); 
      elemwise_add(rtmp02, rtmp01, rtmp03); 
      add_broadcast(rtmp03, b_hr0, rtmp04); 
      elemwise_sigmoid(rtmp04, r0); 


      /* First layer z */
      matmul(W_iz0, emb_out, ztmp00); 
      matmul(W_hz0, hidden0, ztmp01);
      add_broadcast(ztmp00, b_iz0, ztmp02);
      elemwise_add(ztmp02, ztmp01, ztmp03);
      add_broadcast(ztmp03, b_hz0, ztmp04);
      elemwise_sigmoid(ztmp04, z0); 


      matmul(W_in0, emb_out, ntmp00);
      add_broadcast(ntmp00, b_in0, ntmp01);
      matmul(W_hn0, hidden0, ntmp02); 
      add_broadcast(ntmp02, b_hn0, ntmp03); 
      elemwise_mul(r0, ntmp03, ntmp04);
      elemwise_add(ntmp01, ntmp04, ntmp05); 
      elemwise_tanh(ntmp05, n0); 


      /* First layer h (hidden) */
      elemwise_oneminus(z0, htmp00); 
      elemwise_mul(htmp00, n0, htmp01); 
      elemwise_mul(z0, hidden0, htmp02); 
      elemwise_add(htmp01, htmp02, hidden0); 

      /* Second layer r */
      matmul(W_ir1, hidden0, rtmp10); 
      matmul(W_hr1, hidden1, rtmp11); 
      add_broadcast(rtmp10, b_ir1, rtmp12); 
      elemwise_add(rtmp12, rtmp11, rtmp13); 
      add_broadcast(rtmp13, b_hr1, rtmp14); 
      elemwise_sigmoid(rtmp14, r1); 

      /* Second layer z */
      matmul(W_iz1, hidden0, ztmp10); 
      matmul(W_hz1, hidden1, ztmp11); 
      add_broadcast(ztmp10, b_iz1, ztmp12); 
      elemwise_add(ztmp12, ztmp11, ztmp13); 
      add_broadcast(ztmp13, b_hz1, ztmp14); 
      elemwise_sigmoid(ztmp14, z1); 

      matmul(W_in1, hidden0, ntmp10); 
      add_broadcast(ntmp10, b_in1, ntmp11); 
      matmul(W_hn1, hidden1, ntmp12); 
      add_broadcast(ntmp12, b_hn1, ntmp13); 
      elemwise_mul(r1, ntmp13, ntmp14);
      elemwise_add(ntmp11, ntmp14, ntmp15); 
      elemwise_tanh(ntmp15, n1); 
      
      elemwise_oneminus(z1, htmp10); 
      elemwise_mul(htmp10, n1, htmp11); 
      elemwise_mul(z1, hidden1, htmp12); 
      elemwise_add(htmp11, htmp12, hidden1); 

      /* Fully connected layer */
      matmul(W_fc, hidden1, ftmp0); 
      add_broadcast(ftmp0, b_fc, f);

      /* Softmax */
      softmax(f, char_prob);

      /* Random select */
      int selected_chars[bs];
      for (int i = 0; i < bs; i++){
        int n = node_size_start[mpi_rank] + batch_loop * bs + i; 
        selected_chars[i]= random_select_for(char_prob, rfloats, n*MAX_LEN+l, i);
        output[n*(MAX_LEN+1)+l] = selected_chars[i]; 
        input->buf[i] = selected_chars[i];
      }
    }
  }

if (mpi_rank == 0) {
  for (int i=1; i < mpi_size; i++) {
    MPI_Recv(output+node_size_start[i]*(MAX_LEN+1), (node_size_end[i]-node_size_start[i])*(MAX_LEN+1), MPI_CHAR, i, 4, MPI_COMM_WORLD, &status);
  }
} else {
  MPI_Send(output+node_size_start[mpi_rank]*(MAX_LEN+1), (node_size_end[mpi_rank]-node_size_start[mpi_rank])*(MAX_LEN+1), MPI_CHAR, 0, 4, MPI_COMM_WORLD);
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