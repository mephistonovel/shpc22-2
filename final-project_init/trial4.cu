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

static int hostBegin[4], hostEnd[4];
// static int BATCH = N/4;
static MPI_Status status;

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
 * input: [Batch] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIMxBatch]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1]; //embedding_dim
  size_t batch = input->shape[0]; //BATCH
  for (size_t j=0;j<batch;j++){
    for (size_t i = 0; i < n; i++) {
      int x = (int)input->buf[j];
      output->buf[i*batch+j] = weight->buf[x * n + i];
    }
  }
}

/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  /*size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }*/

  size_t sn = input1->num_elem();
  int new_n = sn%8 == 0 ? sn : sn + (8-(sn%8));
  __m256 vec3 = _mm256_setzero_ps();
  float* pt = (float*) &vec3;
  
  for (int i=0; i<new_n/8; i++){
    __m256 vec1 = _mm256_loadu_ps((input1->buf)+8*i);
    __m256 vec2 = _mm256_loadu_ps((input2->buf)+8*i);
    vec3 = _mm256_add_ps(vec1, vec2);
    memcpy((output->buf)+i*8, pt, sizeof(float)*8);
    vec3 = _mm256_setzero_ps();
  };
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
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
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  /*size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }*/
  size_t sn = input1->num_elem();
  int new_n = sn%8 == 0 ? sn : sn + (8-(sn%8));
  __m256 vec3 = _mm256_setzero_ps();
  float* pt = (float*) &vec3;
  
  for (int i=0; i<new_n/8; i++){
    __m256 vec1 = _mm256_loadu_ps((input1->buf)+8*i);
    __m256 vec2 = _mm256_loadu_ps((input2->buf)+8*i);
    vec3 = _mm256_mul_ps(vec1, vec2);
    memcpy((output->buf)+i*8, pt, sizeof(float)*8);
    vec3 = _mm256_setzero_ps();
  };
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
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
void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];

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
void softmax(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  float sum = 0.0;
  // omp_set_num_threads(32);
  // #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    sum += expf(x);
  }
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = expf(x) / sum;
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

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, int rng_seed, char *parameter_fname) {
  // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  for (int i=0; i<mpi_size; ++i) {
    hostBegin[i] =  (N/mpi_size)*i;
    hostEnd[i] = (N/mpi_size)*(i+1);
  }
  int BATCH = N/4;

  /* Only the root process reads the parameter -> all nodes read the parameters */
  // if (mpi_rank == 0) {
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

    b_ir0 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET13);
    b_iz0 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET14);
    b_in0 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET15);
    b_ir1 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET16);
    b_iz1 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET17);
    b_in1 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET18);

    b_hr0 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET19);
    b_hz0 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET20);
    b_hn0 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET21);
    b_hr1 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET22);
    b_hz1 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET23);
    b_hn1 = new Tensor({HIDDEN_DIM, BATCH}, parameter + OFFSET24);

    W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
    b_fc = new Tensor({NUM_CHAR, BATCH}, parameter + OFFSET26);

    /* input, activations, output, etc. */
    input = new Tensor({BATCH});
    emb_out = new Tensor({EMBEDDING_DIM,BATCH});

    hidden0 = new Tensor({HIDDEN_DIM});
    hidden1 = new Tensor({HIDDEN_DIM});

    r0 = new Tensor({HIDDEN_DIM,BATCH});
    r1 = new Tensor({HIDDEN_DIM,BATCH});
    z0 = new Tensor({HIDDEN_DIM,BATCH});
    z1 = new Tensor({HIDDEN_DIM,BATCH});
    n0 = new Tensor({HIDDEN_DIM,BATCH});
    n1 = new Tensor({HIDDEN_DIM,BATCH});
    f = new Tensor({NUM_CHAR,BATCH});

    rtmp00 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp01 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp02 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp03 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp04 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp10 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp11 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp12 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp13 = new Tensor({HIDDEN_DIM,BATCH});
    rtmp14 = new Tensor({HIDDEN_DIM,BATCH});

    ztmp00 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp01 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp02 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp03 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp04 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp10 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp11 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp12 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp13 = new Tensor({HIDDEN_DIM,BATCH});
    ztmp14 = new Tensor({HIDDEN_DIM,BATCH});

    ntmp00 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp01 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp02 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp03 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp04 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp05 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp10 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp11 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp12 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp13 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp14 = new Tensor({HIDDEN_DIM,BATCH});
    ntmp15 = new Tensor({HIDDEN_DIM,BATCH});

    htmp00 = new Tensor({HIDDEN_DIM,BATCH});
    htmp01 = new Tensor({HIDDEN_DIM,BATCH});
    htmp02 = new Tensor({HIDDEN_DIM,BATCH});
    htmp10 = new Tensor({HIDDEN_DIM,BATCH});
    htmp11 = new Tensor({HIDDEN_DIM,BATCH});
    htmp12 = new Tensor({HIDDEN_DIM,BATCH});

    rfloats = new Tensor({N * MAX_LEN});
    ftmp0 = new Tensor({NUM_CHAR,BATCH});
    char_prob = new Tensor({NUM_CHAR,BATCH});
  // } else {
  // }
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

  memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));

  

  /* Generate N names */
for (int n = hostBegin[mpi_rank]; n < hostEnd[mpi_rank]; n++) {  //for (int n = 0; n < N; n++) {
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[n] = SOS;
    hidden0->set_zero();
    hidden1->set_zero();}

  for (int l = 0; l < MAX_LEN; l++) {
    /* Embedding */
    embedding(input, character_embedding, emb_out); // batch -> embedding * batch

    /* First layer r */
    matmul(W_ir0, emb_out, rtmp00); //rtmp00= hid x batch
    matmul(W_hr0, hidden0, rtmp01); //rtmp01 = hid x batch, hidden0 = hid x batch
    elemwise_add(rtmp00, b_ir0, rtmp02); //rtmp02 = hid*batch
    elemwise_add(rtmp02, rtmp01, rtmp03); //rtmp03 = hid*batch
    elemwise_add(rtmp03, b_hr0, rtmp04); //rtmp04 = hid*batch
    elemwise_sigmoid(rtmp04, r0); // rtmp04 = hid*batch, r0=hid *batch

    /* First layer z */
    matmul(W_iz0, emb_out, ztmp00); //
    matmul(W_hz0, hidden0, ztmp01);
    elemwise_add(ztmp00, b_iz0, ztmp02);
    elemwise_add(ztmp02, ztmp01, ztmp03);
    elemwise_add(ztmp03, b_hz0, ztmp04);
    elemwise_sigmoid(ztmp04, z0);// z0 = hid*batch

    /* First layer n */
    matmul(W_in0, emb_out, ntmp00);
    elemwise_add(ntmp00, b_in0, ntmp01);
    matmul(W_hn0, hidden0, ntmp02);
    elemwise_add(ntmp02, b_hn0, ntmp03);
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
    elemwise_add(rtmp10, b_ir1, rtmp12);
    elemwise_add(rtmp12, rtmp11, rtmp13);
    elemwise_add(rtmp13, b_hr1, rtmp14);
    elemwise_sigmoid(rtmp14, r1);

    /* Second layer z */
    matmul(W_iz1, hidden0, ztmp10);
    matmul(W_hz1, hidden1, ztmp11);
    elemwise_add(ztmp10, b_iz1, ztmp12);
    elemwise_add(ztmp12, ztmp11, ztmp13);
    elemwise_add(ztmp13, b_hz1, ztmp14);
    elemwise_sigmoid(ztmp14, z1);

    /* Second layer n */
    matmul(W_in1, hidden0, ntmp10);
    elemwise_add(ntmp10, b_in1, ntmp11);
    matmul(W_hn1, hidden1, ntmp12);
    elemwise_add(ntmp12, b_hn1, ntmp13);
    elemwise_mul(r1, ntmp13, ntmp14);
    elemwise_add(ntmp11, ntmp14, ntmp15);
    elemwise_tanh(ntmp15, n1);

    /* Second layer h (hidden) */
    elemwise_oneminus(z1, htmp10);
    elemwise_mul(htmp10, n1, htmp11);
    elemwise_mul(z1, hidden1, htmp12);
    elemwise_add(htmp11, htmp12, hidden1); //hidden 1 = hid*batch

    /* Fully connected layer */
    matmul(W_fc, hidden1, ftmp0); // W_fc * hidden1 = numchar*batch = ftmp0
    elemwise_add(ftmp0, b_fc, f); //b_fc: numchar*batch , f = numchar*batch

    /* Softmax */
    softmax(f, char_prob); //softmax  char_prob = numchar*batch

    /* Random select */
    // 이걸 어떻게 행렬로 줄 수 있을까? 
    int selected_char = random_select(char_prob, rfloats, n * MAX_LEN + l);

    output[n * (MAX_LEN + 1) + l] = selected_char;
    input->buf[0] = selected_char;
    if (selected_char == EOS)
      break;
  }

  // mpi 송수신

  if (mpi_rank == 0) {
    for (int i=1; i < mpi_size; i++) {
      MPI_Recv(output+hostBegin[i]*MAX_LEN, (hostEnd[i]-hostBegin[i])*MAX_LEN, MPI_CHAR, i, 4, MPI_COMM_WORLD, &status);
    }
  } else {
    MPI_Send(output+hostBegin[mpi_rank]*MAX_LEN, (hostEnd[mpi_rank]-hostBegin[mpi_rank])*MAX_LEN, MPI_CHAR, 0, 4, MPI_COMM_WORLD);
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