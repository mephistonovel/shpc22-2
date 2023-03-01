__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];


  int bx = blockIdx.x, by = blockIdx.y;
  int local_row = threadIdx.y, local_col = threadIdx.x;

  int row = by*TS + local_row;
  int col = bx*TS + local_col;

  if (row >= M +TS || col >= N +TS) {return;}

  float Pvalue = 0; // phase value;
  for (int ph = 0; ph < ceil(K/(float)TS); ++ph){ // phase
    
    int t_col=TS*ph+local_col;
    int t_loc = TS*ph+local_row;
    if((row<M) && (t_col < K))
      Asub[local_row][local_col] = A[row*K + t_col];
    else
      Asub[local_row][local_col] = 0;
    if((col<N) && (t_loc < K))
      Bsub[local_row][local_col] = B[t_loc*N + col];
    else
      Bsub[local_row][local_col] = 0;

    __syncthreads();

    for(int k = 0; k < TS; k++)
      Pvalue += Asub[local_row][k] * Bsub[k][local_col];

    __syncthreads();
  }
  
  if((row<M) && (col < N))
    C[row*N+col] = Pvalue;
  
  // int row=threadIdx.x, col=threadIdx.y;

  // int glob_row = TS * blockIdx.x + row;
  // int glob_col = (TS/width) * blockIdx.y + col;
  // if (row >= M || col >= N)
  //   return;
    
  // __shared__ float4 asub[TS][TS/width];
  // __shared__ float4 bsub[TS][TS/width];

  // float4 mediate_val = make_float4(0.0f,0.0f,0.0f,0.0f);

  // const int num_tiles = K/TS;

  // for (int t=0;t<num_tiles;t++){
  //   const int t_row = TS*t+row;
  //   const int t_col = (TS/width)*t +col;

  //   asub[row][col] = A[glob_row*(K/width)+t_col];
  //   bsub[row][col] = B[t_row*(N/width)+glob_col];

  //   __syncthreads();

  //   float4 veca,vecb;
  //   float vala;
  //   for (int k=0;k<TS/width;k++){
  //     veca=asub[row][k];
  //     for (int w=0;w<width;w++){
  //       vecb=bsub[width*k+w][col];

  //       switch(w) {
  //         case 0: vala = veca.x; break;
  //         case 1: vala = veca.y; break;
  //         case 2: vala = veca.z; break;
  //         case 3: vala = veca.w; break;
  //       }
  //       mediate_val.x += vecb.x*vala;
  //       mediate_val.y += vecb.y*vala;
  //       mediate_val.z += vecb.z*vala;
  //       mediate_val.w += vecb.w*vala;
  //     }
  //   }
  //   __syncthreads();
  // }
  // C[glob_row*(N/width)+glob_col] = mediate_val;

// __shared__ float subA[TS][TS];
// __shared__ float subB[TS][TS];

// int i = blockDim.y * blockIdx.y + threadIdx.y;
// int j = blockDim.x * blockIdx.x + threadIdx.x;
// // if (i >= M || j >= N)
// //   return;
// float sum = 0.0f;

// for (int t = 0; t < (K + TS - 1) / TS; t++){
//   if (i < M && threadIdx.x + t * TS < K){
//     subA[threadIdx.y][threadIdx.x] = A[i * K + threadIdx.x + t * TS];
//   }
//   else{
//     subA[threadIdx.y][threadIdx.x] = 0.0f;
//   }
//   if (j < N && threadIdx.y + t * TS < K){
//     subB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + t * TS) * N + j];
//   }
//   else{
//     subB[threadIdx.y][threadIdx.x] = 0.0f;
//   }
//   __syncthreads();
//   for (int k = 0; k < TS; k++){
//     sum += subA[threadIdx.y][k] * subB[k][threadIdx.x];
//   }
//   __syncthreads();
//   }

//   if (i < M && j < N){
//     C[i * N + j] = sum;
//   }

}


void matmul(Tensor *input1, Tensor *input2, Tensor *output) // 단어개수 : 256개, M : 1024,  K: 512, 1024, N : 64(batchsize)
{
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];
  // printf("mpi_rank :  %d, M_ : %ld, K_ : %ld, N_ : %ld ", mpi_rank, M_, K_, N_);
  for (int i = 0; i < num_devices; i++)
  {
    cudabegin[i] = M_/4 * i ;
    cudaend[i] = M_/4 * (i+1);
  }
  
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (cudaend[i] - cudabegin[i]) * K_ * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K_ * N_ * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (cudaend[i] - cudabegin[i]) * N_ * sizeof(float)));
  } 

  // make input1, 2, 3 to float


  for (int i = 0; i < num_devices; i++) { //a_d (global) <- input1 (cpu)
    CUDA_CALL(cudaMemcpy(a_d[i], input1->buf + cudabegin[i] * K_,
                         (cudaend[i] - cudabegin[i]) * K_ * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_d[i], input2->buf, K_ * N_ * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    // dim3 gridDim(ceil((Mend[i] - Mbegin[i]) /(float) TS), ceil(N/ (float) TS), 1);

    // dim3 gridDim((N_ - 1) / TS + 1, (cudaend[i] - cudabegin[i] - 1) / TS + 1, 1);
    // dim3 blockDim(TS, TS, 1);
    // dim3 gridDim((cudaend[i] - cudabegin[i] +TS- 1)/TS, (N_/width + TS/width-1)/(TS/width),1);
    // dim3 blockDim(TS, TS/width, 1);
  
    dim3 blockDim(TS, TS,1);
    dim3 gridDim((N_+TS-1)/TS,(cudaend[i] - cudabegin[i]+TS-1)/TS,1);

    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M_, N_, K_);
    // gridDim: block 개수
    // // blockDim: thread 개수
    // matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M_, N_, K_);

  }