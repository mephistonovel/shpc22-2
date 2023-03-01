#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int threads_per_process, int mpi_rank, int mpi_world_size) {

  // TODO: FILL_IN_HERE
  float *buff_mat;
  int division, remainder,index;

  buff_mat = (float*)malloc(sizeof(float)*M*(N/(mpi_world_size-1)));
  MPI_Status status;
  
  if (mpi_rank == 0){
    for (int i=1;i<mpi_world_size;i++){
      // MPI_Recv(buff,1000000,MPI_CHAR,i,0,MPI_COMM_WORLD,&status);
      MPI_Recv(buff_mat,M*(N/(mpi_world_size-1)),MPI_FLOAT,i,0,MPI_COMM_WORLD,&status);
      division = M/(mpi_world_size-1);
      remainder = M%(mpi_world_size-1);

      index = division*(i-1);
      index += ((remainder<(i-1))? remainder : i-1);
      if (remainder>=1) division++;

      for (int j=0;j<division;j++){
        for(int k=0;k<K;k++){
          C[(index+j)*N+k]=buff_mat[j*N+k];
        }
      }
    }
  }
  else{
    division = M/(mpi_world_size-1);
    remainder = M%(mpi_world_size-1);

    index = (division*(mpi_rank-1))+((remainder<(mpi_rank-1))? remainder : mpi_rank-1);
    if (remainder>=mpi_rank) division++;

    for (int i = 0; i < division; ++i) {
      for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
          buff_mat[i * N + j] += A[(index+i) * K + k] * B[k * N + j];
        }
      }
    }

    MPI_Send(buff_mat,M*(N/(mpi_world_size-1)),MPI_FLOAT,0,0,MPI_COMM_WORLD);

  }

}
