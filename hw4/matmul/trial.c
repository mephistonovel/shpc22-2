  MPI_Status status;
  MPI_Bcast((float *)&A[0], M*K, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((float *)&B[0], K*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int init = (M/mpi_world_size) * mpi_rank;
  int end = ((mpi_rank+1)*M)/mpi_world_size;
  int c_init = init*N;

  #pragma omp parallel for num_threads(threads_per_process)
    for (int i = init; i < end; ++i) {
      for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }

  int bufcount; 

  if (mpi_rank != mpi_world_size -1){
    bufcount = (M/mpi_world_size) * N;
  }
  else if (mpi_rank == mpi_world_size -1){
    bufcount = M*N-c_init;
  }
  
  if(mpi_rank != 0){
    MPI_Send(&C[c_init], bufcount, MPI_FLOAT, 0, mpi_rank, MPI_COMM_WORLD);
  }
  else{
    for(int i = 1; i < mpi_world_size; i++){
      MPI_Recv(&C[(M/mpi_world_size) * i * N], M*N, MPI_FLOAT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &status);
    }
  }
  