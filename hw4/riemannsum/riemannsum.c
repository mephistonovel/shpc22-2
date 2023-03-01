#include <mpi.h>

#include "riemannsum.h"
#include "util.h"

double riemannsum(int num_intervals, int mpi_rank, int mpi_world_size,
                  int threads_per_process) {
  double pi = 0;
  double h = 1.0 / (double)num_intervals;
  
 

  // TODO: Parallelize the code using mpi_world_size processes (1 process per
  // node.
  // In total, (mpi_world_size * threads_per_process) threads will collaborate
  // to compute the Riemann sum.
  
  // double local_pi;
  // int n_per_process= num_intervals/mpi_world_size ;
  MPI_Bcast(&num_intervals,1,MPI_INT,0,MPI_COMM_WORLD);

  double local_pi;


  #pragma omp parallel for reduction (+: local_pi) num_threads(threads_per_process)
  for (int i = mpi_rank+1; i <= num_intervals; i+=mpi_world_size) {
    double x = h * ((double)i - 0.5);
    local_pi += h * f(x);
  }
  
  MPI_Reduce(&local_pi,&pi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  // Rank 0 should return the estimated PI value
  // Other processes can return any value (don't care)
  return pi;
}