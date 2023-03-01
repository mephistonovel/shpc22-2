#pragma once

double riemannsum(int num_intervals, int mpi_rank, int mpi_world_size,
                  int threads_per_process);