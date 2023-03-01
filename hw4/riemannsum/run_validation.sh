#!/bin/bash

salloc -N 1 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 16 120


salloc -N 2 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 32 50


salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 5 1


salloc -N 4 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 9 5040


salloc -N 1 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 3 2202


salloc -N 2 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 15 1367


salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 26 325


salloc -N 4 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 31 64


salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 12 12345


salloc -N 4 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main -t 53 45126