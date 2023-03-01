#!/bin/bash

salloc -N 4 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 26 831 538 2304

salloc -N 2 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 9 3305 1864 3494

salloc -N 1 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 16 618 3102 1695

salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 30 1876 3453 3590

salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 16 1228 2266 1552

salloc -N 2 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 2 3347 171 688

salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 8 3583 962 765

salloc -N 1 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 30 2962 373 1957

salloc -N 4 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 9 3646 2740 3053

salloc -N 3 --partition shpc22 --exclusive           \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -v -t 26 1949 3317 3868
