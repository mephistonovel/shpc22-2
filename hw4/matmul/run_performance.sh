#!/bin/bash

salloc -N 4 --partition shpc22 --exclusive      \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main  -v -t 32 -n 10 8192 8192 4096
