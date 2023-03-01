#!/bin/bash

: ${NODES:=4}

salloc -N $NODES --partition shpc22 --exclusive      \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-63                         \
  ./main $@
