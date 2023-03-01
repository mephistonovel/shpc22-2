#!/bin/bash

for i in 1 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 
# for i in 32 64
do
    srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t $i -n 5 4096 4096 4096
done