#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 1
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 2
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 4
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 8
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 16
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 4096
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 1048576
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 67108864
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 134217728
srun --nodes=1 --exclusive --partition=shpc22 ./main -n 1 -m parallel 268435456
