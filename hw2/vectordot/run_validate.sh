#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc22 ./main -m naive -n 1 1
srun --nodes=1 --exclusive --partition=shpc22 ./main -m naive -n 1 3
srun --nodes=1 --exclusive --partition=shpc22 ./main -m naive -n 1 11
srun --nodes=1 --exclusive --partition=shpc22 ./main -m naive -n 1 4097
srun --nodes=1 --exclusive --partition=shpc22 ./main -m naive -n 1 1000000
srun --nodes=1 --exclusive --partition=shpc22 ./main -m fma -n 1 1
srun --nodes=1 --exclusive --partition=shpc22 ./main -m fma -n 1 3
srun --nodes=1 --exclusive --partition=shpc22 ./main -m fma -n 1 11
srun --nodes=1 --exclusive --partition=shpc22 ./main -m fma -n 1 4097
srun --nodes=1 --exclusive --partition=shpc22 ./main -m fma -n 1 1000000
