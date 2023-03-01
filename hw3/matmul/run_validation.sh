#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 26 831 538 2304
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 9 3305 1864 3494
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 38 618 3102 1695
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 30 1876 3453 3590
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 16 1228 2266 1552
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 2 3347 171 688
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 39 3583 962 765
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 30 2962 373 1957
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 9 3646 2740 3053
srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main -v -t 26 1949 3317 3868
