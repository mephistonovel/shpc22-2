#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc22 numactl --physcpubind 0-63 ./main $@
