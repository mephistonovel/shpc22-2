#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc22 ./main $@
