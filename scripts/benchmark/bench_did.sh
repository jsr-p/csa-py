#!/usr/bin/env bash

set -e

fp=data/benchmark
mkdir -p $fp

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

for N in 1000 2500 5000; do
  for end in 15 30 45; do
    for delta in 3 6 9; do
      file="data/testing/sim_ub_${N}_${end}_${delta}.csv"
      echo "Running benchmark with file: $file"

      echo "Python Benchmark:"
	  python scripts/profile/bmarkfull.py --N "$N" --e "$end" --delta "$delta"

      echo "R Benchmark:"
	  Rscript scripts/profile/bmarkfull.R --N "$N" --e "$end" --delta "$delta"
      echo
    done
  done
done
