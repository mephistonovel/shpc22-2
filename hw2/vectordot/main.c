#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "vectordot.h"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvh] [-n num_iterations] [-m method] N\n", prog_name);
  printf("Options:\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("     -m : must be either 'naive' or 'fma' (default: naive)\n");
  printf("      N : number of components of vectors A and B. (default: 8)\n");
}

enum method {
  NAIVE,
  FMA,
};

static enum method how = NAIVE;
static int N = 8;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
    case 'n':
      num_iterations = atoi(optarg);
      break;
    case 'm':
      if (!strcmp(optarg, "naive")) {
        how = NAIVE;
      } else if (!strcmp(optarg, "fma")) {
        how = FMA;
      } else {
        print_help(argv[0]);
        exit(0);
      }
      break;
    case 'h':
    default:
      print_help(argv[0]);
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
    case 0:
      N = atoi(argv[i]);
      break;
    default:
      break;
    }
  }
  printf("Options:\n");
  printf("  Vector dot method: %s\n", (how == NAIVE) ? "Naive" : "FMA");
  printf("  Problem Size (N): %d\n", N);
  printf("  Number of Iterations: %d\n", num_iterations);
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... ");
  fflush(stdout);
  float *A, *B;

  // Initialize random seed
  timer_init();

  // Allocate vectors
  alloc_array(&A, N);
  alloc_array(&B, N);

  // Set each element to a random value
  rand_array(A, N);
  rand_array(B, N);

  printf("done!\n");

  float c = 0.f;
  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i);
    fflush(stdout);
    timer_start(0);
    if (how == NAIVE) {
      c = vectordot_naive(A, B, N);
    } else {
      c = vectordot_fma(A, B, N);
    }
    double elapsed_time = timer_stop(0);
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  check_vectordot(A, B, c, N);

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n", 2.0 * N / elapsed_time_avg / 1e9);

  return 0;
}
