#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "prefix_sum.h"
#include "util.h"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-pvh] [-m method] [-n num_iterations] N\n", prog_name);
  printf("Options:\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("     -m : method (sequential or parallel) (default: sequential)\n");
  printf("      N : number of components of array A . (default: 8)\n");
}

enum method {
  SEQUENTIAL,
  PARALLEL,
};

static enum method how = SEQUENTIAL;
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
      if (!strcmp(optarg, "sequential")) {
        how = SEQUENTIAL;
      } else if (!strcmp(optarg, "parallel")) {
        how = PARALLEL;
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
  printf("  METHOD: %s\n", (how == SEQUENTIAL) ? "sequential" : "parallel");
  printf("  Problem Size (N): %d\n", N);
  printf("  Number of Iterations: %d\n", num_iterations);
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... ");
  fflush(stdout);
  double *in, *out;

  // Initialize random seed
  timer_init();

  // Allocate vectors
  alloc_array(&in, N);
  alloc_array(&out, N);

  // Set each element to a random value
  rand_array(in, N);
  zero_array(out, N);

  printf("done!\n");

  // WARM-UP
  if (how == SEQUENTIAL) {
    prefix_sum_sequential(out, in, N);
  } else {
    prefix_sum_parallel(out, in, N);
  }
  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i);
    fflush(stdout);
    timer_start(0);
    if (how == SEQUENTIAL) {
      prefix_sum_sequential(out, in, N);
    } else {
      prefix_sum_parallel(out, in, N);
    }
    double elapsed_time = timer_stop(0);
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  check_prefix_sum(out, in, N);

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n", N / elapsed_time_avg / 1e9);

  return 0;
}
