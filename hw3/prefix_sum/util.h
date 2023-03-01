#pragma once

void timer_init();

void timer_start(int i);

double timer_stop(int i);

void alloc_array(double **m, int N);

void rand_array(double *m, int N);

void zero_array(double *m, int N);

void copy_array(double *a, double *b, int N);

void print_vec(double *m, int N);

void print_mat(double *m, int M, int N);

void check_prefix_sum(const double *out, const double *in, int N);
