#pragma once

void timer_init();

void timer_start(int i);

double timer_stop(int i);

void check_riemannsum(int num_intervals, double parallel_result);

// Assume f is a black-box operation
double f(double x);