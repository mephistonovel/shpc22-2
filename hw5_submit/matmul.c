#define _GNU_SOURCE
#include "matmul.h"
#include "util.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#define TS 32
#define WIDTH 8
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define TSM 32                      // The tile-size in dimension M
#define TSN 32                      // The tile-size in dimension N
#define TSK 32    

#define PADDINGX 32
#define PADDINGY 32

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);              \
    exit(EXIT_FAILURE);                                                        \
  }

static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel,kernel2a,kernel2b,kernel3;
static cl_mem a_d, b_d, c_d;
static cl_mem a_xl, b_xl, c_xl;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // Write to GPU; A (cpu) -> a_d (gpu), B (cpu) -> b_d (gpu)
  if (K%TS==0 && M%TS==0 && N%TS==0){
    err = clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, M * K * sizeof(float), A,
                              0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, K * N * sizeof(float), B,
                              0, NULL, NULL);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);


    // Setup kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &K);
    CHECK_ERROR(err);

    // Setup global work size and local work size
    size_t gws[2] = {M, N/WIDTH};
    size_t lws[2] = {TS, TS/WIDTH};
    for (int i = 0; i < 2; ++i) {
      // By OpenCL spec, global work size should be MULTIPLE of local work size
      // e.g., gws = 25, lws = 16, then (25 + 16 - 1) / 16 * 16 = 40 / 16 * 16 = 2
      // * 16 = 64
      gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
    }

    // Run kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);

    // Read from GPU; c_d (gpu) -> C (cpu)
    err = clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, M * N * sizeof(float), C, 0,
                              NULL, NULL);
    CHECK_ERROR(err);

    // DO NOT REMOVE; NEEDED FOR TIME MEASURE
    err = clFinish(queue);
    CHECK_ERROR(err);}
  else{
    int K_XL = CEIL_DIV(K,TSK)*TSK;
    int M_XL = CEIL_DIV(M,TSM)*TSM;
    int N_XL = CEIL_DIV(N,TSN)*TSN;
    err = clFinish(queue);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, M * K * sizeof(float), A,
                              0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, K * N * sizeof(float), B,
                              0, NULL, NULL);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);

    //A행렬
    err = clSetKernelArg(kernel2a, 0, sizeof(int), (void*)&M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2a, 1, sizeof(int), (void*)&K);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2a, 2, sizeof(cl_mem), (void*)&a_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2a, 3, sizeof(int), (void*)&M_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2a, 4, sizeof(int), (void*)&K_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2a, 5, sizeof(cl_mem), (void*)&a_xl);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel2b, 0, sizeof(int), (void*)&K);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2b, 1, sizeof(int), (void*)&N);
    CHECK_ERROR(err);    
    err = clSetKernelArg(kernel2b, 2, sizeof(cl_mem), (void*)&b_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2b, 3, sizeof(int), (void*)&K_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2b, 4, sizeof(int), (void*)&N_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel2b, 5, sizeof(cl_mem), (void*)&b_xl);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_xl);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_xl);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_xl);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), (void*)&M_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), (void*)&N_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), (void*)&K_XL);
    CHECK_ERROR(err);

    
//c행렬
    err = clSetKernelArg(kernel3, 0, sizeof(int), (void*)&M_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel3, 1, sizeof(int), (void*)&N_XL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void*)&c_xl);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel3, 3, sizeof(int), (void*)&M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel3, 4, sizeof(int), (void*)&N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel3, 5, sizeof(cl_mem), (void*)&c_d);
    CHECK_ERROR(err);

    size_t pLocal[2] = {PADDINGX, PADDINGY};
    size_t pAGlobal[2] = {M_XL,K_XL};
    size_t pBGlobal[2] = {K_XL,N_XL};
    size_t pCGlobal[2] = {M,N};

    size_t lws[2] = {TS, TS/WIDTH};
    size_t gws[2] = {M_XL,N_XL/WIDTH};
    for (int i = 0; i < 2; ++i) {
      // By OpenCL spec, global work size should be MULTIPLE of local work size
      // e.g., gws = 25, lws = 16, then (25 + 16 - 1) / 16 * 16 = 40 / 16 * 16 = 2
      // * 16 = 64
      gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
      pCGlobal[i] = (pCGlobal[i]+pLocal[i]-1)/pLocal[i] * pLocal[i];
      
    }

    err = clEnqueueNDRangeKernel(queue, kernel2a, 2, NULL, pAGlobal, pLocal, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel2b, 2, NULL, pBGlobal, pLocal, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, kernel3, 2, NULL, pCGlobal, pLocal, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue,c_d,CL_TRUE,0,M*N*sizeof(float),C,0,NULL,NULL);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);

  }
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

static cl_program create_and_build_program_with_source(cl_context context,
                                                       cl_device_id device,
                                                       const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char *)malloc(source_size + 1);
  size_t ntotal = 0;
  while (ntotal < source_size) {
    int nread = fread(source_code, sizeof(char), source_size, file);
    ntotal += nread;
  }
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                      NULL, &log_size));
    char *log = (char *)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void matmul_initialize(int M, int N, int K) {

  int K_XL = CEIL_DIV(K,TSK)*TSK;
  int M_XL = CEIL_DIV(M,TSM)*TSM;
  int N_XL = CEIL_DIV(N,TSN)*TSN;
  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device (only 1)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl");

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "sgemm", &err);
  CHECK_ERROR(err);

  kernel2a = clCreateKernel(program,"paddingInsert",&err);
  CHECK_ERROR(err);
  kernel2b = clCreateKernel(program,"paddingInsert",&err);
  CHECK_ERROR(err);
  
  kernel3 = clCreateKernel(program,"paddingRemoveZeroes",&err);
  CHECK_ERROR(err);

  // Create GPU buffers
  if (K%TS==0 && M%TS==0 && N%TS==0){
    a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(float), NULL,
                        &err);
    CHECK_ERROR(err);
    b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL,
                        &err);
    CHECK_ERROR(err);
    c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL,
                        &err);
    CHECK_ERROR(err);}
  else{
    a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(float), NULL,
                        &err);
    CHECK_ERROR(err);
    b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL,
                        &err);
    CHECK_ERROR(err);
    c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL,
                        &err);
    CHECK_ERROR(err);
    a_xl = clCreateBuffer(context,CL_MEM_READ_WRITE,M_XL*K_XL*sizeof(float),NULL,&err);
    CHECK_ERROR(err);
    b_xl = clCreateBuffer(context,CL_MEM_READ_WRITE,K_XL*N_XL*sizeof(float),NULL,&err); 
    CHECK_ERROR(err);
    c_xl = clCreateBuffer(context,CL_MEM_READ_WRITE,M_XL*N_XL*sizeof(float),NULL,&err);
    CHECK_ERROR(err);
  }
}

void matmul_finalize() {
    clReleaseMemObject(a_d);
    clReleaseMemObject(b_d);
    clReleaseMemObject(c_d);
    clReleaseMemObject(a_xl);
    clReleaseMemObject(b_xl);
    clReleaseMemObject(c_xl);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseKernel(kernel2a);
    clReleaseKernel(kernel2b);
    clReleaseKernel(kernel3);
}
