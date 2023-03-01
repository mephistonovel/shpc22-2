// super super slow sgemm kernel by jinpyo
#define PADDINGX 16
#define PADDINGY 16
#define NUM_WORK_ITEM 32
#define VECTOR_WIDTH 8

__kernel void sgemm(__global float8 *A, __global float8 *B, __global float8 *C, int M, int N, int K) {
  const int row = get_local_id(0); // row index of C
  const int col = get_local_id(1); // column index of C
  // if (row >= M || col >= N) return; // boundary check

  //work item 개수: 32개 
  //벡터크기: VECTOR_WIDTH
  const int global_row = NUM_WORK_ITEM*get_group_id(0)+row;
  const int global_col = (NUM_WORK_ITEM/VECTOR_WIDTH)*get_group_id(1)+col;


  __local float8 tileA[NUM_WORK_ITEM][NUM_WORK_ITEM/VECTOR_WIDTH];
  __local float8 tileB[NUM_WORK_ITEM][NUM_WORK_ITEM/VECTOR_WIDTH];

  float8 inter_val = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const int num_tiles = K/NUM_WORK_ITEM;
  for (int t=0;t<num_tiles;t++){
    const int trow = NUM_WORK_ITEM*t+row;
    const int tcol = (NUM_WORK_ITEM/VECTOR_WIDTH)*t+col;
    tileA[row][col] = A[global_row*(K/VECTOR_WIDTH)+tcol];
    tileB[row][col] = B[trow*(N/VECTOR_WIDTH)+global_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    float8 vec_a, vec_b;
    float vala;
    for (int k=0; k<(NUM_WORK_ITEM/VECTOR_WIDTH);k++){
      vec_a = tileA[row][k];
      for (int w=0;w<VECTOR_WIDTH;w++){
        vec_b=tileB[VECTOR_WIDTH*k+w][col];

        switch(w) {
          case 0: vala = vec_a.s0; break;
          case 1: vala = vec_a.s1; break;
          case 2: vala = vec_a.s2; break;
          case 3: vala = vec_a.s3; break;
          case 4: vala = vec_a.s4; break;
          case 5: vala = vec_a.s5; break;
          case 6: vala = vec_a.s6; break;
          case 7: vala = vec_a.s7; break;
        }
        inter_val += vec_b * vala;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[global_row*(N/VECTOR_WIDTH)+global_col] = inter_val;
}

  //A, B, C가 global memory에 있음 즉, 칩 밖에 있음 거기에 계속 접근하면 속도가 떨어짐.
  //local memory에 배치시켜야? 

  // 각 행렬의 부분들을 하나의 workgroup에 넣어놔서(workgroup은 CU단위로 활동하니까) 데이터 재사용비율을 늘린다(?)
  //workgroup: kernel의 scheduling 단위... 32배수의 work_item들을 포함. 즉, 그룹내개수는 32의 배수들
  //근데 이걸 또 스레드당 몇 개를 더 줘서 하라?????

__kernel void paddingInsert(const int P, const int Q,
                               const __global float* input,
                               const int P_XL, const int Q_XL,
                               __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P_XL in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q_XL in blocks of PADDINGY

    // Check whether we are within bounds of the XL matrix
    if (tx < P_XL && ty < Q_XL) {

        // Copy the input or pad a zero
        float value;
        if (tx < P && ty < Q) {
            value = input[ty*P + tx];
        }
        else {
            value = 0.0f;
        }

        // Store the result
        output[ty*P_XL + tx] = value;
    }
}
__kernel void paddingRemoveZeroes(const int P_XL, const int Q_XL,
                                  const __global float* input,
                                  const int P, const int Q,
                                  __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q in blocks of PADDINGY


    // Only store the result if within P * Q bounds
    if (tx < P && ty < Q) {
        output[ty*P + tx] = input[ty*P_XL + tx];
    }
}