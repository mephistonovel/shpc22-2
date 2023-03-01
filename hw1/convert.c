#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fallback_print_usage() {
  printf("Usage: ./convert [int|long|float|double] number\n");
  printf("Example: ./convert float 3.14\n");
  exit(0);
}

void print_int(int x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int nu=x;
   if (nu==0){
    for (int k=0;k<=31;k++){
      output[k] = 0+'0';
    }
  }
  else{
    //양수
    if (nu > 0){
      int dn=0;
      int i = 31; //숫자 넣는 거 시작

      while (nu != 1){
        dn = nu%2; //나머지
        nu = nu/2; //몫
        output[i] = dn+'0';
        i--;
      }
      output[i] = nu+'0';
      i--;

      for (int r=i;r>=0;r--){
        output[r] = 0+'0';
      }
    }
    //음수
    else{
      nu = -x;
      int dn=0;
      int i = 31; //숫자 넣는 거 시작

      while (nu != 1){
        dn = nu%2; //나머지
        nu = nu/2; //몫
        output[i] = (1-dn)+'0'; // 1의 보수
        i--;
      }
      output[i] = (1-nu)+'0'; //1의 보수
      i--;

      for (int r=i;r>=0;r--){
        output[r] = 1+'0';
      }

      int j = 31; // 2의 보수를 만들기 위한 위치index
      for (j=j;j>=0;j--){

        if (output[j]=='0'){
          output[j] = 1+'0';
          break;}
        else{
          output[j] = 0+'0';
        }

      }

    }
  }
  
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_long(long x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int nu=x;
  //0
  if (nu==0){
    for (int k=0;k<=31;k++){
      output[k] = 0+'0';
    }
  }

  else{
    //양수
    if (nu > 0){
      int dn=0;
      int i = 63; //숫자 넣는 거 시작

      while (nu != 1){
        dn = nu%2; //나머지
        nu = nu/2; //몫
        output[i] = dn+'0';
        i--;
      }
      output[i] = nu+'0';
      i--;

      for (int r=i;r>=0;r--){
        output[r] = 0+'0';
      }
    }
    //음수
    else{
      nu = -x;
      int dn=0;
      int i = 63; //숫자 넣는 거 시작

      while (nu != 1){
        dn = nu%2; //나머지
        nu = nu/2; //몫
        output[i] = (1-dn)+'0'; // 1의 보수
        i--;
      }
      output[i] = (1-nu)+'0'; //1의 보수
      i--;

      for (int r=i;r>=0;r--){
        output[r] = 1+'0';
      }

      int j = 63; // 2의 보수를 만들기 위한 위치index
      for (j=j;j>=0;j--){

        if (output[j]=='0'){
          output[j] = 1+'0';
          break;}
        else{
          output[j] = 0+'0';
        }

      }

    }
  }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_float(float x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  float nu = x;
  
  for (int i=31;i>=0;i--){
    int buffer = *(int*)&nu >> i & 1;
    if (buffer ==1){output[31-i]= '1';}
    else{output[31-i]='0';}
  }

  printf("%s\n", output);
}

void print_double(double x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  double nu = x;

  for (int i=63;i>=0;i--){
    int buffer = (*(unsigned long long*)&nu >> i) & 1ULL;
    if (buffer ==1){output[63-i] = '1';}
    else{output[63-i] ='0';}
  }
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

int main(int argc, char **argv) {
  if (argc != 3)
    fallback_print_usage();
  if (strcmp(argv[1], "int") == 0) {
    print_int(atoi(argv[2]));
  } else if (strcmp(argv[1], "long") == 0) {
    print_long(atol(argv[2]));
  } else if (strcmp(argv[1], "float") == 0) {
    print_float(atof(argv[2]));
  } else if (strcmp(argv[1], "double") == 0) {
    print_double(atof(argv[2]));
  } else {
    fallback_print_usage();
  }
}
