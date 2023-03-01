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
  int binary[32 + 1] = {
    0,
  };

  if(x > 0){
      int i = 0;
      while (x > 0){
        binary[i] = x % 2;
        x = x/2;
        i++;
      }

      // reverse
      int start = 0;
      int end = 31;
      while(start < end){
        int temp = binary[start];
        binary[start] = binary[end];
        binary[end] = temp;
        start++;
        end--;
      }
  }

  else if(x == 0){
    for(int i=0; i<32; i++){
        binary[i] = 0;
    }
  }

  else {
      // add 1 b4 flipping
      int positive = -(x + 1);
      int i = 0;
      while (positive > 0){
        binary[i] = positive % 2;
        positive = positive / 2;
        i++;
      }

      // reverse
      int start = 0;
      int end = 31;
      while(start < end){
        int temp = binary[start];
        binary[start] = binary[end];
        binary[end] = temp;
        start++;
        end--;
      }

      // flip
      for(int i=0; i<32; i++){
        if(binary[i] == 0){
            binary[i] = 1;
        }
        else if(binary[i] == 1){
            binary[i] = 0;
        }
      }

  }

  // convert int array to char array
  for(int i=0; i<32; i++){
    output[i] = '0' + binary[i];
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
  int binary[64 + 1] = {
    0,
  };

  if(x > 0){
      int i = 0;
      while (x > 0){
        binary[i] = x % 2;
        x = x/2;
        i++;
      }

      // reverse
      int start = 0;
      int end = 63;
      while(start < end){
        int temp = binary[start];
        binary[start] = binary[end];
        binary[end] = temp;
        start++;
        end--;
      }
  }

  else if(x == 0){
    for(int i=0; i<64; i++){
        binary[i] = 0;
    }
  }

  else {
      // add 1 b4 flipping
      int positive = -(x + 1);
      int i = 0;
      while (positive > 0){
        binary[i] = positive % 2;
        positive = positive / 2;
        i++;
      }

      // reverse
      int start = 0;
      int end = 63;
      while(start < end){
        int temp = binary[start];
        binary[start] = binary[end];
        binary[end] = temp;
        start++;
        end--;
      }

      // flip
      for(int i=0; i<64; i++){
        if(binary[i] == 0){
            binary[i] = 1;
        }
        else if(binary[i] == 1){
            binary[i] = 0;
        }
      }

  }

  // convert int array to char array
  for(int i=0; i<64; i++){
    output[i] = '0' + binary[i];
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
  typedef union v32_union {
	float f;
	unsigned int u;
  } v32;

  v32 v; v.f = x;
  unsigned int mask = 1 << 31;
  int i = 0;
  do {
     output[i] = (v.u & mask ? '1' : '0');
     i++;
  } while (mask >>= 1);

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_double(double x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  typedef union v64_union {
	double f;
    unsigned long long u;
  } v64;

  v64 v; v.f = x;
  unsigned long long mask = 1ULL << 63;
  int i = 0;
  do {
     output[i] = (v.u & mask ? '1' : '0');
     i++;
  } while (mask >>= 1);
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
