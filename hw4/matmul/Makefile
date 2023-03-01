TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f
LDLIBS=-lm -lpthread -lmpi -lmpi_cxx

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
