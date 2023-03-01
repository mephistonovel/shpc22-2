TARGET=main
OBJECTS=util.o riemannsum.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp
LDLIBS=-lm -lmpi -lmpi_cxx -lpthread

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
