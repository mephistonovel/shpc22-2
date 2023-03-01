TARGET=main
OBJECTS=util.o vectordot.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -mno-avx512f
LDFLAGS=-lm -pthread

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
