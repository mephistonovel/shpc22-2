TARGET=main
OBJECTS=main.o util.o prefix_sum.o

CFLAGS=-O0 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp
LDFLAGS=-lm -pthread

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
