CUDA_CC := nvcc
C_CC    := clang
CFLAGS  := -g -O3

all: generate_bodies simulate_n_body simulate_n_body_cpu

clean:
	rm -f generate_bodies simulate_n_body simulate_n_body_cpu
	rm -f *.o

generate_bodies: generate_bodies.cu Makefile
	$(CUDA_CC) $(CFLAGS) -o generate_bodies generate_bodies.cu

simulate_n_body: simulate_n_body.cu Makefile
	$(CUDA_CC) $(CFLAGS) -o simulate_n_body simulate_n_body.cu

simulate_n_body_cpu: simulate_n_body_ver1.c Makefile
	$(C_CC) $(CFLAGS) -o simulate_n_body_cpu simulate_n_body_cpu.c -lm



format:
	@echo "Reformatting source code."
	@clang-format -i --style=file $(wildcard *.c) $(wildcard *.h) $(wildcard *.cu)
	@echo "Done."
