#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

/**
 * The structure to store the information for each individual body
 */
typedef struct body {
  double position[3];  // Position in 3D space: [x, y, z]
  double mass;         // Mass of the body
  double velocity[3];  // Velocity in 3D space: [vx, vy, vz]
  double net_force[3]; // Net force acting on the body in 3D: [Fx, Fy, Fz]
} body_t;

/**
 * CUDA kernel to generate random bodies.
 *
 * @param N The number of bodies to generate.
 * @param n_bodies Pointer to the array of bodies in GPU memory.
 * @param seed Seed value for random number generation.
 * @param range_max Maximum range for position and velocity values.
 */
__global__ void generate_body(int N, body_t *n_bodies, size_t seed, int range_max) {
  size_t index = threadIdx.x;
  curandState state;
  curand_init(seed, index, 0, &state);

  // Store randome data to the body
  body_t body = {.mass = 10 + ((range_max - 10 + 1)) * curand_uniform(&state),
                 .position = {-range_max + (2 * range_max) * curand_uniform(&state),
                              -range_max + (2 * range_max) * curand_uniform(&state),
                              -range_max + (2 * range_max) * curand_uniform(&state)},
                 .velocity = {-range_max + (2 * range_max) * curand_uniform(&state),
                              -range_max + (2 * range_max) * curand_uniform(&state),
                              -range_max + (2 * range_max) * curand_uniform(&state)}};

  n_bodies[index] = body;
}


/**
 * Generates an array of random bodies on the GPU and copies them to the host memory.
 *
 * @param n_bodies Pointer to the array of bodies in host memory.
 * @param N The number of bodies to generate.
 */
void generate_n_body(body_t *n_bodies, int N) {
  body_t *n_bodies_gpu;   // Pointer to bodies array in GPU memory

  // Allocate GPU memory for the bodies
  if (cudaMalloc(&n_bodies_gpu, sizeof(body_t) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory for the bodies");
  }

  // Copy the initial state of bodies to GPU
  if (cudaMemcpy(n_bodies_gpu, n_bodies, sizeof(body_t) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies to the GPU\n");
  }

  // Launch the kernel function to generate random bodies
  generate_body<<<1, N>>>(N, n_bodies_gpu, time(NULL), 100000);

  // Synchronize the device to ensure all threads have completed  
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // Copy the generated bodies from GPU to CPU memory
  if (cudaMemcpy(n_bodies, n_bodies_gpu, sizeof(body_t) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy forces from the GPU\n");
  }

  // Free GPU memory
  cudaFree(n_bodies_gpu);
}

/**
 * Main function to generate n bodies and output their properties to a CSV file.
 */
int main(int argc, char **argv) {
  // Check the command line arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <N> \n", argv[0]);
    exit(1);
  }

  int N = atoi(argv[1]);    // Number of bodies

  // Create the output file for storing random bodies
  FILE *output_file = fopen("./random_data.csv", "w");
  if (!output_file) printf("Can't open file\n");

  // Allocate memory and generate random bodies
  body_t *n_bodies = (body_t *)malloc(N * sizeof(body_t));
  generate_n_body(n_bodies, N);

  // Write body data to the CSV file
  for (int i = 0; i < N; i++) {
    fprintf(output_file, "%lf,", n_bodies[i].mass);

    fprintf(output_file, "%lf,", n_bodies[i].position[0]);
    fprintf(output_file, "%lf,", n_bodies[i].position[1]);
    fprintf(output_file, "%lf,", n_bodies[i].position[2]);

    fprintf(output_file, "%lf,", n_bodies[i].velocity[0]);
    fprintf(output_file, "%lf,", n_bodies[i].velocity[1]);
    fprintf(output_file, "%lf", n_bodies[i].velocity[2]);

    fprintf(output_file, "\n");
  }

  // Close the output file
  fclose(output_file);
  
  // Free allocated memory
  free(n_bodies);
  return 0;
}