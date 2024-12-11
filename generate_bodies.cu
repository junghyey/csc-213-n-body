
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>


typedef struct body { 
    double mass;
    double position[3];
    double velocity[3];
    double net_force[3];
} body_t;


__global__  void generate_body (int N, body_t *n_bodies, size_t seed, int range_max){
    size_t index = threadIdx.x;
    curandState state;
    curand_init(seed, index, 0, &state);

    body_t body = {.mass = 10 + ((range_max-10 +1)) * curand_uniform(&state),
                   .position = {-range_max + (2*range_max) * curand_uniform(&state),
                    -range_max + (2*range_max) * curand_uniform(&state), 
                    -range_max + (2*range_max) * curand_uniform(&state)},
                   .velocity = {-range_max + (2*range_max) * curand_uniform(&state), 
                   -range_max + (2*range_max) * curand_uniform(&state),
                    -range_max + (2*range_max) * curand_uniform(&state)}};

    n_bodies[index] = body;
}

void generate_n_body (body_t* n_bodies, int N){

//allocate memory & copy to gpu
    body_t *n_bodies_gpu;

    // body_t body_per_time[iter_num][N];


    if (cudaMalloc(&n_bodies_gpu, sizeof(body_t) * N) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for the bodies");
    }

 if(cudaMemcpy(n_bodies_gpu, n_bodies, sizeof(body_t) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies to the GPU\n");
  } 
  generate_body<<<1, N>>>(N, n_bodies_gpu, time(NULL), 100000);
   if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

   if(cudaMemcpy(n_bodies, n_bodies_gpu,  sizeof(body_t) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy forces from the GPU\n");
  }
  cudaFree(n_bodies_gpu);

}

int main(int argc, char** argv){
    if (argc != 2) {

    fprintf(stderr, "Usage: %s <N> \n", argv[0]);

    exit(1);

  }
int N = atoi(argv[1]);

FILE *output_file = fopen("./random_data.csv", "w");

if (!output_file)
    printf("Can't open file\n");

body_t *n_bodies = (body_t *)malloc(N * sizeof(body_t));

generate_n_body(n_bodies, N);
printf("create_file\n");

for (int i = 0; i < N; i++)
{

    fprintf(output_file, "%lf,", n_bodies[i].mass);

    fprintf(output_file, "%lf,", n_bodies[i].position[0]);
    fprintf(output_file, "%lf,", n_bodies[i].position[1]);
    fprintf(output_file, "%lf,", n_bodies[i].position[2]);

    fprintf(output_file, "%lf,", n_bodies[i].velocity[0]);
    fprintf(output_file, "%lf,", n_bodies[i].velocity[1]);
    fprintf(output_file, "%lf", n_bodies[i].velocity[2]);

    fprintf(output_file, "\n");
        }
    
fclose(output_file);

return 0;
}