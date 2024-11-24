
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define G 10

// (6.67430 * (pow (10, -11)))
/**
 * Simulate n body problem
 *  
 */


typedef struct body { 
    double mass;
    double position[3];
    double velocity[3];
    double net_force[3];
} body_t;

__host__ __device__ body_t *n_bodies;

__host__ __device__ double (*forces) [3];


/**
 * Function to read input
 */



/**
 * Force calculation
 * force applied to index1 due to index2
 * (1,2)
 * column : body that exerting force
 * row : body that is "affected"by the force of the column
 */

__global__ void calculate_force ( int N){
    size_t index1 = threadIdx.x;
    size_t index2 = threadIdx.y;
    if (index1 == index2)
    {
        forces[index1 * N + index2][0] = 0;
        forces[index1 * N + index2][1] = 0;
        forces[index1 * N + index2][2] = 0;
        return;
    }
    // calculate magnitude
    body_t body1 = n_bodies[index1];
    body_t body2 = n_bodies[index2];
    
    double dx = body2.position[0] - body1.position[0];
    double dy = body2.position[1] - body1.position[1];
    double dz = body2.position[2] - body1.position[2];

    double distance = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));

    double F_mag = (G * body1.mass * body2.mass)/(pow(distance,2));

    double F_x = F_mag * (dx / distance);
    double F_y = F_mag * (dy / distance);
    double F_z = F_mag * (dz / distance);

    printf("body %d\n", index1);
    printf("F_x: %lf, F_y: %lf, F_z: %lf\n", F_x, F_y, F_z);

    forces[index1 * N + index2][0] = F_x;
    forces[index1 * N + index2][1] = F_y;
    forces[index1 * N + index2][2] = F_z;
}

__global__ void net_force (int N){

    size_t index = threadIdx.x;

    int start_index = index * N;
    double F_x =0.0;
    double F_y = 0.0;
    double F_z = 0.0;
    for (int i = start_index; i < start_index + N ; i++)
    {
        F_x += forces[i][0];
        F_y += forces[i][1];
        F_z += forces[i][2];
    }
    printf("net force body %d\n", index);
    printf("F_x: %lf, F_y: %lf, F_z: %lf\n", F_x, F_y, F_z);
    n_bodies[index].net_force[0] = F_x;
    n_bodies[index].net_force[1] = F_y;
    n_bodies[index].net_force[2] = F_z;

}
/**
 * Update funciton (euler's method)
 */

__global__  void update_body ( double time_step){
     size_t index = threadIdx.x;
    body_t body = n_bodies[index];
    double mass = body.mass;
    double f_X = body.net_force[0];
    double f_Y = body.net_force[1];
    double f_Z = body.net_force[2];

    double v_X = body.velocity[0];
    double v_Y = body.velocity[1];
    double v_Z = body.velocity[2];
  
    double a_X = f_X / mass;
    double a_Y = f_Y / mass;
    double a_Z = f_Z / mass;

// velocity
    n_bodies[index].velocity[0] = v_X + a_X * time_step;
    n_bodies[index].velocity[1] = v_Y + a_Y * time_step;
    n_bodies[index].velocity[2] = v_Z + a_Z * time_step;

// position

    n_bodies[index].position[0] = body.position[0] +  n_bodies[index].velocity[0] * time_step;
    n_bodies[index].position[1] = body.position[1] +  n_bodies[index].velocity[1] * time_step;
    n_bodies[index].position[2] = body.position[2] +  n_bodies[index].velocity[2] * time_step;

}

void print_object(int index){

    printf("Body %d\n", index);
    printf("Position:\n");
    for (int i = 0; i < 3; i++)
    {
        printf("%lf, ", n_bodies[index].position[i]);
    }
    printf("\n");
    printf("Velocity:\n");
    for (int i = 0; i < 3; i++)
    {
        printf("%lf, ", n_bodies[index].velocity[i]);
    }
    printf("\n");
    printf("Net Force:\n");
    for (int i = 0; i < 3; i++)
    {
        printf("%lf, ", n_bodies[index].net_force[i]);
    }
    printf("\n");

}

void simulate_n_body (double time_step, int N,  double (*forces) [3], body_t *n_bodies){

//allocate memory & copy to gpu
double(*gpu_force)[3];
    if (cudaMalloc(&gpu_force, sizeof(N * N* sizeof(double[3]))) != cudaSuccess){
        fprintf(stderr, "Failed to allocate memory for the force");
    }

    body_t *n_bodies_gpu;
    if (cudaMalloc(&n_bodies_gpu, sizeof(sizeof(body_t) * N)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for the bodies");
    }

    if(cudaMemcpy(gpu_force, forces, sizeof(N * N* sizeof(double[3])), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy force to the GPU\n");
  }

   if(cudaMemcpy(n_bodies_gpu, n_bodies, sizeof(sizeof(body_t) * N), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies to the GPU\n");
  } 


  // calculate
  calculate_force<<<N, N>>>(N);
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

//     net_force<<<1,N>>>(N);
// //      if(cudaDeviceSynchronize() != cudaSuccess) {
// //     fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
// //   }
//     update_body<<<1, N>>>(time_step);
//      if(cudaDeviceSynchronize() != cudaSuccess) {
//     fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
//   }



//copy back to CPU

    if(cudaMemcpy(forces, gpu_force, sizeof(N * N* sizeof(double[3])), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy forces from the GPU\n");
  }

 if(cudaMemcpy(n_bodies, n_bodies_gpu, sizeof(sizeof(body_t) * N), cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies from the GPU\n");
  }

  cudaFree(gpu_force);
cudaFree(n_bodies_gpu);


}




int main(int argc, char** argv){
    int N = 2;
    n_bodies = (body_t *)malloc(sizeof(body_t) * N);
    forces = (double(*)[3])malloc(N * N* sizeof(double[3]));
    body_t body1 = {.mass = 500, .position = {0, 0, 0}, .velocity = { 0, 0, 0 }};
    body_t body2 = {.mass = 100, .position = {4, 3, 0}, .velocity = { 0, 0, 0 }};

    n_bodies[0] = body1;
    n_bodies[1] = body2;

    double time_step = 0.01;

    simulate_n_body(time_step, N, forces, n_bodies);

    // calculate_force<<<N, N>>>(N);
    // calculate_force(0, 0, N);
    // calculate_force(1, 1, N);

    // calculate_force(0, 1, N);
    // calculate_force(1, 0, N);
   

    for (int i = 0; i < N * N; i++){
        printf("%d\n", i);
        for (int j = 0; j < 3; j++)
        {

            printf("%lf", i, forces[i][j]);
        }
        printf("\n");
    }

  

   
    // net_force(0, N);
    // net_force(1, N);
   // printf("==========afer net force ===================\n");
  //  print_object(0);
    //print_object(1);

    //printf("==========afer update ===================\n");

   // update_body<<<1, N>>>(time_step);
    //print_object(0);

    return 0;
}