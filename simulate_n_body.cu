/*******************************************************************************
 *                                                                             *
 *                           CSC213 Project:                                   *
 *                    CUDA N-Body Gravitational Simulation                     *
 *                                                                             *
 *  Authors : Hyeyun Jung & Lydia Ye                                           *
 *  Date    : December 16, 2024                                                *
 *                                                                             *
 *******************************************************************************
 *                                                                             *
 *  Description:                                                               *
 *  This program simulates gravitational interactions between N                *
 *  bodies using CUDA parallel computing architecture.                         *
 *                                                                             *
 *  Key Components:                                                            *
 *  - Force Calculation : Newton's law of universal gravitation                *
 *  - Integration       : Euler method for position and velocity updates       *
 *  - Parallelization   : CUDA-based computation for performance optimization  *
 *                                                                             *
 *****************************************************************************
 *                                                                             *
 *  Usage:                                                                     *
 *  ./simulate_n_body <input_file> <N> <timestep> <duration>                   *
 *                                                                             *
 *  Input :                                                                    *
 *  - input_file : CSV file with initial configuraitons for body               *
 *  - N          : Number of bodies in simulation                              *
 *  - timestep   : Time step for integration (ex: 0.1, 1...)                   *
 *  - duration   : Total simulation time                                       *
 *                                                                             *
 *****************************************************************************
 *                                                                             *
 *  File Formats:                                                              *
 *  Input (CSV)         : mass,px,py,pz,vx,vy,vz                               *
 *  Output (output.csv) : Body,Mass,Px,Py,Pz,Vx,Vy,Vz,Time                     *
 *                                                                             *
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G 0.1  // Gravitational constant

/**
 * body_t: Structure representing a celestial body
 * @mass: Mass
 * @position: [x,y,z] coordinates
 * @velocity: [vx,vy,vz] velocity components
 * @net_force: [Fx,Fy,Fz] net force components
 * The units are not included because G is not a standard G
 * but made up G. So, user can set any unit as long as they are
 * consistent
 */
typedef struct body {
  double mass;
  double position[3];
  double velocity[3];
  double net_force[3];
} body_t;

/**
 * calculate_force: CUDA kernel computing gravitational forces
 * @N: Number of bodies
 * @n_bodies: Array of body objects
 * @forces: 2D array storing pairwise forces
 *
 * Computes F = G*m1*m2/r^2 between each pair
 */
__global__ void calculate_force(int N, body_t *n_bodies, double (*forces)[3]) {
  // Thread indices representing body pairs
  size_t index1 = threadIdx.x;
  size_t index2 = threadIdx.y;

  // Boundary Check
  if (index1 >= N || index2 >= N) return;

  // NO self-force calculation (diagonal in force matrix)
  if (index1 == index2) {
    forces[index1 * N + index1][0] = 0;
    forces[index1 * N + index1][1] = 0;
    forces[index1 * N + index1][2] = 0;
    return;
  }  // if index==index2

  // Body data to calculate
  body_t body1 = n_bodies[index1];
  body_t body2 = n_bodies[index2];

  // Handle "collision" where two objects' initial positions are same
  if (body1.position[0] == body2.position[0] &&
      body1.position[1] == body2.position[1] &&
      body1.position[2] == body2.position[2]) {
    printf(
        "The initial position of two objects are same, (COLLISION) UNDEFINED "
        "BEHAVIOR.\nThe calculation result in invalid values \n");
  }  // if

  // Calculate distance components for each axes
  double dx = body2.position[0] - body1.position[0];
  double dy = body2.position[1] - body1.position[1];
  double dz = body2.position[2] - body1.position[2];

  // Calculate total distance
  double distance = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));

  // Calculate gravitational force magnitude
  double F_mag = (G * body1.mass * body2.mass) / (pow(distance, 2));

  // Calculate force components (for each x, y, z) and store in forces array
  forces[index1 * N + index2][0] = F_mag * (dx / distance);  // x
  forces[index1 * N + index2][1] = F_mag * (dy / distance);  // y
  forces[index1 * N + index2][2] = F_mag * (dz / distance);  // z

}  // calculate force

/**
 * net_force: Computes total force on each body
 * @N: Number of bodies
 * @n_bodies: Body array
 * @forces: Force components array
 *
 * Sums all pairwise forces to get net force per body
 */
__global__ void net_force(int N, body_t *n_bodies, double (*forces)[3]) {
  // Thread index & index in the 2D array for current body
  size_t index = threadIdx.x;
  int start_index = index * N;

  // Initialize force components
  double F_x = 0.0;
  double F_y = 0.0;
  double F_z = 0.0;

  // Sum all forces acting on this body
  for (int i = start_index; i < start_index + N; i++) {
    F_x += forces[i][0];
    F_y += forces[i][1];
    F_z += forces[i][2];
  }  // for i

  // Update force in body structure
  n_bodies[index].net_force[0] = F_x;
  n_bodies[index].net_force[1] = F_y;
  n_bodies[index].net_force[2] = F_z;
}  // net_force

/**
 * update_body: Updates positions and velocities using Euler's method
 * @time_step: Integration time step (s)
 * @n_bodies: Current body array to update
 * @forces: Force array to use for update
 * @body_per_time: History of body states (per time step)
 * @current_iter: Current timestep (ex: is it 1st or 2nd iteration..etc..)
 * @N: Number of bodies
 */
__global__ void update_body(double time_step, body_t *n_bodies,
                            double (*forces)[3], body_t *body_per_time,
                            int current_iter, int N) {
  // Get current body
  size_t index = threadIdx.x;
  body_t body = n_bodies[index];

  // Generate "temp" to store current body's information
  double mass = body.mass;
  double f_X = body.net_force[0];
  double f_Y = body.net_force[1];
  double f_Z = body.net_force[2];

  double v_X = body.velocity[0];
  double v_Y = body.velocity[1];
  double v_Z = body.velocity[2];

  // Calculate acceleration F= ma
  double a_X = f_X / mass;
  double a_Y = f_Y / mass;
  double a_Z = f_Z / mass;

  // Update positions and velocities
  for (int i = 0; i < 3; i++) {
    n_bodies[index].position[i] += body.velocity[i] * time_step;
    n_bodies[index].velocity[i] += (body.net_force[i] / body.mass) * time_step;
  }

  // Store current iteration's updated body
  body_per_time[current_iter * N + index] = n_bodies[index];
}  // update force

/**
 * Prints detailed information about a body in the N-body simulation
 * @param index: Index of the body to print
 * @param n_bodies: Array of all bodies in simulation
 * @param forces: Array of forces between bodies (unused but kept for
 * consistency)
 * @param N: Total number of bodies in simulation
 *
 * Prints:
 * - Body ID (0,1,...N-1)
 * - Mass
 * - Position vector [x,y,z]
 * - Velocity vector [vx,vy,vz]
 * - Net force vector [Fx,Fy,Fz]
 */
void print_object(int index, body_t *n_bodies, double (*forces)[3], int N) {
  printf("Body %d\n", index % N);                // Body ID
  printf("Mass: %lf \n", n_bodies[index].mass);  // Mass

  // Position
  printf("Position:\n");
  for (int i = 0; i < 3; i++) {
    printf("%lf, ", n_bodies[index].position[i]);
  }
  printf("\n");

  // Velocity
  printf("Velocity:\n");
  for (int i = 0; i < 3; i++) {
    printf("%lf, ", n_bodies[index].velocity[i]);
  }

  printf("\n");

  // Force
  printf("Net Force:\n");
  for (int i = 0; i < 3; i++) {
    printf("%lf, ", n_bodies[index].net_force[i]);
  }

  printf("\n");
}  // print object

/**
 * simulate_n_body: Main simulation driver
 * @time_step: Integration time step (s)
 * @N: Number of bodies
 * @forces: Force array (initialized from main)
 * @n_bodies: Body array to use for calculation (initialized from main)
 * @iter_num: Number of iterations (duration/timestep)
 * @body_per_time_cpu:  History of body states (per time step)
 *
 * Manages GPU memory and runs simulation loop
 */
void simulate_n_body(double time_step, int N, double (*forces)[3],
                     body_t *n_bodies, int iter_num,
                     body_t *body_per_time_cpu) {
  // GPU memory allocation
  double(*gpu_force)[3];
  body_t *body_per_time_gpu;
  body_t *n_bodies_gpu;

  if (cudaMalloc(&gpu_force, N * N * sizeof(double[3])) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory for the force");
  }  // if gpu_force
  if (cudaMalloc(&body_per_time_gpu, iter_num * sizeof(body_t) * N) !=
      cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory for the force");
  }  // if body_per_time_gpu
  if (cudaMalloc(&n_bodies_gpu, sizeof(body_t) * N) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory for the bodies");
  }  // if n_bodies_gpu

  // Copy data to GPU
  if (cudaMemcpy(gpu_force, forces, N * N * sizeof(double[3]),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy force to the GPU\n");
  }  // if gpu_force
  if (cudaMemcpy(body_per_time_gpu, body_per_time_cpu,
                 iter_num * sizeof(body_t) * N,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy force to the GPU\n");
  }  // if body_per_time_gpu
  if (cudaMemcpy(n_bodies_gpu, n_bodies, sizeof(body_t) * N,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies to the GPU\n");
  }  // if n_bodies_gpu

  // Main simulation loop
  for (int i = 0; i < iter_num; i++) {  // for each time step
    // Calculate forces between all pairs
    calculate_force<<<1, dim3(N, N)>>>(N, n_bodies_gpu, gpu_force);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s\n",
              cudaGetErrorString(cudaPeekAtLastError()));
    }  // if

    // Calculate net forces on each body
    net_force<<<1, N>>>(N, n_bodies_gpu, gpu_force);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s\n",
              cudaGetErrorString(cudaPeekAtLastError()));
    }  // if

    // Update body based on calculation
    update_body<<<1, N>>>(time_step, n_bodies_gpu, gpu_force, body_per_time_gpu,
                          i, N);

    if (cudaDeviceSynchronize() != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s\n",
              cudaGetErrorString(cudaPeekAtLastError()));
    }  // if
  }  // fori i

  // copy back to CPU
  if (cudaMemcpy(forces, gpu_force, N * N * sizeof(double[3]),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy forces from the GPU\n");
  }  // if forces
  if (cudaMemcpy(body_per_time_cpu, body_per_time_gpu,
                 iter_num * sizeof(body_t) * N,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy forces from the GPU\n");
  }  // if body_per_time_cpu
  if (cudaMemcpy(n_bodies, n_bodies_gpu, sizeof(body_t) * N,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies from the GPU\n");
  }  // if n_bodies

  // Free Memory
  cudaFree(gpu_force);
  cudaFree(n_bodies_gpu);
  cudaFree(body_per_time_gpu);
}  // simulate_n_body

/*========================================
 * Main Function
 ========================================*/

int main(int argc, char **argv) {
  /*----------------------------------------
   * Process Command Line & Validate Input
   *----------------------------------------*/
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <input file name> <N> <timestep> <duration>\n",
            argv[0]);
    exit(1);
  }

  // Parse N and validate
  int N = atoi(argv[2]);
  if (N <= 0) {
    fprintf(stderr, "Error: N must be positive\n");
    exit(1);
  }

  // Try to open the file & check
  FILE *input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);
    perror(NULL);
    exit(2);
  }

  /*----------------------------------------
   * Simulation Setup Section
   *----------------------------------------*/

  // Generate Bodies from CSV

  // Allocate memory to store bodies
  body_t *n_bodies = (body_t *)malloc(sizeof(body_t) * N);
  if (!n_bodies) {
    fprintf(stderr, "Failed to allocate memory for bodies\n");
    exit(1);
  }
  // Parse CSV file line by line
  char *line = NULL;
  size_t line_capacity = 0;
  size_t counter = 0;
  size_t line_num = 0;

  // The csv is given as mass,px,py,pz,vx,vy,vz
  while (getline(&line, &line_capacity, input) > 0) {
    body_t body = {.mass = 0, .position = {0, 0, 0}, .velocity = {0, 0, 0}};
    char *token = strtok(line, ",");

    while (token != NULL) {
      switch (counter) {
        case 0:  // mass
          body.mass = strtod(token, NULL);
          break;
        case 1:  // px
          body.position[0] = strtod(token, NULL);
          break;
        case 2:  // py
          body.position[1] = strtod(token, NULL);
          break;
        case 3:  // pz
          body.position[2] = strtod(token, NULL);
          break;
        case 4:  // vx
          body.velocity[0] = strtod(token, NULL);
          break;
        case 5:  // vy
          body.velocity[1] = strtod(token, NULL);
          break;
        case 6:  // vz
          body.velocity[2] = strtod(token, NULL);
          break;
      }  // switch counter
      token = strtok(NULL, ",");
      counter++;  // move to next "," (i.e. component)
    }  // while token

    counter = 0;  // reset counter

    // Add that body to n_bodies at [line_num] position
    n_bodies[line_num] = body;
    line_num++;  // Process next body
  }  // while get line

  // Sanity Check: Make sure given N is equal to the number of rows in the file.
  if (line_num != N) {
    fprintf(stderr,
            "The number of objects given as N is not equal to the number of "
            "rows in the file.\n");
    return 1;
  }  // line_num

  // Get time step
  double time_step = strtod(argv[3], NULL);
  // Get duration i.e. how long we will simualte
  double duration = strtod(argv[4], NULL);
  // Get how many times we should run
  int iter_num = ceil(duration / time_step);

  // Create arrays for force & history array to store bodies for each time
  double(*forces)[3] = (double(*)[3])malloc(N * N * sizeof(double[3]));
  body_t *body_per_time_cpu = (body_t *)malloc(iter_num * sizeof(body_t) * N);

  /*----------------------------------------
   * Run Simulation & Create Output File
   *----------------------------------------*/

  // Create outputfile & check
  FILE *output_file = fopen("./output.csv", "w");
  if (!output_file) printf("Can't open file\n");

  // Record initial configuration at time = 0
  fprintf(output_file, "Body,Mass,Px,Py,Pz,Vx,Vy,Vz,Time\n");
  for (int i = 0; i < N; i++) {
    fprintf(output_file, "%d,", i);
    fprintf(output_file, "%lf,", n_bodies[i].mass);

    fprintf(output_file, "%lf,", n_bodies[i].position[0]);
    fprintf(output_file, "%lf,", n_bodies[i].position[1]);
    fprintf(output_file, "%lf,", n_bodies[i].position[2]);

    fprintf(output_file, "%lf,", n_bodies[i].velocity[0]);
    fprintf(output_file, "%lf,", n_bodies[i].velocity[1]);
    fprintf(output_file, "%lf,", n_bodies[i].velocity[2]);

    fprintf(output_file, "0.0");
    fprintf(output_file, "\n");
  }  // for int i

  // Run simulation
  simulate_n_body(time_step, N, forces, n_bodies, iter_num, body_per_time_cpu);

  // Write simulation result
  for (int j = 0; j < iter_num; j++) {
    for (int i = 0; i < N; i++) {
      int index = j * N + i;  // 2D array represented as 1D

      // Body ID
      fprintf(output_file, "%d,", index % N);

      // Mass
      fprintf(output_file, "%lf,", body_per_time_cpu[index].mass);

      // Position
      fprintf(output_file, "%lf,", body_per_time_cpu[index].position[0]);
      fprintf(output_file, "%lf,", body_per_time_cpu[index].position[1]);
      fprintf(output_file, "%lf,", body_per_time_cpu[index].position[2]);

      // Velocity
      fprintf(output_file, "%lf,", body_per_time_cpu[index].velocity[0]);
      fprintf(output_file, "%lf,", body_per_time_cpu[index].velocity[1]);
      fprintf(output_file, "%lf,", body_per_time_cpu[index].velocity[2]);

      // Time
      fprintf(output_file, "%lf", (j + 1) * time_step);

      fprintf(output_file, "\n");
    }  // for i
  }  // for j

  // Close the file
  fclose(output_file);
  return 0;
}  // end of main