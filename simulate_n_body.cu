
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

// __host__ __device__ body_t *n_bodies;

// __host__ __device__ double (*forces) [3];


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

__global__ void calculate_force ( int N, body_t *n_bodies, double (*forces) [3]){
    size_t index1 = threadIdx.x;
    size_t index2 = threadIdx.y;
    if (index1 == index2)
    {
        forces[index1 * N + index1][0] = 0;
        forces[index1 * N + index1][1] = 0;
        forces[index1 * N + index1][2] = 0;
        return;
    }

    // calculate magnitude
    body_t body1 = n_bodies[index1];
    body_t body2 = n_bodies[index2];

   if(body1.position[0] == body2.position[0] 
   && body1.position[1] == body2.position[1] 
   && body1.position[2] == body2.position[2] 
   ){
    printf("The initial position of two objects are same, (COLLISION) UNDEFINED BEHAVIOR.\nThe calculation result in invalid values \n");
   }

    double dx = body2.position[0] - body1.position[0];
    double dy = body2.position[1] - body1.position[1];
    double dz = body2.position[2] - body1.position[2];


    double distance = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));

    double F_mag = (G * body1.mass * body2.mass)/(pow(distance,2));

    double F_x = F_mag * (dx / distance);
    double F_y = F_mag * (dy / distance);
    double F_z = F_mag * (dz / distance);

    //printf("body %lu\n", index1);
  //  printf("F_x: %lf, F_y: %lf, F_z: %lf\n", F_x, F_y, F_z);

    forces[index1 * N + index2][0] = F_x;
    forces[index1 * N + index2][1] = F_y;
    forces[index1 * N + index2][2] = F_z;
}

__global__ void net_force (int N, body_t *n_bodies, double (*forces) [3]){

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
    // printf("net force body %lu\n", index);
    // printf("F_x: %lf, F_y: %lf, F_z: %lf\n", F_x, F_y, F_z);
    n_bodies[index].net_force[0] = F_x;
    n_bodies[index].net_force[1] = F_y;
    n_bodies[index].net_force[2] = F_z;

}
/**
 * Update funciton (euler's method)
 */

__global__  void update_body ( double time_step, body_t *n_bodies, double (*forces) [3]){
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

void print_object(int index, body_t *n_bodies, double (*forces) [3]){

    printf("Body %d\n", index);
    printf("Mass: %lf \n", n_bodies[index].mass);
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
__global__  void print_object_gpu(int index, body_t *n_bodies, double (*forces) [3]){

    printf("Body %d\n", index);
    printf("Mass: %lf \n", n_bodies[index].mass);
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


void simulate_n_body (double time_step, int N,  double (*forces) [3], body_t *n_bodies, int iter_num){

//allocate memory & copy to gpu
    double(*gpu_force)[3];
    body_t *n_bodies_gpu;

    if (cudaMalloc(&gpu_force, N * N* sizeof(double[3])) != cudaSuccess){
        fprintf(stderr, "Failed to allocate memory for the force");
    }


    if (cudaMalloc(&n_bodies_gpu, sizeof(body_t) * N) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for the bodies");
    }

    if(cudaMemcpy(gpu_force, forces,  N * N* sizeof(double[3]), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy force to the GPU\n");
  }

   if(cudaMemcpy(n_bodies_gpu, n_bodies, sizeof(body_t) * N, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies to the GPU\n");
  } 


  // calculate
for (int i = 0; i < iter_num; i++){
    printf("iter: %d\n", i);
    calculate_force<<<1, dim3(N, N)>>>(N, n_bodies_gpu, gpu_force);

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

    net_force<<<1,N>>>(N, n_bodies_gpu, gpu_force);
     if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }


    update_body<<<1, N>>>(time_step, n_bodies_gpu, gpu_force);
     if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

    for (int i = 0; i < N; i++){
        print_object_gpu<<<1,N>>>(i, n_bodies_gpu, gpu_force);

     }
        

}

//copy back to CPU

    if(cudaMemcpy(forces, gpu_force, N * N* sizeof(double[3]), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy forces from the GPU\n");
  }

 if(cudaMemcpy(n_bodies, n_bodies_gpu, sizeof(body_t) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy n bodies from the GPU\n");
  }


    // for (int i = 0; i < N * N; i++){
    //     printf("%d\n", i);
    //     for (int j = 0; j < 3; j++)
    //     {

    //         printf("%lf", i, forces[i][j]);
    //     }
    //     printf("\n");
    // }

  
  
  cudaFree(gpu_force);
cudaFree(n_bodies_gpu);


}




int main(int argc, char** argv){
    
    if (argc != 5) {

    fprintf(stderr, "Usage: %s <input file name> <N> <timestep> <duration>\n", argv[0]);

    exit(1);

  }

// Get the total number of objects, namely N
int N = atoi(argv[2]);
//printf("N: %d\n", N);

// Get bodies from the csv file and store it into the n_bodies array


  // Try to open the input file
 FILE* input = fopen(argv[1], "r");

  if (input == NULL) {

    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);

    perror(NULL);

    exit(2);

  }

// Read row by row and create body object
 body_t *n_bodies = (body_t *)malloc(sizeof(body_t) * N);

  char* line = NULL;

  size_t line_capacity = 0;

size_t counter = 0;
size_t line_num = 0;

  while (getline(&line, &line_capacity, input) > 0) {
    body_t body = {.mass = 0, .position = {0, 0, 0}, .velocity = { 0, 0, 0 }};

    char *token = strtok(line, ",");
   
    
    while (token != NULL)
    {
        
     switch(counter){
        case 0:
            body.mass = strtod(token, NULL);
            break;
        case 1:
            body.position[0] = strtod(token, NULL);
            break;
        case 2:
            body.position[1] = strtod(token, NULL);
            break;
        case 3:
            body.position[2] = strtod(token, NULL);
            break;
        case 4:
            body.velocity[0] = strtod(token, NULL);
            break;
        case 5:
            body.velocity[1] = strtod(token, NULL);
            break;
        case 6:
            body.velocity[2] = strtod(token, NULL);
            break;

     }//switch counter
       
      
        token = strtok(NULL, ",");
        counter++;
    }//while token
    counter = 0; //reset counter

    //add that body to the array at line_num position
    n_bodies[line_num] = body;
    line_num++;

}//while get line

// Make sure given N is equal to the number of rows in the file.
if(line_num!= N){
     fprintf(stderr, "The number of objects given as N is not equal to the number of rows in the file.\n");
     return 1;
}

    // get the time step 
    double time_step = strtod(argv[3], NULL);

   

    // get the duration i.e. how long we will simualte

    double duration = strtod(argv[4], NULL);

    // int N = 2;
    
   
     double (*forces) [3] = (double(*)[3])malloc(N * N* sizeof(double[3]));
    // body_t body1 = {.mass = 500, .position = {0, 0, 0}, .velocity = { 0, 0, 0 }};
    // body_t body2 = {.mass = 100, .position = {4, 3, 0}, .velocity = { 0, 0, 0 }};

    // n_bodies[0] = body1;
    // n_bodies[1] = body2;

    int iter_num = ceil(duration / time_step);

  
    simulate_n_body(time_step, N, forces, n_bodies, iter_num);
  

       

    // calculate_force<<<N, N>>>(N);
    // calculate_force(0, 0, N);
    // calculate_force(1, 1, N);

    // calculate_force(0, 1, N);
    // calculate_force(1, 0, N);



    // net_force(0, N);
    // net_force(1, N);
  //printf("==========afer net force ===================\n");

     for (int i = 0; i < N; i++){
        print_object(i, n_bodies, forces);

     }
        
     // print_object(1 ,n_bodies, forces);

     // printf("==========afer update ===================\n");

     // update_body<<<1, N>>>(time_step);
     // print_object(0);

     return 0;
}