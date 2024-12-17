#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/**
 * Customized gravitational constant, G, used for force calculations
 * (Real-life value for G should be 6.6743 × 10^-11 m^3 /kg^-1 s^-2)
 **/
#define G 10


/**
 * The structure to store the information for each individual body
 */
typedef struct body {
  double position[3];  // Position in 3D space: [x, y, z]
  double mass;         // Mass of the body
  double velocity[3];  // Velocity in 3D space: [vx, vy, vz]
  double net_force[3]; // Net force acting on the body in 3D: [Fx, Fy, Fz]
} body_t;

body_t *n_bodies; // Array of n bodies in the simulation

double (*forces)[3]; // 2D array to store forces between each pair of bodies


/**
 * Calculate the gravitational force applied on one body by another.
 *
 * @param index1 Index of the body experiencing the force.
 * @param index2 Index of the body exerting the force.
 * @param N      Total number of bodies in the simulation.
 */
void calculate_force(int index1, int index2, int N) {
  // If two bodies at the same position, no force exerted
  if (index1 == index2) {
    forces[index1 * N + index2][0] = 0;
    forces[index1 * N + index2][1] = 0;
    forces[index1 * N + index2][2] = 0;
    return;
  }

  // Get body data
  body_t body1 = n_bodies[index1];
  body_t body2 = n_bodies[index2];

  // Calculate distance components
  double dx = body2.position[0] - body1.position[0];
  double dy = body2.position[1] - body1.position[1];
  double dz = body2.position[2] - body1.position[2];

  // Calculate the distance between the bodies
  double distance = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));

  // Calculate the magnitude of the gravitational force
  double F_mag = (G * body1.mass * body2.mass) / (pow(distance, 2));

  // Calculate force components
  double F_x = F_mag * (dx / distance);
  double F_y = F_mag * (dy / distance);
  double F_z = F_mag * (dz / distance);

  // Store the calculated forces
  forces[index1 * N + index2][0] = F_x;
  forces[index1 * N + index2][1] = F_y;
  forces[index1 * N + index2][2] = F_z;
}


/**
 * Calculate the net force acting on a body from all other bodies.
 * 
 * @param index Index of the body to calculate the net force for.
 * @param N     Total number of bodies in the simulation.
 */
void net_force(int index, int N) {
  int start_index = index * N;
  double F_x = 0.0;
  double F_y = 0.0;
  double F_z = 0.0;
  
  // Calculate the sum of forces exerted on the body
  for (int i = start_index; i < start_index + N; i++) {
    F_x += forces[i][0];
    F_y += forces[i][1];
    F_z += forces[i][2];
  }

  // Store the net force in the body struct
  n_bodies[index].net_force[0] = F_x;
  n_bodies[index].net_force[1] = F_y;
  n_bodies[index].net_force[2] = F_z;
}

/**
 * Update the position and velocity of a body using Euler's method.
 * 
 * @param index     Index of the body to update.
 * @param time_step The timestep for the simulation.
 */
void update_body(int index, double time_step) {
  body_t body = n_bodies[index];
  double mass = body.mass;

  // Get force components
  double f_x = body.net_force[0];
  double f_y = body.net_force[1];
  double f_z = body.net_force[2];

  // Get velocity components
  double v_x = body.velocity[0];
  double v_y = body.velocity[1];
  double v_z = body.velocity[2];

  // Calculate acceleration components
  double a_x = f_x / mass;
  double a_y = f_y / mass;
  double a_z = f_z / mass;

  // Update velocity components
  n_bodies[index].velocity[0] = v_x + a_x * time_step;
  n_bodies[index].velocity[1] = v_y + a_y * time_step;
  n_bodies[index].velocity[2] = v_z + a_z * time_step;

  // Update position components
  n_bodies[index].position[0] = body.position[0] + n_bodies[index].velocity[0] * time_step;
  n_bodies[index].position[1] = body.position[1] + n_bodies[index].velocity[1] * time_step;
  n_bodies[index].position[2] = body.position[2] + n_bodies[index].velocity[2] * time_step;
}


/**
 * Print the details of a body, including its position, velocity, and net force.
 * 
 * @param index Index of the body to print.
 */
void print_object(int index) {
  printf("Body %d\n", index);

  printf("Position: ");
  for (int i = 0; i < 3; i++) {
    printf("%lf, ", n_bodies[index].position[i]);
  }
  printf("\n");

  printf("Velocity: ");
  for (int i = 0; i < 3; i++) {
    printf("%lf, ", n_bodies[index].velocity[i]);
  }
  printf("\n");

  printf("Net Force:");
  for (int i = 0; i < 3; i++) {
    printf("%lf, ", n_bodies[index].net_force[i]);
  }
  printf("\n\n");
}


/**
 * Main function to initialize the simulation, run iterations, and display results.
 */
int main(int argc, char **argv) {
  int N = 4; // Number of bodies
  double time_step = 0.1; // Simulation timestep

  // Allocate memory for bodies and force matrix
  n_bodies = (body_t *)malloc(sizeof(body_t) * N);
  forces = (double(*)[3])malloc(N * N * sizeof(double[3]));

  // Initialize bodies
  body_t body1 = {.mass = 500, .position = {0, 0, 0}, .velocity = {0, 0, 0}};
  body_t body2 = {.mass = 100, .position = {4, 3, 0}, .velocity = {0, 0, 0}};
  body_t body3 = {.mass = 100, .position = {3, 3, 3}, .velocity = {0, 0, 0}};
  body_t body4 = {.mass = 300, .position = {1, 0, 0}, .velocity = {0, 0, 0}};

  n_bodies[0] = body1;
  n_bodies[1] = body2;
  n_bodies[2] = body3;
  n_bodies[3] = body4;
  
  // Simulation loop
  for (int i = 0; i < 10; i++) {
    // Calculate pairwise forces
    calculate_force(0, 0, N);
    calculate_force(1, 1, N);
    calculate_force(2, 2, N);
    calculate_force(3, 3, N);
    calculate_force(0, 1, N);
    calculate_force(1, 0, N);
    calculate_force(0, 2, N);
    calculate_force(2, 0, N);
    calculate_force(1, 2, N);
    calculate_force(2, 1, N);
    calculate_force(0, 3, N);
    calculate_force(3, 0, N);
    calculate_force(1, 3, N);
    calculate_force(3, 1, N);
    calculate_force(2, 3, N);
    calculate_force(3, 2, N);

    // Calculate net forces
    net_force(0, N);
    net_force(1, N);
    net_force(2, N);
    net_force(3, N);

    // Update bodies
    update_body(0, time_step);
    update_body(1, time_step);
    update_body(2, time_step);
    update_body(3, time_step);

    printf("====================== Time: %lf second ======================\n", (i+1) * time_step);

    // Print object data
    print_object(0);
    print_object(1);
    print_object(2);
    print_object(3);
    printf("\n");
  }

  // Free allocated memory
  free(n_bodies);
  free(forces);

  return 0;
}