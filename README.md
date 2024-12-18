# Highly Parallelized N-Body Simulation

**Course Project**: This is project is developed as the final project for CSC-213: Operating System.

**Authors**: Hyeyun Jung & Lydia Ye

## Overview

This project simulates gravitational interactions between bodies in 3D space using CUDA for parallel processing. It includes:

1. **`simulate_n_body.cu`**: Parallelized simulation of N-Body interactions based on CUDA.
2. **`generate_bodies.cu`**: Generates random data for N bodies.
3. **`visualization.py`**: Visualizes simulation results.
4. **`simulate_n_body_cpu.c`**: CPU-based simulation of N-Body interactions (N = 4).

## Related Concepts

- Parallelism with GPU: Use CUDA to accelerate pairwise force calculations across N bodies.
- Thread Synchronization: Share intermediate results on GPU’s shared memory.
- File System: Get input data from and output results to CSV files.

## Requirements

- NVIDIA GPU with CUDA
- CUDA Toolkit
- Python 3.6+
- Libraries: `matplotlib`, `numpy`

## Usage

#### Build and clean the programs

Use the provided `Makefile` to compile the programs:

```bash
make
```

This compiles the following executables: `generate_bodies`, `simulate_n_body`, `simulate_n_body_cpu`.

To clean up generated files:

```bash
make clean
```

#### Generate random bodies

Run the `generate_bodies` program to create initial random data:

```bash
./generate_bodies <N>
```

Replace `<N>` with the number of bodies you want to simulate. The program outputs a CSV file `random_data.csv` with the generated data.

#### Simulate N-body interactions (GPU)

Run the `simulate_n_body` program:

```bash
./simulate_n_body <input filename> <N> <timestep> <duration>
```

- `<input filename>`: Path to the CSV file containing initial body data.
- `<N>`: Number of bodies
- `<timestep>`: Length of each timestep in seconds (e.g., `0.01`).
- `<duration>`: Duration of simulation in seconds (e.g., `10`).

The program generates an output CSV file `output.csv` with the simulation results.

#### Simulate N-body interactions (CPU)

Run the `simulate_n_body_cpu` program:

```bash
./simulate_n_body_cpu
```

This runs a CPU-based version of the simulation for comparison.

#### Visualize N-body simulation

Run the `visualization.py` script:

```bash
python visualization.py
```

The program reads data from a local `output.csv` file, displays the animation to show how the position of the bodies changes throught the time steps and creates `visualization.gif` to store the animation.

## Example Usage

1. Generates random data for 100 bodies and saves it in `random_data.csv`.

```bash
./generate_bodies 100
```

2. Runs the CUDA simuation for the interactions between 100 bodies with a timestep of 0.1 second and the duration of 10 seconds, using the input data from `random_data.csv`.

```bash
./simulate_n_body random_data.csv 100 0.1 10
```

3. Runs the CPU implementation of the simulation for 4 bodies.

```bash
./simulate_n_body_cpu
```

4. Visualizes N-body simulation output from the GPU version.

```bash
python visualization.py
```

_The gif creation code is commented out for this version_

## References

1. Chapter 31. Fast N-Body Simulation with CUDA. (n.d.). NVIDIA Developer. https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda

2. Computation and astrophysics of the N-body problem. (n.d.). In Lecture 3 (p. 2). https://www.maths.ed.ac.uk/~heggie/lecture3.pdf

3. Wikipedia contributors. (2024, November 21). N-body problem.https://en.wikipedia.org/wiki/N-body_problem

4. CSC213 Lab codes

5. Professor Charlie Curtsinger & Mentor Session

6. visualization.py code source : https://github.com/arpmay/N-body-Problem/blob/main/main.py
