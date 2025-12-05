# SE3082 - Parallel Computing Assignment 03
## Matrix Multiplication using Parallel Computing

**Student:** SAMARAWEERA K D N  
**Student ID:** IT23279766  
**University:** Sri Lanka Institute of Information Technology

---

## Problem Domain
Numerical Computation and Scientific Computing

## Algorithm
Matrix Multiplication (1000 x 1000 matrices)

---

## Implementations

### 1. Serial (Baseline)
- Standard O(n³) matrix multiplication
- Execution time: ~1.92 seconds

### 2. OpenMP (Shared Memory)
- Parallelizes outer loop using `#pragma omp parallel for`
- Best result: 0.54 seconds with 8 threads (3.53x speedup)

### 3. MPI (Distributed Memory)
- Row-wise distribution using MPI_Scatter/MPI_Gather
- Best result: 0.84 seconds with 8 processes (2.28x speedup)

### 4. CUDA (GPU)
- Each thread computes one element of result matrix
- Best result: 0.022 seconds kernel time (85x speedup)
- Tested on Google Colab with Tesla T4 GPU

---

## Project Structure
```
SE3082_Assignment/
├── Serial/
│   └── matrix_serial.c
├── OpenMP/
│   └── matrix_openmp.c
├── MPI/
│   └── matrix_mpi.c
├── CUDA/
│   └── matrix_cuda.cu
├── Screenshots/
│   └── (execution screenshots)
└── Data/
    └── generate_graphs.py
```

---

## Compilation Instructions

### Serial
```bash
gcc -O2 -o matrix_serial matrix_serial.c -lm
./matrix_serial 1000
```

### OpenMP
```bash
gcc -O2 -fopenmp -o matrix_openmp matrix_openmp.c -lm
OMP_NUM_THREADS=8 ./matrix_openmp 1000
```

### MPI
```bash
mpicc -O2 -o matrix_mpi matrix_mpi.c -lm
mpirun -np 4 ./matrix_mpi 1000
```

### CUDA (Google Colab)
```bash
nvcc -arch=sm_75 -o matrix_cuda matrix_cuda.cu
./matrix_cuda 1000 16
```

---

## Results Summary

| Implementation | Best Time | Speedup |
|----------------|-----------|---------|
| Serial | 1.918s | 1.00x |
| OpenMP (8 threads) | 0.544s | 3.53x |
| MPI (8 processes) | 0.841s | 2.28x |
| CUDA (kernel) | 0.022s | 85.43x |

---

## Technologies Used
- C Programming Language
- OpenMP for shared memory parallelism
- MPICH for distributed memory parallelism
- NVIDIA CUDA for GPU parallelism
- Python (matplotlib) for graphs

---

## References
1. OpenMP API Specification v5.0
2. MPI Standard v3.1
3. NVIDIA CUDA Programming Guide v12.0
