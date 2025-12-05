#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void init_matrix(double *mat, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++)
        mat[i] = (double)rand() / RAND_MAX * 10.0;
}

void multiply_local(double *A_local, double *B, double *C_local, int rows, int n) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A_local[i * n + k] * B[k * n + j];
            C_local[i * n + j] = sum;
        }
    }
}

double checksum(double *mat, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++)
        sum += mat[i];
    return sum;
}

int main(int argc, char *argv[]) {
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    
    // Make n divisible by nprocs
    if (n % nprocs != 0)
        n = ((n / nprocs) + 1) * nprocs;
    
    int local_rows = n / nprocs;
    
    if (rank == 0) {
        printf("MPI Matrix Multiplication\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Processes: %d\n", nprocs);
        printf("Rows per process: %d\n\n", local_rows);
    }
    
    double *A = NULL, *C = NULL;
    double *A_local = malloc(local_rows * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C_local = malloc(local_rows * n * sizeof(double));
    
    if (rank == 0) {
        A = malloc(n * n * sizeof(double));
        C = malloc(n * n * sizeof(double));
        init_matrix(A, n, 12345);
        init_matrix(B, n, 67890);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    
    // Distribute data
    MPI_Scatter(A, local_rows * n, MPI_DOUBLE, 
                A_local, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Each process computes its portion
    multiply_local(A_local, B, C_local, local_rows, n);
    
    // Gather results
    MPI_Gather(C_local, local_rows * n, MPI_DOUBLE,
               C, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    
    if (rank == 0) {
        printf("Running distributed multiplication...\n");
        printf("\nResults:\n");
        printf("Time: %.6f seconds\n", end - start);
        printf("Checksum: %.2f\n", checksum(C, n * n));
        printf("C[0][0] = %.2f, C[0][1] = %.2f\n", C[0], C[1]);
        free(A); free(C);
    }
    
    free(A_local); free(B); free(C_local);
    MPI_Finalize();
    return 0;
}