#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void init_matrix(double *mat, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++)
        mat[i] = (double)rand() / RAND_MAX * 10.0;
}

void multiply(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

double checksum(double *mat, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++)
        sum += mat[i];
    return sum;
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    
    printf("Serial Matrix Multiplication\n");
    printf("Matrix size: %d x %d\n\n", n, n);
    
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    init_matrix(A, n, 12345);
    init_matrix(B, n, 67890);
    
    printf("Running serial multiplication...\n");
    double start = get_time();
    multiply(A, B, C, n);
    double end = get_time();
    
    double time_taken = end - start;
    
    printf("\nResults:\n");
    printf("Time: %.6f seconds\n", time_taken);
    printf("Checksum: %.2f\n", checksum(C, n));
    printf("C[0][0] = %.2f, C[0][1] = %.2f\n", C[0], C[1]);
    
    free(A); free(B); free(C);
    return 0;
}