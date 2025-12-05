
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(double *A, double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(double *mat, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++)
        mat[i] = (double)rand() / RAND_MAX * 10.0;
}

double checksum(double *mat, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++)
        sum += mat[i];
    return sum;
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    int block_size = (argc > 2) ? atoi(argv[2]) : 16;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("CUDA Matrix Multiplication\n");
    printf("GPU: %s\n", prop.name);
    printf("Matrix size: %d x %d\n", n, n);
    printf("Block size: %d x %d\n\n", block_size, block_size);
    
    size_t bytes = n * n * sizeof(double);
    
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);
    
    init_matrix(h_A, n, 12345);
    init_matrix(h_B, n, 67890);
    
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    printf("Copying data to GPU...\n");
    cudaEvent_t start_copy, end_copy;
    cudaEventCreate(&start_copy);
    cudaEventCreate(&end_copy);
    
    cudaEventRecord(start_copy);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(end_copy);
    cudaEventSynchronize(end_copy);
    
    float copy_to_time;
    cudaEventElapsedTime(&copy_to_time, start_copy, end_copy);
    
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    printf("Running kernel...\n");
    
    cudaEvent_t start_kernel, end_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    
    cudaEventRecord(start_kernel);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(end_kernel);
    cudaEventSynchronize(end_kernel);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    
    cudaEvent_t start_back, end_back;
    cudaEventCreate(&start_back);
    cudaEventCreate(&end_back);
    
    cudaEventRecord(start_back);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(end_back);
    cudaEventSynchronize(end_back);
    
    float copy_back_time;
    cudaEventElapsedTime(&copy_back_time, start_back, end_back);
    
    float kernel_sec = kernel_time / 1000.0f;
    float copy_to_sec = copy_to_time / 1000.0f;
    float copy_back_sec = copy_back_time / 1000.0f;
    float total_sec = kernel_sec + copy_to_sec + copy_back_sec;
    
    printf("\nResults:\n");
    printf("Kernel time: %.6f seconds\n", kernel_sec);
    printf("Copy to GPU: %.6f seconds\n", copy_to_sec);
    printf("Copy from GPU: %.6f seconds\n", copy_back_sec);
    printf("Total time: %.6f seconds\n", total_sec);
    printf("Checksum: %.2f\n", checksum(h_C, n));
    printf("C[0][0] = %.2f, C[0][1] = %.2f\n", h_C[0], h_C[1]);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
