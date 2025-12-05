import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder
os.makedirs('../Screenshots', exist_ok=True)

# YOUR ACTUAL RESULTS
SERIAL_TIME = 1.917798

# OpenMP results
openmp_threads = [1, 2, 4, 8, 16]
openmp_times = [1.858072, 1.014331, 0.847814, 0.543830, 0.977148]

# MPI results
mpi_processes = [1, 2, 4, 8]
mpi_times = [1.989548, 1.206136, 0.935267, 0.840791]

# CUDA results (kernel time)
cuda_blocks = ['8x8', '16x16', '32x32']
cuda_kernel_times = [0.022450, 0.022488, 0.023003]
cuda_total_times = [0.031707, 0.031989, 0.032484]

# Calculate speedups
openmp_speedup = [SERIAL_TIME / t for t in openmp_times]
mpi_speedup = [SERIAL_TIME / t for t in mpi_times]
cuda_speedup = [SERIAL_TIME / t for t in cuda_kernel_times]

# Graph 1: OpenMP Time
plt.figure(figsize=(10, 6))
plt.plot(openmp_threads, openmp_times, 'b-o', linewidth=2, markersize=10)
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title('OpenMP: Threads vs Execution Time')
plt.xticks(openmp_threads)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('../Screenshots/graph_openmp_time.png', dpi=150)
plt.close()
print("Created: graph_openmp_time.png")

# Graph 2: OpenMP Speedup
plt.figure(figsize=(10, 6))
plt.plot(openmp_threads, openmp_speedup, 'g-o', linewidth=2, markersize=10, label='Actual')
plt.plot(openmp_threads, openmp_threads, 'r--', linewidth=1, label='Ideal')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title('OpenMP: Threads vs Speedup')
plt.xticks(openmp_threads)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('../Screenshots/graph_openmp_speedup.png', dpi=150)
plt.close()
print("Created: graph_openmp_speedup.png")

# Graph 3: MPI Time
plt.figure(figsize=(10, 6))
plt.plot(mpi_processes, mpi_times, 'b-s', linewidth=2, markersize=10)
plt.xlabel('Number of Processes')
plt.ylabel('Execution Time (seconds)')
plt.title('MPI: Processes vs Execution Time')
plt.xticks(mpi_processes)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('../Screenshots/graph_mpi_time.png', dpi=150)
plt.close()
print("Created: graph_mpi_time.png")

# Graph 4: MPI Speedup
plt.figure(figsize=(10, 6))
plt.plot(mpi_processes, mpi_speedup, 'g-s', linewidth=2, markersize=10, label='Actual')
plt.plot(mpi_processes, mpi_processes, 'r--', linewidth=1, label='Ideal')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('MPI: Processes vs Speedup')
plt.xticks(mpi_processes)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('../Screenshots/graph_mpi_speedup.png', dpi=150)
plt.close()
print("Created: graph_mpi_speedup.png")

# Graph 5: CUDA Time
plt.figure(figsize=(10, 6))
x = np.arange(len(cuda_blocks))
width = 0.35
plt.bar(x - width/2, cuda_kernel_times, width, label='Kernel Only', color='steelblue')
plt.bar(x + width/2, cuda_total_times, width, label='Total (with transfer)', color='coral')
plt.xlabel('Block Size')
plt.ylabel('Time (seconds)')
plt.title('CUDA: Block Size vs Execution Time')
plt.xticks(x, cuda_blocks)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.savefig('../Screenshots/graph_cuda_time.png', dpi=150)
plt.close()
print("Created: graph_cuda_time.png")

# Graph 6: CUDA Speedup
plt.figure(figsize=(10, 6))
plt.bar(cuda_blocks, cuda_speedup, color='orange', edgecolor='black')
plt.xlabel('Block Size')
plt.ylabel('Speedup (vs Serial)')
plt.title('CUDA: Block Size vs Speedup')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.savefig('../Screenshots/graph_cuda_speedup.png', dpi=150)
plt.close()
print("Created: graph_cuda_speedup.png")

# Graph 7: Comparative - Time
plt.figure(figsize=(12, 6))
implementations = ['Serial', 'OpenMP\n(8 threads)', 'MPI\n(8 processes)', 'CUDA\n(kernel)']
best_times = [SERIAL_TIME, min(openmp_times), min(mpi_times), min(cuda_kernel_times)]
colors = ['gray', 'blue', 'green', 'orange']
bars = plt.bar(implementations, best_times, color=colors, edgecolor='black')
plt.ylabel('Execution Time (seconds)')
plt.title('Comparison: Best Execution Times')
for bar, t in zip(bars, best_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{t:.4f}s', ha='center', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.savefig('../Screenshots/graph_comparative_time.png', dpi=150)
plt.close()
print("Created: graph_comparative_time.png")

# Graph 8: Comparative - Speedup
plt.figure(figsize=(12, 6))
best_speedups = [1.0, max(openmp_speedup), max(mpi_speedup), max(cuda_speedup)]
bars = plt.bar(implementations, best_speedups, color=colors, edgecolor='black')
plt.ylabel('Speedup')
plt.title('Comparison: Best Speedups')
for bar, s in zip(bars, best_speedups):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{s:.2f}x', ha='center', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.savefig('../Screenshots/graph_comparative_speedup.png', dpi=150)
plt.close()
print("Created: graph_comparative_speedup.png")

print("\n" + "="*50)
print("All 8 graphs created in Screenshots folder!")
print("="*50)