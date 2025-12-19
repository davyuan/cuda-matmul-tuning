#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iomanip>
#include <functional>
#include <random>
#include <numeric>

const uint BLOCKSIZE = 32;

#define CHECK_CUDA(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// New function for CUDA event-based timing
float time_kernel(std::function<void()> kernel_func) {
    cudaEvent_t start, stop;
    float elapsed_time;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_func();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}

// Function to perform warmup and benchmark runs
float benchmark_kernel(std::function<void()> kernel_func, int warmup_runs, int benchmark_runs) {
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_func();
    }
    
    // Benchmark runs
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        float time = time_kernel(kernel_func);
        times.push_back(time);
    }
    
    // Calculate average time
    float avg_time = std::accumulate(times.begin(), times.end(), 0.000f) / benchmark_runs;
    return avg_time;
}

int main(int argc, char* argv[]) {
    int M, N, K;
    const int warmup_runs = 3;
    const int benchmark_runs = 20;

    // ——— Option 1: Command-line arguments (recommended) ———
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);

        if (M <= 0 || N <= 0 || K <= 0) {
            std::cerr << "Error: Dimensions must be positive integers.\n";
            return -1;
        }
    }
    // ——— Option 2: Interactive input (fallback) ———
    else if (argc == 1) {
        std::cout << "Enter matrix dimensions M, N, K (e.g., 1024 1024 1024): ";
        if (!(std::cin >> M >> N >> K)) {
            std::cerr << "Error: Invalid input.\n";
            return -1;
        }
        if (M <= 0 || N <= 0 || K <= 0) {
            std::cerr << "Error: Dimensions must be positive.\n";
            return -1;
        }
    }
    // ——— Invalid usage ———
    else {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " <M> <N> <K>(command-line)\n";
        std::cerr << "  " << argv[0] << "                (interactive)\n";
        return -1;
    }

    // ——— Print dimensions ———
    std::cout << "Matrix multiplication: C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]" << ", BLOCKSIZE:" <<
        BLOCKSIZE << std::endl;

    // Calculate matrix sizes in bytes (unpadded)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    int gridX = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    int gridY = (M + BLOCKSIZE - 1) / BLOCKSIZE;

    // Declare device pointers
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Copy data to buffers
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    // Warm-up runs
    printf("Performing warm-up runs...\n");
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(gridX, gridY);
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    
    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    float gpu_avg_time = benchmark_kernel([&]() {
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    }, warmup_runs, benchmark_runs);
    std::cout << "CUDA kernel average time: " << gpu_avg_time << " ms" << std::endl;  
    
    // Verify the top-left M×N block matches the CPU result (outside timing loop)
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    bool results_match = true;
    float tolerance = 1e-3f;
    for (int i = 0; i < M && results_match; ++i) {
        for (int j = 0; j < N; ++j) {
            float cpu_val = h_C_cpu[i * N + j];
            float gpu_val = h_C_gpu[i * N + j];
            if (std::fabs(cpu_val - gpu_val) > tolerance) {
                std::cerr << "Mismatch at (" << i << "," << j << "): CPU=" << cpu_val << " GPU=" << gpu_val << std::endl;
                results_match = false;
                break;
            }
        }
    }
    if (results_match)
        std::cout << "CPU and GPU outputs match for the " << M << "×" << N << " block." << std::endl;


    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e3f));
    printf("Speedup: %fx\n", cpu_avg_time *1000 / gpu_avg_time);
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;

}
