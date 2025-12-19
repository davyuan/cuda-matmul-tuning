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
#include <cassert>
#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
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
    const int PADDED_TO = 16;

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
    std::cout << "Matrix multiplication: C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]" << std::endl;

    // Calculate matrix sizes in bytes (unpadded)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    int M_pad = ((M + PADDED_TO - 1) / PADDED_TO) * PADDED_TO;
    int N_pad = ((N + PADDED_TO - 1) / PADDED_TO) * PADDED_TO;
    int K_pad = ((K + PADDED_TO - 1) / PADDED_TO) * PADDED_TO;
    
    // Calculate padded matrix sizes
    size_t size_A_padded = (size_t)M_pad * K_pad * sizeof(float);
    size_t size_B_padded = (size_t)K_pad * N_pad * sizeof(float);
    size_t size_C_padded = (size_t)M_pad * N_pad * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C_cpu;
    float *h_A_padded, *h_B_padded, *h_C_gpu;

    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_A_padded = (float*)malloc(M_pad * K_pad * sizeof(float));
    h_B_padded = (float*)malloc(K_pad * N_pad * sizeof(float));
    h_C_gpu = (float*)malloc(size_C_padded);
    memset(h_A_padded, 0, size_A_padded);
    memset(h_B_padded, 0, size_B_padded);
    
    // Allocate device memory for padded buffers
    cudaMalloc(&d_A, size_A_padded);
    cudaMalloc(&d_B, size_B_padded);
    cudaMalloc(&d_C, size_C_padded);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Copy data to padded buffers
    for (int i = 0; i < M; ++i) {
        memcpy(h_A_padded + (size_t)i * K_pad, h_A + (size_t)i * K, K * sizeof(float));
    }
    for (int i = 0; i < K; ++i) {
        memcpy(h_B_padded + (size_t)i * N_pad, h_B + (size_t)i * N, N * sizeof(float));
    }

    // Copy padded data to device
    cudaMemcpy(d_A, h_A_padded, size_A_padded, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_padded, size_B_padded, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C_padded);
    
    // Warm-up runs
    printf("Performing warm-up runs...\n");
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < 1; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_pad, M_pad, K_pad, &alpha, d_B, N_pad, d_A, K_pad, &beta, d_C, N_pad));
        cudaDeviceSynchronize();
    }    
    
    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    float gpu_avg_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_pad, M_pad, K_pad, &alpha, d_B, N_pad, d_A, K_pad, &beta, d_C, N_pad));
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
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e3f));
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_A_padded);
    free(h_B_padded);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    CHECK_CUBLAS(cublasDestroy(handle));

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;

}
