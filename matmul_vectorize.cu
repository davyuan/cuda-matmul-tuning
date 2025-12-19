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

const uint TILE_SIZE = 64;
const uint MINI_TILE_SIZE = 4;

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

__global__ void matmul_vectorize(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int inner_row_A = ty * MINI_TILE_SIZE;
    int inner_col_A = tx * MINI_TILE_SIZE;
    int inner_row_B = ty * MINI_TILE_SIZE;
    int inner_col_B = tx * MINI_TILE_SIZE;
    int row = by * TILE_SIZE + inner_row_A;
    int col = bx * TILE_SIZE + inner_col_B; 

    // allocate thread-local cache for results in registerfile
    float threadResults[MINI_TILE_SIZE][MINI_TILE_SIZE] = {0.0};    
#pragma unroll
    for (int tile = 0; tile < K / TILE_SIZE; ++tile) {
        int global_offset_A = row * K + tile * TILE_SIZE + inner_col_A;
        int global_offset_B = (tile * TILE_SIZE + inner_row_B) * N + col;     

        for(int i = 0; i < MINI_TILE_SIZE; i++){
            for(int j = 0; j < MINI_TILE_SIZE; j+=4){
                reinterpret_cast<float4 *>(&sharedA[inner_row_A + i][inner_col_A + j])[0] = 
                    reinterpret_cast<float4 *>(&A[global_offset_A + j])[0];                

                reinterpret_cast<float4 *>(&sharedB[inner_row_B + i][inner_col_B + j])[0] = 
                    reinterpret_cast<float4 *>(&B[global_offset_B + j])[0];   
            }
            global_offset_A += K;
            global_offset_B += N;         
        }        
        __syncthreads();
        
        float fragA[MINI_TILE_SIZE];
        float fragB[MINI_TILE_SIZE];

#pragma unroll
        for(int k=0; k < TILE_SIZE; ++k) {
            for(int i=0; i < MINI_TILE_SIZE; i++) {
                fragA[i] = sharedA[inner_row_A + i][k];
            }
            for(int j=0; j < MINI_TILE_SIZE; j++) {
                fragB[j] = sharedB[k][inner_col_B + j];
            }
            for(int i=0; i < MINI_TILE_SIZE; ++i) {
                for(int j=0; j < MINI_TILE_SIZE; ++j) {
                    threadResults[i][j] += fragA[i] * fragB[j];
                }
            }
        } 
        __syncthreads();
    }

#pragma unroll    
    for (uint i = 0; i < MINI_TILE_SIZE; ++i) {
        for (uint j = 0; j < MINI_TILE_SIZE; j+=4) {
            reinterpret_cast<float4 *>(
                &C[(row + i) * N + col + j])[0] = reinterpret_cast<float4 *>(&threadResults[i][j])[0];            
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
    std::cout << "Matrix multiplication: C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]" << ", TILE_SIZE:" <<
        TILE_SIZE << " MINI_TILE_SIZE:" << MINI_TILE_SIZE << std::endl;

    // Calculate matrix sizes in bytes (unpadded)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    int M_pad = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    int N_pad = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    int K_pad = ((K + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    
    // Calculate padded matrix sizes
    size_t size_A_padded = (size_t)M_pad * K_pad * sizeof(float);
    size_t size_B_padded = (size_t)K_pad * N_pad * sizeof(float);
    size_t size_C_padded = (size_t)M_pad * N_pad * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C_cpu;
    float *h_A_padded, *h_B_padded, *h_C_gpu;

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
    dim3 blockDim(TILE_SIZE/MINI_TILE_SIZE, TILE_SIZE/MINI_TILE_SIZE);
    dim3 gridDim((N_pad + TILE_SIZE - 1) / TILE_SIZE, (M_pad + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < 1; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_vectorize<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
        cudaDeviceSynchronize();
    }
   
    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    float gpu_avg_time = benchmark_kernel([&]() {
        matmul_vectorize<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
        //sgemmVectorize<BM, BN, BK, TM, TN>
        //    <<<gridDim, blockDim>>>(M_pad, N_pad, K_pad, 1.0, d_A, d_B, 0.0, d_C);  
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

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;

}
