#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define TILE_ROWS 16
#define TILE_COLS 32

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

__global__ void matrixMultiplyOptimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int tx = threadIdx.x;         // 0..TILE_COLS-1
    const int ty = threadIdx.y;         // 0..TILE_ROWS-1

    const int row = blockIdx.y * TILE_ROWS + ty;
    const int col = blockIdx.x * TILE_COLS + tx;

    // Shared memory tiles
    __shared__ float sharedA[TILE_ROWS][TILE_COLS];
    __shared__ float sharedB[TILE_COLS][TILE_COLS];

    float sum = 0.0f;

    const int numTiles = K / TILE_COLS;

    for (int t = 0; t < numTiles; t++)
    {
        int globalA_col = t * TILE_COLS + tx;
        sharedA[ty][tx] = A[row * K + globalA_col];

        const int LOADS_PER_THREAD = TILE_COLS / TILE_ROWS;

#pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; i++) {
            int brow = ty + i * TILE_ROWS;     // expand TY across extra rows
            int bcol = tx;

            int globalB_row = t * TILE_COLS + brow;
            sharedB[brow][bcol] = B[globalB_row * N + col];
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_COLS; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    // Write result
    C[row * N + col] = sum;
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

int main(int argc, char* argv[]) {
    int M, N, K;
    
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
    std::cout << "Matrix multiplication: C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]"
        << ", TILE_ROWS=" << TILE_ROWS << " TILE_COLS=" << TILE_COLS << std::endl;

    // Calculate matrix sizes in bytes (unpadded)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    int M_pad = ((M + TILE_ROWS - 1) / TILE_ROWS) * TILE_ROWS;
    int N_pad = ((N + TILE_COLS - 1) / TILE_COLS) * TILE_COLS;
    int K_pad = ((K + TILE_COLS - 1) / TILE_COLS) * TILE_COLS;
    
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
    dim3 blockDim(TILE_COLS, TILE_ROWS);
    dim3 gridDim((N_pad + TILE_COLS - 1) / TILE_COLS, (M_pad + TILE_ROWS - 1) / TILE_ROWS);

    auto launch_gpu_kernel = [&](const char* phase, int iteration) -> bool {
        matrixMultiplyOptimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << phase << " kernel failed at iteration " << iteration << ": "
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
        return true;
    };

    bool kernel_failed = false;
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        if (!launch_gpu_kernel("Warm-up", i)) {
            kernel_failed = true;
            break;
        }
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
    double gpu_total_time = 0.0;
    int successful_gpu_runs = 0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        if (!launch_gpu_kernel("GPU benchmark", i)) {
            kernel_failed = true;
            break;
        }
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
        ++successful_gpu_runs;
    }
    double gpu_avg_time = (successful_gpu_runs > 0) ? (gpu_total_time / successful_gpu_runs) : 0.0;
    
    // Verify the top-left M×N block matches the CPU result (outside timing loop)
    if (!kernel_failed && successful_gpu_runs > 0) {
        cudaMemcpy(h_C_gpu, d_C, size_C_padded, cudaMemcpyDeviceToHost);
    bool results_match = true;
    float tolerance = 1e-3f;
    for (int i = 0; i < M && results_match; ++i) {
        for (int j = 0; j < N; ++j) {
            float cpu_val = h_C_cpu[i * N + j];
            float gpu_val = h_C_gpu[i * N_pad + j];
            if (std::fabs(cpu_val - gpu_val) > tolerance) {
                std::cerr << "Mismatch at (" << i << "," << j << "): CPU=" << cpu_val << " GPU=" << gpu_val << std::endl;
                results_match = false;
                break;
            }
        }
    }
    if (results_match)
        std::cout << "CPU and GPU outputs match for the " << M << "×" << N << " block." << std::endl;
    } else if (kernel_failed) {
        std::cerr << "Skipping verification because GPU kernel failed.\n";
    }


    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    if (successful_gpu_runs > 0) {
        printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
        printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    } else {
        std::cerr << "GPU benchmark failed; no valid timing to report.\n";
    }
    
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
