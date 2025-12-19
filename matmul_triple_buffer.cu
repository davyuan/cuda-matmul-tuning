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
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>

namespace cg = cooperative_groups;

const uint TILE_COLS_A= 32;
const uint TILE_ROWS_A= 64;
const uint TILE_COLS_B = 64;
const uint TILE_ROWS_B = 32;
const uint MINI_TILE_ROWS = 8;
const uint MINI_TILE_COLS = 4;

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

/*__device__ __forceinline__ void issue_load(
    float* A, float* B, float* dstA_tile, float* dstB_tile, 
    int tile_idx, cuda::barrier<cuda::thread_scope::thread_scope_block>& barrier, 
    int row, int col, int inner_row_A, int inner_col_A, int inner_row_B, int inner_col_B, int K, int N){

    int global_offset_A = row * K + tile_idx * TILE_COLS_A + inner_col_A;
    int global_offset_B = (tile_idx * TILE_ROWS_B + inner_row_B) * N + col;     

#pragma unroll
    for(int i = 0; i < MINI_TILE_ROWS; i++){
        for(int j = 0; j < MINI_TILE_COLS; j+=4){
            if(inner_col_A + j + 3 < TILE_COLS_A){
                cuda::memcpy_async((float4*)(dstA_tile + (inner_row_A + i) * TILE_COLS_A + inner_col_A + j), 
                    (float4*)(A + global_offset_A + j), 16, barrier);             
            }

            if(inner_row_B + i < TILE_ROWS_B){
                cuda::memcpy_async((float4*)(dstB_tile + (inner_row_B + i) * TILE_COLS_B + inner_col_B + j), 
                    (float4*)(B + global_offset_B + j), 16, barrier);
            }
        }
        global_offset_A += K;
        global_offset_B += N;         
    }     
}*/

__device__ __forceinline__ void issue_load(
    float* A, float* B, float* dstA_tile, float* dstB_tile, 
    int tile_idx, cuda::barrier<cuda::thread_scope::thread_scope_block>& barrier, 
    int tid, int stride, int K, int N) {

    const int v_K = K / 4;
    const int v_N = N / 4;
    const int v_T_COLS_A = TILE_COLS_A / 4;
    const int v_T_COLS_B = TILE_COLS_B / 4;

    float4* srcA = (float4*)&A[blockIdx.y * TILE_ROWS_A * K + tile_idx * TILE_COLS_A];
    float4* srcB = (float4*)&B[tile_idx * TILE_ROWS_B * N + blockIdx.x * TILE_COLS_B];

    #pragma unroll
    for (int i = tid; i < (TILE_ROWS_A * v_T_COLS_A); i += stride) {
        int row_in_tile = i / v_T_COLS_A; 
        int col_in_tile = i % v_T_COLS_A;

        float4* current_srcA = srcA + (row_in_tile * v_K) + col_in_tile;
        cuda::memcpy_async((float4*)dstA_tile + i, current_srcA, 16, barrier);
    }

    #pragma unroll
    for (int i = tid; i < (TILE_ROWS_B * v_T_COLS_B); i += stride) {
        int row_in_tile = i / v_T_COLS_B;
        int col_in_tile = i % v_T_COLS_B;

        float4* current_srcB = srcB + (row_in_tile * v_N) + col_in_tile;
        cuda::memcpy_async((float4*)dstB_tile + i, current_srcB, 16, barrier);
    }
}

__device__ void compute_tile(
    const float* __restrict__ sA_tile, // Pointer to the current tile in sA
    const float* __restrict__ sB_tile, // Pointer to the current tile in sB
    float threadResults[MINI_TILE_ROWS][MINI_TILE_COLS]) 
{
    float fragA[MINI_TILE_ROWS];
    float fragB[MINI_TILE_COLS];

    // Thread local offsets within the tile
    int inner_row_A = threadIdx.y * MINI_TILE_ROWS;
    int inner_col_B = threadIdx.x * MINI_TILE_COLS;
    const int strideA = TILE_COLS_A; 
    const int strideB = TILE_COLS_B;

#pragma unroll
    for (int k = 0; k < TILE_COLS_A; ++k) {
        for (int i = 0; i < MINI_TILE_ROWS; ++i) {
            fragA[i] = sA_tile[(inner_row_A + i) * strideA + k];
        }
        for (int j = 0; j < MINI_TILE_COLS; ++j) {
            fragB[j] = sB_tile[k * strideB + (inner_col_B + j)];
        }
        for (int i = 0; i < MINI_TILE_ROWS; ++i) {
            #pragma unroll
            for (int j = 0; j < MINI_TILE_COLS; ++j) {
                threadResults[i][j] += fragA[i] * fragB[j];
            }
        }
    }
}

__global__ void matmul_triple_buffer(float* A, float* B, float* C, int M, int N, int K) {
    int num_elements_A = TILE_ROWS_A * TILE_COLS_A;
    int num_elements_B = TILE_ROWS_B * TILE_COLS_B;
    extern __shared__ char smem[];
    float* sharedA = reinterpret_cast<float*>(smem);
    float* sharedB = sharedA + (3 * num_elements_A);
    
    // Barriers: Start after Buffer B
    auto* barriers = reinterpret_cast<cuda::barrier<cuda::thread_scope::thread_scope_block>*>(
        sharedB + (3 * num_elements_B)
    );

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for(int i = 0; i < 3; ++i) init(&barriers[i], blockDim.x * blockDim.y);
    }
    __syncthreads();

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int load_threads = blockDim.x * blockDim.y; // Let everyone help load for 4090
    // Variables derived from blockIdx
    int R_start_A = by * TILE_ROWS_A;
    int C_start_B = bx * TILE_COLS_B;   
    int inner_row_A = ty * MINI_TILE_ROWS;
    int inner_col_A = tx * MINI_TILE_COLS;
    int inner_row_B = ty * MINI_TILE_ROWS;
    int inner_col_B = tx * MINI_TILE_COLS;
    int row = R_start_A + inner_row_A;
    int col = C_start_B + inner_col_B;

    // Registers
    float threadResults[MINI_TILE_ROWS][MINI_TILE_COLS] = {0.0};
    int num_tiles = K / TILE_COLS_A;

    // Indices
    int write_idx = 0; // Buffer being filled
    int read_idx = 0;  // Buffer being consumed

    if (num_tiles > 0) {
        issue_load(A, B, sharedA, sharedB, 0, barriers[0], tid, load_threads, K, N);
        //issue_load(A, B, sharedA, sharedB, 0, barriers[0], row,  col, inner_row_A, inner_col_A, inner_row_B, inner_col_B, K, N);
        write_idx = 1; 
    }
    if (num_tiles > 1) {
        issue_load(A, B, sharedA + num_elements_A, sharedB + num_elements_B, 1, barriers[1], tid, load_threads, K, N);
        //issue_load(A, B, sharedA + num_elements_A, sharedB + num_elements_B, 1, barriers[1], row,  col, inner_row_A, inner_col_A, inner_row_B, inner_col_B, K, N);
        write_idx = 2;
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        int next_load_tile = tile + 2;
        if (next_load_tile < num_tiles) {
            issue_load(A, B, sharedA + write_idx * num_elements_A, sharedB + write_idx * num_elements_B, next_load_tile, barriers[write_idx], tid, load_threads, K, N);
            //issue_load(A, B, sharedA + write_idx * num_elements_A, sharedB + write_idx * num_elements_B, next_load_tile, barriers[write_idx], 
            //    row,  col, inner_row_A, inner_col_A, inner_row_B, inner_col_B, K, N);
        }

        barriers[read_idx].arrive_and_wait();

        compute_tile(sharedA + read_idx * num_elements_A, sharedB + read_idx * num_elements_B, threadResults);
        __syncthreads();

        read_idx = (read_idx + 1) % 3;
        write_idx = (write_idx + 1) % 3;
    }

    compute_tile(sharedA + read_idx * num_elements_A, sharedB + read_idx * num_elements_B, threadResults);
    __syncthreads();

    read_idx = (read_idx + 1) % 3;

    compute_tile(sharedA + read_idx * num_elements_A, sharedB + read_idx * num_elements_B, threadResults);
    __syncthreads();

#pragma unroll
    for (uint i = 0; i < MINI_TILE_ROWS; ++i) {
        for (uint j = 0; j < MINI_TILE_COLS; j+=4) {
             reinterpret_cast<float4 *>(&C[(row + i) * N + col + j])[0] = 
                 reinterpret_cast<float4 *>(&threadResults[i][j])[0];            
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
    std::cout << "Matrix multiplication: C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]" << ", TILE_ROWS_A:" <<
        TILE_ROWS_A << " TILE_COLS_A:" << TILE_COLS_A << " TILE_COLS_B:" << TILE_COLS_B << " TILE_ROWS_B:" << TILE_ROWS_B << std::endl 
        << " MINI_TILE_ROWS:" << MINI_TILE_ROWS << " MINI_TILE_COLS:" << MINI_TILE_COLS << std::endl;

    // Calculate matrix sizes in bytes (unpadded)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    int M_pad = ((M + TILE_ROWS_A - 1) / TILE_ROWS_A) * TILE_ROWS_A;
    int N_pad = ((N + TILE_COLS_B - 1) / TILE_COLS_B) * TILE_COLS_B;
    int K_pad = ((K + TILE_COLS_A - 1) / TILE_COLS_A) * TILE_COLS_A;
    
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
    dim3 blockDim(TILE_COLS_B/MINI_TILE_COLS, TILE_ROWS_A/MINI_TILE_ROWS);
    dim3 gridDim(N_pad / TILE_COLS_B, M_pad / TILE_ROWS_A);

    size_t smem_size = (3 * TILE_ROWS_A * TILE_COLS_A * sizeof(float)) + 
                    (3 * TILE_ROWS_B * TILE_COLS_B * sizeof(float)) + 
                    (3 * sizeof(cuda::barrier<cuda::thread_scope::thread_scope_block>)) + 
                    128; // extra padding for alignment
    cudaFuncSetAttribute(matmul_triple_buffer, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    for (int i = 0; i < 1; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_triple_buffer<<<gridDim, blockDim, smem_size>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
        cudaDeviceSynchronize();
    }

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    float gpu_avg_time = benchmark_kernel([&]() {
        matmul_triple_buffer<<<gridDim, blockDim, smem_size>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
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
