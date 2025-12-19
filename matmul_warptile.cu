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

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const uint TILE_COLS_A= 32;
const uint TILE_ROWS_A= 64;
const uint TILE_COLS_B = 64;
const uint TILE_ROWS_B = 32;
const uint MINI_TILE_COLS = 4;
const uint MINI_TILE_ROWS = 8;

const uint K10_NUM_THREADS = 128;
const uint K10_BN = 128;
const uint K10_BM = 128;
const uint K10_BK = 16;
const uint K10_WN = 64;
const uint K10_WM = 64;
const uint K10_WNITER = 4;
const uint K10_TN = 4;
const uint K10_TM = 8;

constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

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

__global__ void matmul_rectangular_tile(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sharedA[TILE_ROWS_A][TILE_COLS_A];
    __shared__ float sharedB[TILE_ROWS_B][TILE_COLS_B];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // row is the index in C and A's row, col is the index in C and B's column
    int inner_row_A = ty * MINI_TILE_ROWS;
    int inner_col_A = tx * MINI_TILE_COLS;
    int inner_row_B = ty * MINI_TILE_ROWS;
    int inner_col_B = tx * MINI_TILE_COLS;
    int row = by * TILE_ROWS_A + inner_row_A;
    int col = bx * TILE_COLS_B + inner_col_B;
    
    // allocate thread-local cache for results in registerfile
    float threadResults[MINI_TILE_ROWS][MINI_TILE_COLS] = {0.0};    
#pragma unroll
    for (int tile = 0; tile < K / TILE_COLS_A; ++tile) {
        int global_offset_A = row * K + tile * TILE_COLS_A + inner_col_A;
        int global_offset_B = (tile * TILE_ROWS_B + inner_row_B) * N + col;     

        for(int i = 0; i < MINI_TILE_ROWS; i++){
            for(int j = 0; j < MINI_TILE_COLS; j+=4){
                if(inner_col_A + j + 3 < TILE_COLS_A){
                    /*float4 tmp =
                        reinterpret_cast<float4 *>(&A[(row + i) * K + tile * TILE_COLS_A + inner_col_A + j])[0];
                    sharedA[inner_col_A + j + 0][inner_row_A + i] = tmp.x;
                    sharedA[inner_col_A + j + 1][inner_row_A + i] = tmp.y;
                    sharedA[inner_col_A + j + 2][inner_row_A + i] = tmp.z;
                    sharedA[inner_col_A + j + 3][inner_row_A + i] = tmp.w;*/
                    reinterpret_cast<float4 *>(&sharedA[inner_row_A + i][inner_col_A + j])[0] = 
                        reinterpret_cast<float4 *>(&A[global_offset_A + j])[0];                      
                }

                if(inner_row_B + i < TILE_ROWS_B){
                    reinterpret_cast<float4 *>(&sharedB[inner_row_B + i][inner_col_B + j])[0] = 
                        reinterpret_cast<float4 *>(&B[(tile * TILE_ROWS_B + inner_row_B + i) * N + col + j])[0];   
                }
            }
            global_offset_A += K;
            global_offset_B += N;         
        }        
        __syncthreads();
        
        float fragA[MINI_TILE_ROWS];
        float fragB[MINI_TILE_COLS];
#pragma unroll
        for(int k=0; k < TILE_COLS_A; ++k) {
            for(int i=0; i < MINI_TILE_ROWS; i++) {
                fragA[i] = sharedA[inner_row_A + i][k];
            }
            for(int j=0; j < MINI_TILE_COLS; j++) {
                fragB[j] = sharedB[k][inner_col_B + j];
            }
            for(int i=0; i < MINI_TILE_ROWS; ++i) {
                for(int j=0; j < MINI_TILE_COLS; ++j) {
                    threadResults[i][j] += fragA[i] * fragB[j];
                }
            }
        } 
        __syncthreads();
    }
    
 #pragma unroll
    for (uint i = 0; i < MINI_TILE_ROWS; ++i) {
        for (uint j = 0; j < MINI_TILE_COLS; j+=4) {
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
    std::cout << "Matrix multiplication: C[" << M << "×" << N << "] = A[" << M << "×" << K << "] × B[" << K << "×" << N << "]" << ", TILE_ROWS_A:" <<
        TILE_ROWS_A << " TILE_COLS_A:" << TILE_COLS_A << " TILE_COLS_B:" << TILE_COLS_B << " TILE_ROWS_B:" << TILE_ROWS_B << std::endl <<
        " MINI_TILE_ROWS:" << MINI_TILE_ROWS << " MINI_TILE_COLS:" << MINI_TILE_COLS << std::endl;

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
    //dim3 blockDim(TILE_COLS_B/MINI_TILE_COLS, TILE_ROWS_A/MINI_TILE_ROWS);
    //dim3 gridDim(N_pad / TILE_COLS_B, M_pad / TILE_ROWS_A);
    dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
    dim3 blockDim(K10_NUM_THREADS);

    for (int i = 0; i < 1; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        //matmul_rectangular_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
        sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                        K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(M_pad, N_pad, K_pad, 1.0, d_A, d_B, 0, d_C);
        cudaDeviceSynchronize();
    }   
    
    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    float gpu_avg_time = benchmark_kernel([&]() {
        //matmul_rectangular_tile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_pad, N_pad, K_pad);
        sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                        K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(M_pad, N_pad, K_pad, 1.0, d_A, d_B, 0, d_C);
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
