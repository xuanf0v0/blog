---
title: 'INT8 量化矩阵乘法'
published: 2026-07-08
description: 'INT8 量化矩阵乘法 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 中等题
  - 矩阵计算
  - 全局内存
  - 量化
  - 深度学习算子
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/int8-quantized-matmul'
---

刷 LeetGPU 的时候，**INT8 量化矩阵乘法** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/int8-quantized-matmul](https://leetgpu.com/challenges/int8-quantized-matmul)

英文名可以先记一下：**INT8 Quantized Matmul**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、INT8 Quantized Matmul

INT8 矩阵乘法核心仍是 matmul，只是输入是量化值，累加通常用 int32，最后再按 scale 反量化。

这题的核心关系可以写成：`acc += int(A) * int(B), output = acc * scale`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
int acc = 0;
for (int k = 0; k < K; ++k)
    acc += int(A[k]) * int(B[k]);
output = acc * scale;
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>
#include <mma.h>
#include <sm_61_intrinsics.h>

#include <cstdint>

#ifndef ONLINE_JUDGE
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#endif

#define UPPER_DIV(x, y) (((x) + (y) - 1) / (y))

namespace {

constexpr int kBaselineTile = 16;

constexpr int kScalarBlockM = 128;
constexpr int kScalarBlockN = 64;
constexpr int kScalarBlockK = 32;
constexpr int kScalarThreadTileM = 8;
constexpr int kScalarThreadTileN = 4;
constexpr int kScalarThreadsX = kScalarBlockN / kScalarThreadTileN;
constexpr int kScalarThreadsY = kScalarBlockM / kScalarThreadTileM;

constexpr int kDp4aBlockM = 128;
constexpr int kDp4aBlockN = 64;
constexpr int kDp4aBlockK = 64;
constexpr int kDp4aPack = 4;
constexpr int kDp4aThreadTileM = 8;
constexpr int kDp4aThreadTileN = 4;
constexpr int kDp4aThreadsX = kDp4aBlockN / kDp4aThreadTileN;
constexpr int kDp4aThreadsY = kDp4aBlockM / kDp4aThreadTileM;

constexpr int kTcWmmaM = 16;
constexpr int kTcWmmaN = 16;
constexpr int kTcWmmaK = 16;
constexpr int kTcWarpsM = 4;
constexpr int kTcWarpsN = 2;
constexpr int kTcBlockM = kTcWarpsM * kTcWmmaM;
constexpr int kTcBlockN = kTcWarpsN * kTcWmmaN;
constexpr int kTcThreads = 32 * kTcWarpsM * kTcWarpsN;

struct Workspace {
  int* raw_C = nullptr;
  int* row_sums = nullptr;
  int* col_sums = nullptr;
  size_t raw_capacity = 0;
  size_t row_capacity = 0;
  size_t col_capacity = 0;
};

__host__ __device__ __forceinline__ int clamp_int(int value, int low, int high) {
  value = value < low ? low : value;
  value = value > high ? high : value;
  return value;
}

__host__ __device__ __forceinline__ int8_t clamp_to_int8(int value) {
  return static_cast<int8_t>(clamp_int(value, -128, 127));
}

__host__ __device__ __forceinline__ int round_to_nearest_even(float value) {
  return static_cast<int>(nearbyintf(value));
}

__host__ __device__ __forceinline__ int quantize_accumulator(int acc,
                                                             float scale_A,
                                                             float scale_B,
                                                             float scale_C,
                                                             int zero_point_C) {
  float scaled = static_cast<float>(acc) * scale_A;
  scaled *= scale_B;
  scaled /= scale_C;
  return round_to_nearest_even(scaled) + zero_point_C;
}

__device__ __forceinline__ int pack_int8x4_device(const int8_t* values) {
  return static_cast<int>(static_cast<unsigned char>(values[0])) |
         (static_cast<int>(static_cast<unsigned char>(values[1])) << 8) |
         (static_cast<int>(static_cast<unsigned char>(values[2])) << 16) |
         (static_cast<int>(static_cast<unsigned char>(values[3])) << 24);
}

__device__ __forceinline__ int dp4a_int8x4(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  c += static_cast<int>(static_cast<int8_t>(a & 0xFF)) * static_cast<int>(static_cast<int8_t>(b & 0xFF));
  c += static_cast<int>(static_cast<int8_t>((a >> 8) & 0xFF)) *
       static_cast<int>(static_cast<int8_t>((b >> 8) & 0xFF));
  c += static_cast<int>(static_cast<int8_t>((a >> 16) & 0xFF)) *
       static_cast<int>(static_cast<int8_t>((b >> 16) & 0xFF));
  c += static_cast<int>(static_cast<int8_t>((a >> 24) & 0xFF)) *
       static_cast<int>(static_cast<int8_t>((b >> 24) & 0xFF));
  return c;
#endif
}

__global__ void baseline_kernel(const int8_t* A,
                                const int8_t* B,
                                int8_t* C,
                                int M,
                                int N,
                                int K,
                                float scale_A,
                                float scale_B,
                                float scale_C,
                                int zero_point_A,
                                int zero_point_B,
                                int zero_point_C) {
  const int c_row = blockIdx.y * blockDim.y + threadIdx.y;
  const int c_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (c_row >= M || c_col >= N) {
    return;
  }

  int acc = 0;
  for (int k = 0; k < K; ++k) {
    const int a = static_cast<int>(A[c_row * K + k]) - zero_point_A;
    const int b = static_cast<int>(B[k * N + c_col]) - zero_point_B;
    acc += a * b;
  }

  C[c_row * N + c_col] = clamp_to_int8(
      quantize_accumulator(acc, scale_A, scale_B, scale_C, zero_point_C));
}

__global__ void row_sum_kernel(const int8_t* A, int* row_sums, int M, int K) {
  const int row = blockIdx.x;
  if (row >= M) {
    return;
  }

  int local = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    local += static_cast<int>(A[row * K + k]);
  }

  __shared__ int scratch[256];
  scratch[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    row_sums[row] = scratch[0];
  }
}

__global__ void col_sum_kernel(const int8_t* B, int* col_sums, int K, int N) {
  const int col = blockIdx.x;
  if (col >= N) {
    return;
  }

  int local = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    local += static_cast<int>(B[k * N + col]);
  }

  __shared__ int scratch[256];
  scratch[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    col_sums[col] = scratch[0];
  }
}

__global__ void matmul_raw_tiled_scalar_kernel(const int8_t* A,
                                               const int8_t* B,
                                               int* raw_C,
                                               int M,
                                               int N,
                                               int K) {
  __shared__ int8_t shmem_A[kScalarBlockM][kScalarBlockK];
  __shared__ int8_t shmem_B[kScalarBlockK][kScalarBlockN];

  const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_row = blockIdx.y * kScalarBlockM;
  const int block_col = blockIdx.x * kScalarBlockN;
  const int row_base = block_row + threadIdx.y * kScalarThreadTileM;
  const int col_base = block_col + threadIdx.x * kScalarThreadTileN;

  int acc[kScalarThreadTileM][kScalarThreadTileN];
#pragma unroll
  for (int i = 0; i < kScalarThreadTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kScalarThreadTileN; ++j) {
      acc[i][j] = 0;
    }
  }

  for (int kk = 0; kk < K; kk += kScalarBlockK) {
    for (int index = thread_linear; index < kScalarBlockM * kScalarBlockK;
         index += blockDim.x * blockDim.y) {
      const int local_row = index / kScalarBlockK;
      const int local_col = index - local_row * kScalarBlockK;
      const int global_row = block_row + local_row;
      const int global_col = kk + local_col;
      shmem_A[local_row][local_col] =
          (global_row < M && global_col < K) ? A[global_row * K + global_col] : static_cast<int8_t>(0);
    }

    for (int index = thread_linear; index < kScalarBlockK * kScalarBlockN;
         index += blockDim.x * blockDim.y) {
      const int local_row = index / kScalarBlockN;
      const int local_col = index - local_row * kScalarBlockN;
      const int global_row = kk + local_row;
      const int global_col = block_col + local_col;
      shmem_B[local_row][local_col] =
          (global_row < K && global_col < N) ? B[global_row * N + global_col] : static_cast<int8_t>(0);
    }
    __syncthreads();

#pragma unroll
    for (int k_inner = 0; k_inner < kScalarBlockK; ++k_inner) {
      int a_frag[kScalarThreadTileM];
      int b_frag[kScalarThreadTileN];

#pragma unroll
      for (int i = 0; i < kScalarThreadTileM; ++i) {
        a_frag[i] = static_cast<int>(shmem_A[threadIdx.y * kScalarThreadTileM + i][k_inner]);
      }
#pragma unroll
      for (int j = 0; j < kScalarThreadTileN; ++j) {
        b_frag[j] = static_cast<int>(shmem_B[k_inner][threadIdx.x * kScalarThreadTileN + j]);
      }

#pragma unroll
      for (int i = 0; i < kScalarThreadTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kScalarThreadTileN; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kScalarThreadTileM; ++i) {
    const int row = row_base + i;
    if (row >= M) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < kScalarThreadTileN; ++j) {
      const int col = col_base + j;
      if (col < N) {
        raw_C[row * N + col] = acc[i][j];
      }
    }
  }
}

__global__ void matmul_raw_dp4a_kernel(const int8_t* A,
                                       const int8_t* B,
                                       int* raw_C,
                                       int M,
                                       int N,
                                       int K) {
  __shared__ int shmem_A[kDp4aBlockM][kDp4aBlockK / kDp4aPack];
  __shared__ int shmem_B[kDp4aBlockN][kDp4aBlockK / kDp4aPack];

  const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_row = blockIdx.y * kDp4aBlockM;
  const int block_col = blockIdx.x * kDp4aBlockN;
  const int row_base = block_row + threadIdx.y * kDp4aThreadTileM;
  const int col_base = block_col + threadIdx.x * kDp4aThreadTileN;

  int acc[kDp4aThreadTileM][kDp4aThreadTileN];
#pragma unroll
  for (int i = 0; i < kDp4aThreadTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kDp4aThreadTileN; ++j) {
      acc[i][j] = 0;
    }
  }

  for (int kk = 0; kk < K; kk += kDp4aBlockK) {
    for (int index = thread_linear; index < kDp4aBlockM * (kDp4aBlockK / kDp4aPack);
         index += blockDim.x * blockDim.y) {
      const int local_row = index / (kDp4aBlockK / kDp4aPack);
      const int pack_idx = index - local_row * (kDp4aBlockK / kDp4aPack);
      const int global_row = block_row + local_row;
      const int k_base = kk + pack_idx * kDp4aPack;

      int8_t bytes[kDp4aPack] = {0, 0, 0, 0};
#pragma unroll
      for (int t = 0; t < kDp4aPack; ++t) {
        if (global_row < M && (k_base + t) < K) {
          bytes[t] = A[global_row * K + k_base + t];
        }
      }
      shmem_A[local_row][pack_idx] = pack_int8x4_device(bytes);
    }

    for (int index = thread_linear; index < kDp4aBlockN * (kDp4aBlockK / kDp4aPack);
         index += blockDim.x * blockDim.y) {
      const int local_col = index / (kDp4aBlockK / kDp4aPack);
      const int pack_idx = index - local_col * (kDp4aBlockK / kDp4aPack);
      const int global_col = block_col + local_col;
      const int k_base = kk + pack_idx * kDp4aPack;

      int8_t bytes[kDp4aPack] = {0, 0, 0, 0};
#pragma unroll
      for (int t = 0; t < kDp4aPack; ++t) {
        if ((k_base + t) < K && global_col < N) {
          bytes[t] = B[(k_base + t) * N + global_col];
        }
      }
      shmem_B[local_col][pack_idx] = pack_int8x4_device(bytes);
    }
    __syncthreads();

#pragma unroll
    for (int pack_idx = 0; pack_idx < (kDp4aBlockK / kDp4aPack); ++pack_idx) {
      int a_frag[kDp4aThreadTileM];
      int b_frag[kDp4aThreadTileN];

#pragma unroll
      for (int i = 0; i < kDp4aThreadTileM; ++i) {
        a_frag[i] = shmem_A[threadIdx.y * kDp4aThreadTileM + i][pack_idx];
      }
#pragma unroll
      for (int j = 0; j < kDp4aThreadTileN; ++j) {
        b_frag[j] = shmem_B[threadIdx.x * kDp4aThreadTileN + j][pack_idx];
      }

#pragma unroll
      for (int i = 0; i < kDp4aThreadTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kDp4aThreadTileN; ++j) {
          acc[i][j] = dp4a_int8x4(a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kDp4aThreadTileM; ++i) {
    const int row = row_base + i;
    if (row >= M) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < kDp4aThreadTileN; ++j) {
      const int col = col_base + j;
      if (col < N) {
        raw_C[row * N + col] = acc[i][j];
      }
    }
  }
}

__global__ void matmul_quantized_dp4a_zero_zp_kernel(const int8_t* A,
                                                     const int8_t* B,
                                                     int8_t* C,
                                                     int M,
                                                     int N,
                                                     int K,
                                                     float scale_A,
                                                     float scale_B,
                                                     float scale_C,
                                                     int zero_point_C) {
  __shared__ int shmem_A[kDp4aBlockM][kDp4aBlockK / kDp4aPack];
  __shared__ int shmem_B[kDp4aBlockN][kDp4aBlockK / kDp4aPack];

  const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_row = blockIdx.y * kDp4aBlockM;
  const int block_col = blockIdx.x * kDp4aBlockN;
  const int row_base = block_row + threadIdx.y * kDp4aThreadTileM;
  const int col_base = block_col + threadIdx.x * kDp4aThreadTileN;

  int acc[kDp4aThreadTileM][kDp4aThreadTileN];
#pragma unroll
  for (int i = 0; i < kDp4aThreadTileM; ++i) {
#pragma unroll
    for (int j = 0; j < kDp4aThreadTileN; ++j) {
      acc[i][j] = 0;
    }
  }

  for (int kk = 0; kk < K; kk += kDp4aBlockK) {
    for (int index = thread_linear; index < kDp4aBlockM * (kDp4aBlockK / kDp4aPack);
         index += blockDim.x * blockDim.y) {
      const int local_row = index / (kDp4aBlockK / kDp4aPack);
      const int pack_idx = index - local_row * (kDp4aBlockK / kDp4aPack);
      const int global_row = block_row + local_row;
      const int k_base = kk + pack_idx * kDp4aPack;

      int8_t bytes[kDp4aPack] = {0, 0, 0, 0};
#pragma unroll
      for (int t = 0; t < kDp4aPack; ++t) {
        if (global_row < M && (k_base + t) < K) {
          bytes[t] = A[global_row * K + k_base + t];
        }
      }
      shmem_A[local_row][pack_idx] = pack_int8x4_device(bytes);
    }

    for (int index = thread_linear; index < kDp4aBlockN * (kDp4aBlockK / kDp4aPack);
         index += blockDim.x * blockDim.y) {
      const int local_col = index / (kDp4aBlockK / kDp4aPack);
      const int pack_idx = index - local_col * (kDp4aBlockK / kDp4aPack);
      const int global_col = block_col + local_col;
      const int k_base = kk + pack_idx * kDp4aPack;

      int8_t bytes[kDp4aPack] = {0, 0, 0, 0};
#pragma unroll
      for (int t = 0; t < kDp4aPack; ++t) {
        if ((k_base + t) < K && global_col < N) {
          bytes[t] = B[(k_base + t) * N + global_col];
        }
      }
      shmem_B[local_col][pack_idx] = pack_int8x4_device(bytes);
    }
    __syncthreads();

#pragma unroll
    for (int pack_idx = 0; pack_idx < (kDp4aBlockK / kDp4aPack); ++pack_idx) {
      int a_frag[kDp4aThreadTileM];
      int b_frag[kDp4aThreadTileN];

#pragma unroll
      for (int i = 0; i < kDp4aThreadTileM; ++i) {
        a_frag[i] = shmem_A[threadIdx.y * kDp4aThreadTileM + i][pack_idx];
      }
#pragma unroll
      for (int j = 0; j < kDp4aThreadTileN; ++j) {
        b_frag[j] = shmem_B[threadIdx.x * kDp4aThreadTileN + j][pack_idx];
      }

#pragma unroll
      for (int i = 0; i < kDp4aThreadTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kDp4aThreadTileN; ++j) {
          acc[i][j] = dp4a_int8x4(a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kDp4aThreadTileM; ++i) {
    const int row = row_base + i;
    if (row >= M) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < kDp4aThreadTileN; ++j) {
      const int col = col_base + j;
      if (col < N) {
        C[row * N + col] =
            clamp_to_int8(quantize_accumulator(acc[i][j], scale_A, scale_B, scale_C, zero_point_C));
      }
    }
  }
}

__global__ void matmul_raw_tensor_core_kernel(const int8_t* A,
                                              const int8_t* B,
                                              int* raw_C,
                                              int M,
                                              int N,
                                              int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  namespace wmma = nvcuda::wmma;

  const int warp_id = threadIdx.x / warpSize;
  const int warp_m = warp_id / kTcWarpsN;
  const int warp_n = warp_id - warp_m * kTcWarpsN;
  const int row = blockIdx.y * kTcBlockM + warp_m * kTcWmmaM;
  const int col = blockIdx.x * kTcBlockN + warp_n * kTcWmmaN;

  wmma::fragment<wmma::accumulator, kTcWmmaM, kTcWmmaN, kTcWmmaK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int kk = 0; kk < K; kk += kTcWmmaK) {
    wmma::fragment<wmma::matrix_a, kTcWmmaM, kTcWmmaN, kTcWmmaK, signed char, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTcWmmaM, kTcWmmaN, kTcWmmaK, signed char, wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, reinterpret_cast<const signed char*>(A + row * K + kk), K);
    wmma::load_matrix_sync(b_frag, reinterpret_cast<const signed char*>(B + kk * N + col), N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(raw_C + row * N + col, acc_frag, N, wmma::mem_row_major);
#endif
}

__global__ void matmul_quantized_tensor_core_zero_zp_kernel(const int8_t* A,
                                                            const int8_t* B,
                                                            int8_t* C,
                                                            int M,
                                                            int N,
                                                            int K,
                                                            float scale_A,
                                                            float scale_B,
                                                            float scale_C,
                                                            int zero_point_C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  namespace wmma = nvcuda::wmma;

  __shared__ int warp_tiles[kTcWarpsM * kTcWarpsN][kTcWmmaM * kTcWmmaN];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x / warpSize;
  const int warp_m = warp_id / kTcWarpsN;
  const int warp_n = warp_id - warp_m * kTcWarpsN;
  const int row = blockIdx.y * kTcBlockM + warp_m * kTcWmmaM;
  const int col = blockIdx.x * kTcBlockN + warp_n * kTcWmmaN;

  wmma::fragment<wmma::accumulator, kTcWmmaM, kTcWmmaN, kTcWmmaK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  for (int kk = 0; kk < K; kk += kTcWmmaK) {
    wmma::fragment<wmma::matrix_a, kTcWmmaM, kTcWmmaN, kTcWmmaK, signed char, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kTcWmmaM, kTcWmmaN, kTcWmmaK, signed char, wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, reinterpret_cast<const signed char*>(A + row * K + kk), K);
    wmma::load_matrix_sync(b_frag, reinterpret_cast<const signed char*>(B + kk * N + col), N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  int* warp_tile = warp_tiles[warp_id];
  wmma::store_matrix_sync(warp_tile, acc_frag, kTcWmmaN, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane; idx < kTcWmmaM * kTcWmmaN; idx += warpSize) {
    const int local_row = idx / kTcWmmaN;
    const int local_col = idx - local_row * kTcWmmaN;
    const int global_row = row + local_row;
    const int global_col = col + local_col;
    if (global_row < M && global_col < N) {
      C[global_row * N + global_col] = clamp_to_int8(
          quantize_accumulator(warp_tile[idx], scale_A, scale_B, scale_C, zero_point_C));
    }
  }
#endif
}

__global__ void finalize_quantized_no_correction_kernel(const int* raw_C,
                                                        int8_t* C,
                                                        int M,
                                                        int N,
                                                        float scale_A,
                                                        float scale_B,
                                                        float scale_C,
                                                        int zero_point_C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  const int raw = raw_C[row * N + col];
  C[row * N + col] = clamp_to_int8(
      quantize_accumulator(raw, scale_A, scale_B, scale_C, zero_point_C));
}

__global__ void finalize_quantized_kernel(const int* raw_C,
                                          const int* row_sums,
                                          const int* col_sums,
                                          int8_t* C,
                                          int M,
                                          int N,
                                          int K,
                                          float scale_A,
                                          float scale_B,
                                          float scale_C,
                                          int zero_point_A,
                                          int zero_point_B,
                                          int zero_point_C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  const int raw = raw_C[row * N + col];
  const int corrected = raw - zero_point_B * row_sums[row] - zero_point_A * col_sums[col] +
                        K * zero_point_A * zero_point_B;
  C[row * N + col] = clamp_to_int8(
      quantize_accumulator(corrected, scale_A, scale_B, scale_C, zero_point_C));
}

void launch_baseline_impl(const int8_t* A,
                          const int8_t* B,
                          int8_t* C,
                          int M,
                          int N,
                          int K,
                          float scale_A,
                          float scale_B,
                          float scale_C,
                          int zero_point_A,
                          int zero_point_B,
                          int zero_point_C) {
  const dim3 threads(kBaselineTile, kBaselineTile);
  const dim3 blocks(UPPER_DIV(N, kBaselineTile), UPPER_DIV(M, kBaselineTile));
  baseline_kernel<<<blocks, threads>>>(A,
                                       B,
                                       C,
                                       M,
                                       N,
                                       K,
                                       scale_A,
                                       scale_B,
                                       scale_C,
                                       zero_point_A,
                                       zero_point_B,
                                       zero_point_C);
}

void ensure_workspace(Workspace& workspace, int M, int N) {
  const size_t raw_needed = static_cast<size_t>(M) * static_cast<size_t>(N);
  const size_t row_needed = static_cast<size_t>(M);
  const size_t col_needed = static_cast<size_t>(N);

  if (raw_needed > workspace.raw_capacity) {
    if (workspace.raw_C != nullptr) {
      cudaFree(workspace.raw_C);
    }
    cudaMalloc(&workspace.raw_C, raw_needed * sizeof(int));
    workspace.raw_capacity = raw_needed;
  }
  if (row_needed > workspace.row_capacity) {
    if (workspace.row_sums != nullptr) {
      cudaFree(workspace.row_sums);
    }
    cudaMalloc(&workspace.row_sums, row_needed * sizeof(int));
    workspace.row_capacity = row_needed;
  }
  if (col_needed > workspace.col_capacity) {
    if (workspace.col_sums != nullptr) {
      cudaFree(workspace.col_sums);
    }
    cudaMalloc(&workspace.col_sums, col_needed * sizeof(int));
    workspace.col_capacity = col_needed;
  }
}

void launch_scalar_impl(const int8_t* A,
                        const int8_t* B,
                        int8_t* C,
                        int M,
                        int N,
                        int K,
                        float scale_A,
                        float scale_B,
                        float scale_C,
                        int zero_point_A,
                        int zero_point_B,
                        int zero_point_C) {
  static Workspace workspace;
  ensure_workspace(workspace, M, N);

  const dim3 matmul_threads(kScalarThreadsX, kScalarThreadsY);
  const dim3 matmul_blocks(UPPER_DIV(N, kScalarBlockN), UPPER_DIV(M, kScalarBlockM));
  matmul_raw_tiled_scalar_kernel<<<matmul_blocks, matmul_threads>>>(A, B, workspace.raw_C, M, N, K);

  constexpr int kReduceThreads = 256;
  row_sum_kernel<<<M, kReduceThreads>>>(A, workspace.row_sums, M, K);
  col_sum_kernel<<<N, kReduceThreads>>>(B, workspace.col_sums, K, N);

  const dim3 finalize_threads(kBaselineTile, kBaselineTile);
  const dim3 finalize_blocks(UPPER_DIV(N, kBaselineTile), UPPER_DIV(M, kBaselineTile));
  finalize_quantized_kernel<<<finalize_blocks, finalize_threads>>>(workspace.raw_C,
                                                                   workspace.row_sums,
                                                                   workspace.col_sums,
                                                                   C,
                                                                   M,
                                                                   N,
                                                                   K,
                                                                   scale_A,
                                                                   scale_B,
                                                                   scale_C,
                                                                   zero_point_A,
                                                                   zero_point_B,
                                                                   zero_point_C);
}

void launch_dp4a_impl(const int8_t* A,
                      const int8_t* B,
                      int8_t* C,
                      int M,
                      int N,
                      int K,
                      float scale_A,
                      float scale_B,
                      float scale_C,
                      int zero_point_A,
                      int zero_point_B,
                      int zero_point_C) {
  const dim3 dp4a_threads(kDp4aThreadsX, kDp4aThreadsY);
  const dim3 dp4a_blocks(UPPER_DIV(N, kDp4aBlockN), UPPER_DIV(M, kDp4aBlockM));

  if (zero_point_A == 0 && zero_point_B == 0) {
    matmul_quantized_dp4a_zero_zp_kernel<<<dp4a_blocks, dp4a_threads>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_C);
    return;
  }

  static Workspace workspace;
  ensure_workspace(workspace, M, N);

  matmul_raw_dp4a_kernel<<<dp4a_blocks, dp4a_threads>>>(A, B, workspace.raw_C, M, N, K);

  constexpr int kReduceThreads = 256;
  row_sum_kernel<<<M, kReduceThreads>>>(A, workspace.row_sums, M, K);
  col_sum_kernel<<<N, kReduceThreads>>>(B, workspace.col_sums, K, N);

  const dim3 finalize_threads(kBaselineTile, kBaselineTile);
  const dim3 finalize_blocks(UPPER_DIV(N, kBaselineTile), UPPER_DIV(M, kBaselineTile));
  finalize_quantized_kernel<<<finalize_blocks, finalize_threads>>>(workspace.raw_C,
                                                                   workspace.row_sums,
                                                                   workspace.col_sums,
                                                                   C,
                                                                   M,
                                                                   N,
                                                                   K,
                                                                   scale_A,
                                                                   scale_B,
                                                                   scale_C,
                                                                   zero_point_A,
                                                                   zero_point_B,
                                                                   zero_point_C);
}

void launch_tensor_core_impl(const int8_t* A,
                             const int8_t* B,
                             int8_t* C,
                             int M,
                             int N,
                             int K,
                             float scale_A,
                             float scale_B,
                             float scale_C,
                             int zero_point_A,
                             int zero_point_B,
                             int zero_point_C) {
  if ((M % kTcWmmaM) != 0 || (N % kTcWmmaN) != 0 || (K % kTcWmmaK) != 0) {
    launch_scalar_impl(A,
                       B,
                       C,
                       M,
                       N,
                       K,
                       scale_A,
                       scale_B,
                       scale_C,
                       zero_point_A,
                       zero_point_B,
                       zero_point_C);
    return;
  }

  static Workspace workspace;
  ensure_workspace(workspace, M, N);

  const dim3 tc_blocks(UPPER_DIV(N, kTcBlockN), UPPER_DIV(M, kTcBlockM));
  if (zero_point_A == 0 && zero_point_B == 0) {
    matmul_quantized_tensor_core_zero_zp_kernel<<<tc_blocks, kTcThreads>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_C);
    return;
  }

  matmul_raw_tensor_core_kernel<<<tc_blocks, kTcThreads>>>(A, B, workspace.raw_C, M, N, K);

  const dim3 finalize_threads(kBaselineTile, kBaselineTile);
  const dim3 finalize_blocks(UPPER_DIV(N, kBaselineTile), UPPER_DIV(M, kBaselineTile));

  constexpr int kReduceThreads = 256;
  row_sum_kernel<<<M, kReduceThreads>>>(A, workspace.row_sums, M, K);
  col_sum_kernel<<<N, kReduceThreads>>>(B, workspace.col_sums, K, N);
  finalize_quantized_kernel<<<finalize_blocks, finalize_threads>>>(workspace.raw_C,
                                                                   workspace.row_sums,
                                                                   workspace.col_sums,
                                                                   C,
                                                                   M,
                                                                   N,
                                                                   K,
                                                                   scale_A,
                                                                   scale_B,
                                                                   scale_C,
                                                                   zero_point_A,
                                                                   zero_point_B,
                                                                   zero_point_C);
}

void launch_optimized_impl(const int8_t* A,
                           const int8_t* B,
                           int8_t* C,
                           int M,
                           int N,
                           int K,
                           float scale_A,
                           float scale_B,
                           float scale_C,
                           int zero_point_A,
                           int zero_point_B,
                           int zero_point_C) {
  launch_tensor_core_impl(A,
                          B,
                          C,
                          M,
                          N,
                          K,
                          scale_A,
                          scale_B,
                          scale_C,
                          zero_point_A,
                          zero_point_B,
                          zero_point_C);
}

}  // namespace

// A, B, C are device pointers
extern "C" void solve(const int8_t* A,
                      const int8_t* B,
                      int8_t* C,
                      int M,
                      int N,
                      int K,
                      float scale_A,
                      float scale_B,
                      float scale_C,
                      int zero_point_A,
                      int zero_point_B,
                      int zero_point_C) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return;
  }

  launch_optimized_impl(A,
                        B,
                        C,
                        M,
                        N,
                        K,
                        scale_A,
                        scale_B,
                        scale_C,
                        zero_point_A,
                        zero_point_B,
                        zero_point_C);
}

#ifndef ONLINE_JUDGE
namespace {

std::vector<int8_t> quantized_matmul_reference(const std::vector<int8_t>& A,
                                               const std::vector<int8_t>& B,
                                               int M,
                                               int N,
                                               int K,
                                               float scale_A,
                                               float scale_B,
                                               float scale_C,
                                               int zero_point_A,
                                               int zero_point_B,
                                               int zero_point_C) {
  std::vector<int8_t> expected(static_cast<size_t>(M) * static_cast<size_t>(N), 0);
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      int acc = 0;
      for (int k = 0; k < K; ++k) {
        const int a = static_cast<int>(A[row * K + k]) - zero_point_A;
        const int b = static_cast<int>(B[k * N + col]) - zero_point_B;
        acc += a * b;
      }
      expected[row * N + col] = clamp_to_int8(
          quantize_accumulator(acc, scale_A, scale_B, scale_C, zero_point_C));
    }
  }
  return expected;
}

std::vector<int8_t> make_patterned_data(int rows, int cols, int period, int scale, int bias) {
  std::vector<int8_t> values(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int centered = ((row * 7 + col * 11) % period) - (period / 2);
      const int value = centered * scale + bias;
      values[row * cols + col] = static_cast<int8_t>(clamp_int(value, -128, 127));
    }
  }
  return values;
}

std::vector<int8_t> make_lcg_data(int count, int seed) {
  std::vector<int8_t> values(count, 0);
  uint32_t state = static_cast<uint32_t>(seed);
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values[i] = static_cast<int8_t>(static_cast<int>((state >> 24) & 0xFF) - 128);
  }
  return values;
}

bool run_case_with_launcher(const std::string& name,
                            void (*launcher)(const int8_t*,
                                             const int8_t*,
                                             int8_t*,
                                             int,
                                             int,
                                             int,
                                             float,
                                             float,
                                             float,
                                             int,
                                             int,
                                             int),
                            const std::vector<int8_t>& A,
                            const std::vector<int8_t>& B,
                            int M,
                            int N,
                            int K,
                            float scale_A,
                            float scale_B,
                            float scale_C,
                            int zero_point_A,
                            int zero_point_B,
                            int zero_point_C) {
  const size_t a_bytes = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(int8_t);
  const size_t b_bytes = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(int8_t);
  const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int8_t);

  const std::vector<int8_t> expected = quantized_matmul_reference(A,
                                                                  B,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  scale_A,
                                                                  scale_B,
                                                                  scale_C,
                                                                  zero_point_A,
                                                                  zero_point_B,
                                                                  zero_point_C);
  std::vector<int8_t> actual(static_cast<size_t>(M) * static_cast<size_t>(N), 0);

  int8_t* d_A = nullptr;
  int8_t* d_B = nullptr;
  int8_t* d_C = nullptr;
  auto cleanup = [&]() {
    if (d_A != nullptr) cudaFree(d_A);
    if (d_B != nullptr) cudaFree(d_B);
    if (d_C != nullptr) cudaFree(d_C);
  };

  if (cudaMalloc(&d_A, a_bytes) != cudaSuccess ||
      cudaMalloc(&d_B, b_bytes) != cudaSuccess ||
      cudaMalloc(&d_C, c_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  bool ok = true;
  ok = ok && (cudaMemcpy(d_A, A.data(), a_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_B, B.data(), b_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemset(d_C, 0, c_bytes) == cudaSuccess);
  if (!ok) {
    std::cerr << "[FAIL] " << name << ": device upload failed\n";
    cleanup();
    return false;
  }

  launcher(d_A,
           d_B,
           d_C,
           M,
           N,
           K,
           scale_A,
           scale_B,
           scale_C,
           zero_point_A,
           zero_point_B,
           zero_point_C);

  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_C, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution failed\n";
    cleanup();
    return false;
  }

  for (int i = 0; i < M * N; ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << i
                << ", expected=" << static_cast<int>(expected[i])
                << ", actual=" << static_cast<int>(actual[i]) << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  return true;
}

bool run_case(const std::string& name,
              const std::vector<int8_t>& A,
              const std::vector<int8_t>& B,
              int M,
              int N,
              int K,
              float scale_A,
              float scale_B,
              float scale_C,
              int zero_point_A,
              int zero_point_B,
              int zero_point_C) {
  const bool baseline_ok = run_case_with_launcher(name + "/baseline",
                                                  launch_baseline_impl,
                                                  A,
                                                  B,
                                                  M,
                                                  N,
                                                  K,
                                                  scale_A,
                                                  scale_B,
                                                  scale_C,
                                                  zero_point_A,
                                                  zero_point_B,
                                                  zero_point_C);
  if (!baseline_ok) {
    return false;
  }

  const bool optimized_ok = run_case_with_launcher(name + "/optimized",
                                                   launch_optimized_impl,
                                                   A,
                                                   B,
                                                   M,
                                                   N,
                                                   K,
                                                   scale_A,
                                                   scale_B,
                                                   scale_C,
                                                   zero_point_A,
                                                   zero_point_B,
                                                   zero_point_C);
  if (!optimized_ok) {
    return false;
  }

  std::cout << "[PASS] " << name << '\n';
  return true;
}

float benchmark_launcher(void (*launcher)(const int8_t*,
                                          const int8_t*,
                                          int8_t*,
                                          int,
                                          int,
                                          int,
                                          float,
                                          float,
                                          float,
                                          int,
                                          int,
                                          int),
                         const std::vector<int8_t>& A,
                         const std::vector<int8_t>& B,
                         int M,
                         int N,
                         int K,
                         float scale_A,
                         float scale_B,
                         float scale_C,
                         int zero_point_A,
                         int zero_point_B,
                         int zero_point_C,
                         int warmup,
                         int iterations) {
  const size_t a_bytes = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(int8_t);
  const size_t b_bytes = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(int8_t);
  const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int8_t);

  int8_t* d_A = nullptr;
  int8_t* d_B = nullptr;
  int8_t* d_C = nullptr;
  cudaMalloc(&d_A, a_bytes);
  cudaMalloc(&d_B, b_bytes);
  cudaMalloc(&d_C, c_bytes);
  cudaMemcpy(d_A, A.data(), a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), b_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, c_bytes);

  for (int i = 0; i < warmup; ++i) {
    launcher(d_A,
             d_B,
             d_C,
             M,
             N,
             K,
             scale_A,
             scale_B,
             scale_C,
             zero_point_A,
             zero_point_B,
             zero_point_C);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    launcher(d_A,
             d_B,
             d_C,
             M,
             N,
             K,
             scale_A,
             scale_B,
             scale_C,
             zero_point_A,
             zero_point_B,
             zero_point_C);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return elapsed_ms / static_cast<float>(iterations);
}

void run_bench_case(const std::string& name,
                    int M,
                    int N,
                    int K,
                    float scale_A,
                    float scale_B,
                    float scale_C,
                    int zero_point_A,
                    int zero_point_B,
                    int zero_point_C,
                    int warmup,
                    int iterations) {
  const std::vector<int8_t> A = make_lcg_data(M * K, 123 + M + K);
  const std::vector<int8_t> B = make_lcg_data(K * N, 987 + N + K);

  const float baseline_ms = benchmark_launcher(launch_baseline_impl,
                                               A,
                                               B,
                                               M,
                                               N,
                                               K,
                                               scale_A,
                                               scale_B,
                                               scale_C,
                                               zero_point_A,
                                               zero_point_B,
                                               zero_point_C,
                                               warmup,
                                               iterations);
  const float optimized_ms = benchmark_launcher(launch_optimized_impl,
                                                A,
                                                B,
                                                M,
                                                N,
                                                K,
                                                scale_A,
                                                scale_B,
                                                scale_C,
                                                zero_point_A,
                                                zero_point_B,
                                                zero_point_C,
                                                warmup,
                                                iterations);
  std::cout << "[BENCH] " << name << " M=" << M << " N=" << N << " K=" << K
            << " baseline=" << baseline_ms << " ms"
            << " optimized=" << optimized_ms << " ms"
            << " speedup=" << (baseline_ms / optimized_ms) << "x\n";
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int M = 2;
    const int N = 2;
    const int K = 2;
    const std::vector<int8_t> A = {1, 2, 3, 4};
    const std::vector<int8_t> B = {5, 6, 7, 8};
    passed += run_case("sample_case_basic_scales", A, B, M, N, K, 0.1f, 0.2f, 0.05f, 0, 0, 0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 1;
    const int K = 2;
    const std::vector<int8_t> A = {1, 2};
    const std::vector<int8_t> B = {3, 4};
    passed += run_case("sample_case_zero_points", A, B, M, N, K, 1.0f, 1.0f, 1.0f, 1, 3, 5) ? 1
                                                                                               : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 3;
    const int K = 4;
    const std::vector<int8_t> A = {-128, 127, -120, 126};
    const std::vector<int8_t> B = {
        127, -128, 127,
        -128, 127, -128,
        127, -128, 127,
        -128, 127, -128,
    };
    passed += run_case("clamp_extreme_values", A, B, M, N, K, 1.0f, 1.0f, 0.25f, 0, 0, 0) ? 1
                                                                                             : 0;
  }

  {
    ++total;
    const int M = 3;
    const int N = 5;
    const int K = 2;
    const std::vector<int8_t> A = {
        46, -43,
        42, 17,
        -31, -42,
    };
    const std::vector<int8_t> B = {
        -50, 3, 15, -6, 49,
        -13, 49, -33, -5, 2,
    };
    passed += run_case("ties_to_even_rounding_regression",
                       A,
                       B,
                       M,
                       N,
                       K,
                       0.05f,
                       0.1f,
                       0.01f,
                       0,
                       0,
                       0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 3;
    const int N = 5;
    const int K = 2;
    const std::vector<int8_t> A = {
        40, 33,
        45, -30,
        34, 39,
    };
    const std::vector<int8_t> B = {
        -20, 33, 36, -17, 2,
        40, 22, 43, -20, -4,
    };
    passed += run_case("scale_order_rounding_regression",
                       A,
                       B,
                       M,
                       N,
                       K,
                       0.05f,
                       0.1f,
                       0.01f,
                       0,
                       0,
                       0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 1;
    const int K = 1;
    const std::vector<int8_t> A = {5};
    const std::vector<int8_t> B = {1};
    passed += run_case("half_tie_positive_to_even_down", A, B, M, N, K, 0.5f, 1.0f, 1.0f, 0, 0, 0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 1;
    const int K = 1;
    const std::vector<int8_t> A = {7};
    const std::vector<int8_t> B = {1};
    passed += run_case("half_tie_positive_to_even_up", A, B, M, N, K, 0.5f, 1.0f, 1.0f, 0, 0, 0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 1;
    const int K = 1;
    const std::vector<int8_t> A = {-5};
    const std::vector<int8_t> B = {1};
    passed += run_case("half_tie_negative_to_even_up", A, B, M, N, K, 0.5f, 1.0f, 1.0f, 0, 0, 0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 1;
    const int K = 1;
    const std::vector<int8_t> A = {-7};
    const std::vector<int8_t> B = {1};
    passed += run_case("half_tie_negative_to_even_down", A, B, M, N, K, 0.5f, 1.0f, 1.0f, 0, 0, 0)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 37;
    const int N = 29;
    const int K = 41;
    const std::vector<int8_t> A = make_patterned_data(M, K, 17, 5, -3);
    const std::vector<int8_t> B = make_patterned_data(K, N, 19, 4, 2);
    passed += run_case("larger_patterned_case", A, B, M, N, K, 0.125f, 0.25f, 0.5f, -3, 4, -7)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int M = 128;
    const int N = 96;
    const int K = 80;
    const std::vector<int8_t> A = make_lcg_data(M * K, 11);
    const std::vector<int8_t> B = make_lcg_data(K * N, 29);
    passed += run_case("randomized_mid_shape", A, B, M, N, K, 0.05f, 0.1f, 0.02f, -7, 9, -3) ? 1
                                                                                                 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
  if (passed != total) {
    return 1;
  }

  run_bench_case("square_512", 512, 512, 512, 0.05f, 0.1f, 0.01f, 0, 0, 0, 5, 30);
  run_bench_case("target_shape", 8192, 4096, 2048, 0.05f, 0.1f, 0.01f, 0, 0, 0, 3, 10);
  return 0;
}
#endif
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

输出矩阵仍然用二维 grid/tile 覆盖。不同点在于内部乘加类型和 scale 地址。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程定位到 C 的某个 row/col 后，沿 K 维读取 int8 数据，累加到 int32，再写回反量化结果。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
int corrected = raw - zp_b * row_sum_a - zp_a * col_sum_b + K * zp_a * zp_b;
int q = nearbyintf(raw_or_corrected * scale_a * scale_b / scale_c) + zp_c;
```

这段代码我会这样读：

- 这段代码的主要价值是把题目的数学关系落成 CUDA 的线程分工和地址计算。

### 版本 2

先看代码：

```cpp
constexpr int kTile = 16;

__device__ __forceinline__ int8_t clamp_to_int8(int x) {
  return static_cast<int8_t>(max(-128, min(127, x)));
}

__global__ void scalar_kernel(const int8_t* A,
                              const int8_t* B,
                              int8_t* C,
                              int M,
                              int N,
                              int K,
                              float scale_A,
                              float scale_B,
                              float scale_C,
                              int zp_A,
                              int zp_B,
                              int zp_C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  int acc = 0;
  for (int k = 0; k < K; ++k) {
    int a = static_cast<int>(A[row * K + k]) - zp_A;
    int b = static_cast<int>(B[k * N + col]) - zp_B;
    acc += a * b;
  }

  float x = static_cast<float>(acc) * scale_A * scale_B / scale_C;
  C[row * N + col] = clamp_to_int8(static_cast<int>(nearbyintf(x)) + zp_C);
}

extern "C" void solve(...) {
  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  scalar_kernel<<<blocks, threads>>>(...);
}
```

这段代码我会这样读：

- 这里用了二维或三维 launch。相比一维数组题，多出来的维度不是装饰，而是在表达输出张量本身的行、列、head 或 channel。
- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程定位到 C 的某个 row/col 后，沿 K 维读取 int8 数据，累加到 int32，再写回反量化结果。

### 版本 3

先看代码：

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_int8_kernel(const int8_t* A,
                                 const int8_t* B,
                                 int8_t* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_A,
                                 float scale_B,
                                 float scale_C,
                                 int zp_C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  constexpr int WM = 16, WN = 16, WK = 16;
  int warp_id = threadIdx.x / 32;
  int row = blockIdx.y * 64 + (warp_id / 2) * WM;
  int col = blockIdx.x * 32 + (warp_id % 2) * WN;

  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
  wmma::fill_fragment(acc, 0);

  for (int kk = 0; kk < K; kk += WK) {
    wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::row_major> b;
    wmma::load_matrix_sync(a, reinterpret_cast<const signed char*>(A + row * K + kk), K);
    wmma::load_matrix_sync(b, reinterpret_cast<const signed char*>(B + kk * N + col), N);
    wmma::mma_sync(acc, a, b, acc);
  }

  // store acc -> int32 tile, then quantize to int8
#endif
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程定位到 C 的某个 row/col 后，沿 K 维读取 int8 数据，累加到 int32，再写回反量化结果。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

## 六、性能和可改方向

这题我会优先看这些点：

- 量化题的性能重点在打包读取和整数指令利用。
- scale 粒度会影响访存模式。
- 更高阶优化会走 Tensor Core integer 路线。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 累加不能用 int8。
- zero point、scale 的维度必须按题目定义。
- 有符号/无符号 int8 不要搞混。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **INT8 Quantized Matmul** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
