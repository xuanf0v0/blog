---
title: '权重量化反解'
published: 2026-07-08
description: '权重量化反解 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 中等题
  - 量化
  - 深度学习算子
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/weight-dequantization'
---

刷 LeetGPU 的时候，**权重量化反解** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/weight-dequantization](https://leetgpu.com/challenges/weight-dequantization)

英文名可以先记一下：**Weight Dequantization**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Weight Dequantization

权重量化反解是把压缩/量化后的权重还原成计算用的浮点值。LLM 推理里这类 kernel 很常见。

这题的核心关系可以写成：`float_w = (q - zero) * scale`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
for (int i = 0; i < N; ++i) {
    output[i] = (float(input[i]) - zero) * scale;
}
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>

#ifndef ONLINE_JUDGE
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#endif

namespace {

constexpr int kBaselineBlockEdge = 16;
constexpr int kScalarBlockSize = 256;
constexpr int kVectorBlockSize = 256;

int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

template <int kTileSize>
__global__ void weight_dequant_vec4_kernel(const float* X, const float* S, float* Y, int M, int N) {
  const int row = blockIdx.y;
  const int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = vec_index * 4;
  if (row >= M || col >= N) {
    return;
  }

  constexpr int kTileShift = kTileSize == 16 ? 4 : kTileSize == 32 ? 5 : kTileSize == 64 ? 6 : 7;
  const int scale_cols = (N + kTileSize - 1) >> kTileShift;
  const float scale = S[(row >> kTileShift) * scale_cols + (col >> kTileShift)];

  const float4 values = reinterpret_cast<const float4*>(X + row * N + col)[0];
  float4 result;
  result.x = values.x * scale;
  result.y = values.y * scale;
  result.z = values.z * scale;
  result.w = values.w * scale;
  reinterpret_cast<float4*>(Y + row * N + col)[0] = result;
}

template <int kTileSize>
__global__ void weight_dequant_tile_vec4_kernel(const float* X, const float* S, float* Y, int M, int N) {
  constexpr int kBlockRows = 8;
  constexpr int kVecWidth = 4;
  constexpr int kTileShift = kTileSize == 16 ? 4 : kTileSize == 32 ? 5 : kTileSize == 64 ? 6 : 7;

  const int tile_col = blockIdx.x;
  const int row = blockIdx.y * kBlockRows + threadIdx.y;
  const int col = tile_col * kTileSize + threadIdx.x * kVecWidth;
  if (row >= M || col >= N) {
    return;
  }

  const int scale_cols = (N + kTileSize - 1) >> kTileShift;
  const float scale = S[(row >> kTileShift) * scale_cols + tile_col];
  const float4 values = reinterpret_cast<const float4*>(X + row * N + col)[0];
  float4 result;
  result.x = values.x * scale;
  result.y = values.y * scale;
  result.z = values.z * scale;
  result.w = values.w * scale;
  reinterpret_cast<float4*>(Y + row * N + col)[0] = result;
}

template <int kTileSize>
__global__ void weight_dequant_vec4x2_kernel(const float* X, const float* S, float* Y, int M, int N) {
  const int row = blockIdx.y;
  const int vec_index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const int col0 = vec_index * 4;
  const int col1 = col0 + 4;
  if (row >= M || col0 >= N) {
    return;
  }

  constexpr int kTileShift = kTileSize == 16 ? 4 : kTileSize == 32 ? 5 : kTileSize == 64 ? 6 : 7;
  const int scale_cols = (N + kTileSize - 1) >> kTileShift;
  const int scale_row = row >> kTileShift;

  const float scale0 = S[scale_row * scale_cols + (col0 >> kTileShift)];
  const float4 values0 = reinterpret_cast<const float4*>(X + row * N + col0)[0];
  float4 result0;
  result0.x = values0.x * scale0;
  result0.y = values0.y * scale0;
  result0.z = values0.z * scale0;
  result0.w = values0.w * scale0;
  reinterpret_cast<float4*>(Y + row * N + col0)[0] = result0;

  if (col1 < N) {
    const float scale1 = S[scale_row * scale_cols + (col1 >> kTileShift)];
    const float4 values1 = reinterpret_cast<const float4*>(X + row * N + col1)[0];
    float4 result1;
    result1.x = values1.x * scale1;
    result1.y = values1.y * scale1;
    result1.z = values1.z * scale1;
    result1.w = values1.w * scale1;
    reinterpret_cast<float4*>(Y + row * N + col1)[0] = result1;
  }
}

__global__ void weight_dequant_scalar_kernel(const float* X,
                                             const float* S,
                                             float* Y,
                                             int M,
                                             int N,
                                             int tile_size) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  const int scale_cols = (N + tile_size - 1) / tile_size;
  Y[row * N + col] = X[row * N + col] * S[(row / tile_size) * scale_cols + (col / tile_size)];
}

__global__ void weight_dequant_baseline_kernel(const float* X,
                                               const float* S,
                                               float* Y,
                                               int M,
                                               int N,
                                               int tile_size) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bxo = blockIdx.x * tile_size;
  const int byo = blockIdx.y * tile_size;
  const int scale_cols = (N + tile_size - 1) / tile_size;

  for (int yoffset = 0; yoffset < tile_size; yoffset += kBaselineBlockEdge) {
    for (int xoffset = 0; xoffset < tile_size; xoffset += kBaselineBlockEdge) {
      const int xo = xoffset + bxo + tx;
      const int yo = yoffset + byo + ty;
      if (xo < N && yo < M) {
        Y[yo * N + xo] = X[yo * N + xo] * S[(yo / tile_size) * scale_cols + (xo / tile_size)];
      }
    }
  }
}

void solve_baseline_impl(const float* X, const float* S, float* Y, int M, int N, int tile_size) {
  weight_dequant_baseline_kernel<<<dim3(ceil_div(N, tile_size), ceil_div(M, tile_size)),
                                   dim3(kBaselineBlockEdge, kBaselineBlockEdge)>>>(
      X, S, Y, M, N, tile_size);
}

void solve_candidate_impl(const float* X, const float* S, float* Y, int M, int N, int tile_size) {
  if ((N & 3) == 0 && tile_size >= 64) {
    switch (tile_size) {
      case 64:
        weight_dequant_tile_vec4_kernel<64><<<dim3(ceil_div(N, 64), ceil_div(M, 8)), dim3(16, 8)>>>(
            X, S, Y, M, N);
        return;
      case 128:
        weight_dequant_tile_vec4_kernel<128><<<dim3(ceil_div(N, 128), ceil_div(M, 8)),
                                               dim3(32, 8)>>>(X, S, Y, M, N);
        return;
      default:
        break;
    }
  }

  if ((N & 3) == 0) {
    const dim3 block(kVectorBlockSize);
    const dim3 grid(ceil_div(N / 4, kVectorBlockSize), M);
    switch (tile_size) {
      case 16:
        weight_dequant_vec4_kernel<16><<<grid, block>>>(X, S, Y, M, N);
        return;
      case 32:
        weight_dequant_vec4_kernel<32><<<grid, block>>>(X, S, Y, M, N);
        return;
      case 64:
        weight_dequant_vec4_kernel<64><<<grid, block>>>(X, S, Y, M, N);
        return;
      case 128:
        weight_dequant_vec4_kernel<128><<<grid, block>>>(X, S, Y, M, N);
        return;
      default:
        break;
    }
  }

  weight_dequant_scalar_kernel<<<dim3(ceil_div(N, kScalarBlockSize), M), kScalarBlockSize>>>(
      X, S, Y, M, N, tile_size);
}

void solve_candidate_variant_impl(const float* X,
                                  const float* S,
                                  float* Y,
                                  int M,
                                  int N,
                                  int tile_size,
                                  int variant_id) {
  switch (variant_id) {
    case 0:
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    case 1:
      if ((N & 3) == 0) {
        const dim3 block(128);
        const dim3 grid(ceil_div(N / 8, 128), M);
        switch (tile_size) {
          case 16:
            weight_dequant_vec4x2_kernel<16><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 32:
            weight_dequant_vec4x2_kernel<32><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 64:
            weight_dequant_vec4x2_kernel<64><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 128:
            weight_dequant_vec4x2_kernel<128><<<grid, block>>>(X, S, Y, M, N);
            return;
          default:
            break;
        }
      }
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    case 2:
      if ((N & 3) == 0) {
        const dim3 block(128);
        const dim3 grid(ceil_div(N / 4, 128), M);
        switch (tile_size) {
          case 16:
            weight_dequant_vec4_kernel<16><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 32:
            weight_dequant_vec4_kernel<32><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 64:
            weight_dequant_vec4_kernel<64><<<grid, block>>>(X, S, Y, M, N);
            return;
          case 128:
            weight_dequant_vec4_kernel<128><<<grid, block>>>(X, S, Y, M, N);
            return;
          default:
            break;
        }
      }
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    case 3:
      if ((N & 3) == 0) {
        switch (tile_size) {
          case 16:
            weight_dequant_tile_vec4_kernel<16><<<dim3(ceil_div(N, 16), ceil_div(M, 8)),
                                                  dim3(4, 8)>>>(X, S, Y, M, N);
            return;
          case 32:
            weight_dequant_tile_vec4_kernel<32><<<dim3(ceil_div(N, 32), ceil_div(M, 8)),
                                                  dim3(8, 8)>>>(X, S, Y, M, N);
            return;
          case 64:
            weight_dequant_tile_vec4_kernel<64><<<dim3(ceil_div(N, 64), ceil_div(M, 8)),
                                                  dim3(16, 8)>>>(X, S, Y, M, N);
            return;
          case 128:
            weight_dequant_tile_vec4_kernel<128><<<dim3(ceil_div(N, 128), ceil_div(M, 8)),
                                                   dim3(32, 8)>>>(X, S, Y, M, N);
            return;
          default:
            break;
        }
      }
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
    default:
      solve_candidate_impl(X, S, Y, M, N, tile_size);
      return;
  }
}

}  // namespace

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
  solve_candidate_impl(X, S, Y, M, N, TILE_SIZE);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-5f;

std::vector<float> dequant_reference(const std::vector<float>& X,
                                     const std::vector<float>& S,
                                     int M,
                                     int N,
                                     int tile_size) {
  const int scale_cols = ceil_div(N, tile_size);
  std::vector<float> Y(M * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      const int scale_row = i / tile_size;
      const int scale_col = j / tile_size;
      Y[i * N + j] = X[i * N + j] * S[scale_row * scale_cols + scale_col];
    }
  }
  return Y;
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kEps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

bool run_case(const std::string& name,
              const std::vector<float>& X,
              const std::vector<float>& S,
              int M,
              int N,
              int tile_size) {
  const size_t matrix_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  const size_t scale_bytes =
      static_cast<size_t>(ceil_div(M, tile_size)) * static_cast<size_t>(ceil_div(N, tile_size)) *
      sizeof(float);
  const std::vector<float> expected = dequant_reference(X, S, M, N, tile_size);
  std::vector<float> actual(M * N, 0.0f);

  float* d_X = nullptr;
  float* d_S = nullptr;
  float* d_Y = nullptr;
  auto cleanup = [&]() {
    if (d_X != nullptr) cudaFree(d_X);
    if (d_S != nullptr) cudaFree(d_S);
    if (d_Y != nullptr) cudaFree(d_Y);
  };

  if (cudaMalloc(&d_X, matrix_bytes) != cudaSuccess ||
      cudaMalloc(&d_S, scale_bytes) != cudaSuccess ||
      cudaMalloc(&d_Y, matrix_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_X, X.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_S, S.data(), scale_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemset(d_Y, 0, matrix_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": device upload failed\n";
    cleanup();
    return false;
  }

  solve(d_X, d_S, d_Y, M, N, tile_size);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_Y, matrix_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution failed\n";
    cleanup();
    return false;
  }

  for (int index = 0; index < M * N; ++index) {
    if (!almost_equal(actual[index], expected[index])) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << index
                << ", expected=" << expected[index] << ", actual=" << actual[index] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

float time_impl(bool use_candidate,
                const float* d_X,
                const float* d_S,
                float* d_Y,
                int M,
                int N,
                int tile_size,
                int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    if (use_candidate) {
      solve_candidate_impl(d_X, d_S, d_Y, M, N, tile_size);
    } else {
      solve_baseline_impl(d_X, d_S, d_Y, M, N, tile_size);
    }
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    if (use_candidate) {
      solve_candidate_impl(d_X, d_S, d_Y, M, N, tile_size);
    } else {
      solve_baseline_impl(d_X, d_S, d_Y, M, N, tile_size);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iterations);
}

float time_variant(int variant_id,
                   const float* d_X,
                   const float* d_S,
                   float* d_Y,
                   int M,
                   int N,
                   int tile_size,
                   int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    solve_candidate_variant_impl(d_X, d_S, d_Y, M, N, tile_size, variant_id);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    solve_candidate_variant_impl(d_X, d_S, d_Y, M, N, tile_size, variant_id);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iterations);
}

bool run_benchmark_case(int M, int N, int tile_size) {
  const size_t matrix_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  const int scale_rows = ceil_div(M, tile_size);
  const int scale_cols = ceil_div(N, tile_size);
  const size_t scale_bytes = static_cast<size_t>(scale_rows) * static_cast<size_t>(scale_cols) *
                             sizeof(float);

  std::vector<float> X(M * N);
  std::vector<float> S(scale_rows * scale_cols);
  for (int i = 0; i < M * N; ++i) {
    X[i] = static_cast<float>((i % 127) - 63) * 0.125f;
  }
  for (int i = 0; i < scale_rows * scale_cols; ++i) {
    S[i] = 0.25f + static_cast<float>(i % 11) * 0.125f;
  }

  float* d_X = nullptr;
  float* d_S = nullptr;
  float* d_Y = nullptr;
  auto cleanup = [&]() {
    if (d_X != nullptr) cudaFree(d_X);
    if (d_S != nullptr) cudaFree(d_S);
    if (d_Y != nullptr) cudaFree(d_Y);
  };

  if (cudaMalloc(&d_X, matrix_bytes) != cudaSuccess ||
      cudaMalloc(&d_S, scale_bytes) != cudaSuccess ||
      cudaMalloc(&d_Y, matrix_bytes) != cudaSuccess) {
    std::cerr << "[BENCH] allocation failed for M=" << M << ", N=" << N
              << ", TILE_SIZE=" << tile_size << '\n';
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_X, X.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_S, S.data(), scale_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[BENCH] device upload failed for M=" << M << ", N=" << N
              << ", TILE_SIZE=" << tile_size << '\n';
    cleanup();
    return false;
  }

  const int iterations = M >= 4096 ? 50 : 200;
  const float baseline_ms = time_impl(false, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float optimized_ms = time_impl(true, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float variant_vec4x2_ms = time_variant(1, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float variant_vec4_128_ms = time_variant(2, d_X, d_S, d_Y, M, N, tile_size, iterations);
  const float variant_tile_ms = time_variant(3, d_X, d_S, d_Y, M, N, tile_size, iterations);
  std::cout << "[BENCH] M=" << std::setw(5) << M
            << " N=" << std::setw(5) << N
            << " TILE_SIZE=" << std::setw(3) << tile_size
            << " baseline=" << std::fixed << std::setprecision(4) << baseline_ms << " ms"
            << " optimized=" << optimized_ms << " ms"
            << " speedup=" << baseline_ms / optimized_ms << "x\n";
  std::cout << "        variants: vec4x1-256=" << optimized_ms
            << " vec4x2-128=" << variant_vec4x2_ms
            << " vec4x1-128=" << variant_vec4_128_ms
            << " tile-vec4=" << variant_tile_ms << '\n';

  cleanup();
  return true;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int M = 4;
    const int N = 4;
    const int tile_size = 2;
    const std::vector<float> X = {
        10.0f, 10.0f, 5.0f, 5.0f,
        10.0f, 10.0f, 5.0f, 5.0f,
        2.0f,  2.0f,  8.0f, 8.0f,
        2.0f,  2.0f,  8.0f, 8.0f,
    };
    const std::vector<float> S = {
        0.5f, 2.0f,
        4.0f, 0.25f,
    };
    passed += run_case("sample_case_prompt_matrix_tiles", X, S, M, N, tile_size) ? 1 : 0;
  }

  {
    ++total;
    const int M = 3;
    const int N = 5;
    const int tile_size = 2;
    const std::vector<float> X = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    };
    const std::vector<float> S = {
        1.0f, 0.5f, -1.0f,
        2.0f, -0.25f, 4.0f,
    };
    passed += run_case("partial_tiles_non_multiple_dimensions", X, S, M, N, tile_size) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int N = 3;
    const int tile_size = 16;
    const std::vector<float> X = {3.0f, -2.0f, 7.5f};
    const std::vector<float> S = {0.25f};
    passed += run_case("single_row_tile_larger_than_matrix", X, S, M, N, tile_size) ? 1 : 0;
  }

  {
    ++total;
    const int M = 5;
    const int N = 4;
    const int tile_size = 4;
    const std::vector<float> X = {
        -1.0f, -2.0f, -3.0f, -4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        -9.0f, 10.0f, -11.0f, 12.0f,
        13.0f, -14.0f, 15.0f, -16.0f,
        2.0f,  4.0f,  6.0f,  8.0f,
    };
    const std::vector<float> S = {
        -0.5f,
        2.0f,
    };
    passed += run_case("mixed_signs_across_row_tiles", X, S, M, N, tile_size) ? 1 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
  if (passed == total) {
    run_benchmark_case(1024, 1024, 16);
    run_benchmark_case(2048, 2048, 32);
    run_benchmark_case(4096, 4096, 64);
    run_benchmark_case(8192, 8192, 128);
  }
  return passed == total ? 0 : 1;
}
#endif
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

一维 grid 覆盖量化权重元素。如果是 4-bit 打包，一个线程可能要从一个 byte 中拆两个值。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程根据 idx 找到量化值和对应 scale。scale 可能是 per tensor、per channel 或 per group。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
template <int kTileSize>
__global__ void weight_dequant_tile_vec4_kernel(const float* X, const float* S, float* Y, int M, int N) {
  constexpr int kBlockRows = 8;
  constexpr int kVecWidth = 4;

  int tile_col = blockIdx.x;
  int row = blockIdx.y * kBlockRows + threadIdx.y;
  int col = tile_col * kTileSize + threadIdx.x * kVecWidth;
  if (row >= M || col >= N) return;

  int scale_cols = (N + kTileSize - 1) / kTileSize;
  float scale = S[(row / kTileSize) * scale_cols + tile_col];
  float4 v = reinterpret_cast<const float4*>(X + row * N + col)[0];
  reinterpret_cast<float4*>(Y + row * N + col)[0] =
      make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程根据 idx 找到量化值和对应 scale。scale 可能是 per tensor、per channel 或 per group。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

## 六、性能和可改方向

这题我会优先看这些点：

- 性能重点在位运算解包和连续写出。
- scale 复用粒度越大，cache 越容易受益。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- int8/uint8 解释不同。
- 4-bit 高低位不要拿反。
- scale group 的地址公式要和题目一致。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Weight Dequantization** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
