---
title: '批量归一化'
published: 2026-07-08
description: '批量归一化 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 中等题
  - 归一化
  - 深度学习算子
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/batch-normalization'
---

刷 LeetGPU 的时候，**批量归一化** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/batch-normalization](https://leetgpu.com/challenges/batch-normalization)

英文名可以先记一下：**Batch Normalization**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Batch Normalization

BatchNorm 推理态就是按通道读取 mean、var、gamma、beta，然后对每个元素做归一化。

这题的核心关系可以写成：`y = (x - mean[c]) * rsqrt(var[c] + eps) * gamma[c] + beta[c]`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
for each n, c, h, w:
    y = (x - mean[c]) * rsqrt(var[c] + eps) * gamma[c] + beta[c];
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

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kTileChannels = 32;
constexpr int kStatsThreadRows = 8;
constexpr int kStatsRowsPerBlock = 64;
constexpr int kNormThreadRows = 8;

float* g_sum_buffer = nullptr;
float* g_sumsq_buffer = nullptr;
float* g_mean_buffer = nullptr;
float* g_inv_std_buffer = nullptr;
int g_workspace_capacity = 0;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[kBlockSize / kWarpSize];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int warp_count = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp_id] = value;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (warp_id == 0) {
    block_sum = lane < warp_count ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
  }
  return block_sum;
}

bool ensure_workspace(int C) {
  if (g_workspace_capacity >= C && g_sum_buffer != nullptr) {
    g_sumsq_buffer = g_sum_buffer + g_workspace_capacity;
    g_mean_buffer = g_sumsq_buffer + g_workspace_capacity;
    g_inv_std_buffer = g_mean_buffer + g_workspace_capacity;
    return true;
  }

  if (g_sum_buffer != nullptr) {
    cudaFree(g_sum_buffer);
    g_sum_buffer = nullptr;
    g_sumsq_buffer = nullptr;
    g_mean_buffer = nullptr;
    g_inv_std_buffer = nullptr;
    g_workspace_capacity = 0;
  }

  float* workspace = nullptr;
  if (cudaMalloc(&workspace, static_cast<size_t>(4 * C) * sizeof(float)) != cudaSuccess) {
    return false;
  }

  g_sum_buffer = workspace;
  g_sumsq_buffer = g_sum_buffer + C;
  g_mean_buffer = g_sumsq_buffer + C;
  g_inv_std_buffer = g_mean_buffer + C;
  g_workspace_capacity = C;
  return true;
}

}  // namespace

__global__ void batch_norm_baseline_kernel(const float* input,
                                           const float* gamma,
                                           const float* beta,
                                           float* output,
                                           int N,
                                           int C,
                                           float eps) {
  __shared__ float mean;
  __shared__ float variance;

  const int channel = blockIdx.x;
  const int tid = threadIdx.x;

  float thread_sum = 0.0f;
  for (int row = tid; row < N; row += blockDim.x) {
    thread_sum += input[row * C + channel];
  }

  const float sum = block_reduce_sum(thread_sum);
  if (tid == 0) {
    mean = sum / static_cast<float>(N);
  }
  __syncthreads();

  float thread_sq_sum = 0.0f;
  for (int row = tid; row < N; row += blockDim.x) {
    const float diff = input[row * C + channel] - mean;
    thread_sq_sum += diff * diff;
  }

  const float sq_sum = block_reduce_sum(thread_sq_sum);
  if (tid == 0) {
    variance = sq_sum / static_cast<float>(N);
  }
  __syncthreads();

  const float inv_std = rsqrtf(variance + eps);
  const float channel_gamma = gamma[channel];
  const float channel_beta = beta[channel];
  for (int row = tid; row < N; row += blockDim.x) {
    const int index = row * C + channel;
    output[index] = channel_gamma * ((input[index] - mean) * inv_std) + channel_beta;
  }
}

template <int kThreads, int kUnroll>
__global__ void batch_norm_optimized_kernel(const float* input,
                                            const float* gamma,
                                            const float* beta,
                                            float* output,
                                            int N,
                                            int C,
                                            float eps) {
  __shared__ float mean;
  __shared__ float inv_std;

  const int channel = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_step = kThreads * kUnroll;

  float thread_sum = 0.0f;
  float thread_sumsq = 0.0f;
  int row = tid;
  for (; row + (kUnroll - 1) * kThreads < N; row += row_step) {
    #pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const float value = input[(row + i * kThreads) * C + channel];
      thread_sum += value;
      thread_sumsq += value * value;
    }
  }
  for (; row < N; row += kThreads) {
    const float value = input[row * C + channel];
    thread_sum += value;
    thread_sumsq += value * value;
  }

  const float sum = block_reduce_sum(thread_sum);
  const float sumsq = block_reduce_sum(thread_sumsq);
  if (tid == 0) {
    mean = sum / static_cast<float>(N);
    const float second_moment = sumsq / static_cast<float>(N);
    const float variance = fmaxf(second_moment - mean * mean, 0.0f);
    inv_std = rsqrtf(variance + eps);
  }
  __syncthreads();

  const float channel_gamma = gamma[channel];
  const float channel_beta = beta[channel];
  row = tid;
  for (; row + (kUnroll - 1) * kThreads < N; row += row_step) {
    #pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      const int index = (row + i * kThreads) * C + channel;
      output[index] = channel_gamma * ((input[index] - mean) * inv_std) + channel_beta;
    }
  }
  for (; row < N; row += kThreads) {
    const int index = row * C + channel;
    output[index] = channel_gamma * ((input[index] - mean) * inv_std) + channel_beta;
  }
}

__global__ void accumulate_stats_tiled_kernel(const float* input,
                                              float* sum,
                                              float* sumsq,
                                              int N,
                                              int C) {
  __shared__ float shared_sum[kStatsThreadRows][kTileChannels];
  __shared__ float shared_sumsq[kStatsThreadRows][kTileChannels];

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_block_begin = blockIdx.y * kStatsRowsPerBlock;
  const int row_block_end = min(row_block_begin + kStatsRowsPerBlock, N);

  float local_sum = 0.0f;
  float local_sumsq = 0.0f;
  if (channel < C) {
    for (int row = row_block_begin + threadIdx.y; row < row_block_end; row += blockDim.y) {
      const float value = input[row * C + channel];
      local_sum += value;
      local_sumsq += value * value;
    }
  }

  shared_sum[threadIdx.y][threadIdx.x] = local_sum;
  shared_sumsq[threadIdx.y][threadIdx.x] = local_sumsq;
  __syncthreads();

  if (threadIdx.y == 0 && channel < C) {
    float total_sum = 0.0f;
    float total_sumsq = 0.0f;
    #pragma unroll
    for (int lane = 0; lane < kStatsThreadRows; ++lane) {
      total_sum += shared_sum[lane][threadIdx.x];
      total_sumsq += shared_sumsq[lane][threadIdx.x];
    }
    atomicAdd(sum + channel, total_sum);
    atomicAdd(sumsq + channel, total_sumsq);
  }
}

__global__ void finalize_tiled_stats_kernel(const float* sum,
                                            const float* sumsq,
                                            float* mean,
                                            float* inv_std,
                                            int N,
                                            int C,
                                            float eps) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= C) {
    return;
  }

  const float mean_value = sum[channel] / static_cast<float>(N);
  const float second_moment = sumsq[channel] / static_cast<float>(N);
  const float variance = fmaxf(second_moment - mean_value * mean_value, 0.0f);
  mean[channel] = mean_value;
  inv_std[channel] = rsqrtf(variance + eps);
}

__global__ void normalize_tiled_kernel(const float* input,
                                       const float* gamma,
                                       const float* beta,
                                       const float* mean,
                                       const float* inv_std,
                                       float* output,
                                       int N,
                                       int C) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (channel >= C || row >= N) {
    return;
  }

  const int index = row * C + channel;
  const float normalized = (input[index] - mean[channel]) * inv_std[channel];
  output[index] = gamma[channel] * normalized + beta[channel];
}

void solve_baseline_impl(const float* input,
                         const float* gamma,
                         const float* beta,
                         float* output,
                         int N,
                         int C,
                         float eps) {
  if (N <= 0 || C <= 0) {
    return;
  }
  batch_norm_baseline_kernel<<<C, kBlockSize>>>(input, gamma, beta, output, N, C, eps);
}

void solve_candidate_impl(const float* input,
                          const float* gamma,
                          const float* beta,
                          float* output,
                          int N,
                          int C,
                          float eps) {
  if (N <= 0 || C <= 0) {
    return;
  }
  batch_norm_optimized_kernel<128, 8><<<C, 128>>>(input, gamma, beta, output, N, C, eps);
}

void solve_tiled_impl(const float* input,
                      const float* gamma,
                      const float* beta,
                      float* output,
                      int N,
                      int C,
                      float eps) {
  if (N <= 0 || C <= 0) {
    return;
  }
  if (!ensure_workspace(C)) {
    solve_candidate_impl(input, gamma, beta, output, N, C, eps);
    return;
  }

  cudaMemset(g_sum_buffer, 0, static_cast<size_t>(C) * sizeof(float));
  cudaMemset(g_sumsq_buffer, 0, static_cast<size_t>(C) * sizeof(float));
  accumulate_stats_tiled_kernel<<<dim3((C + kTileChannels - 1) / kTileChannels,
                                       (N + kStatsRowsPerBlock - 1) / kStatsRowsPerBlock),
                                dim3(kTileChannels, kStatsThreadRows)>>>(
      input, g_sum_buffer, g_sumsq_buffer, N, C);
  finalize_tiled_stats_kernel<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
      g_sum_buffer, g_sumsq_buffer, g_mean_buffer, g_inv_std_buffer, N, C, eps);
  normalize_tiled_kernel<<<dim3((C + kTileChannels - 1) / kTileChannels,
                                (N + kNormThreadRows - 1) / kNormThreadRows),
                           dim3(kTileChannels, kNormThreadRows)>>>(
      input, gamma, beta, g_mean_buffer, g_inv_std_buffer, output, N, C);
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
  solve_tiled_impl(input, gamma, beta, output, N, C, eps);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-4f;

std::vector<float> batch_norm_reference(const std::vector<float>& input,
                                        const std::vector<float>& gamma,
                                        const std::vector<float>& beta,
                                        int N,
                                        int C,
                                        float eps) {
  std::vector<float> mean(C, 0.0f);
  std::vector<float> var(C, 0.0f);
  std::vector<float> output(N * C, 0.0f);
  for (int j = 0; j < C; ++j) {
    for (int i = 0; i < N; ++i) {
      mean[j] += input[i * C + j];
    }
    mean[j] /= static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
      const float diff = input[i * C + j] - mean[j];
      var[j] += diff * diff;
    }
    var[j] /= static_cast<float>(N);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      const float centered = input[i * C + j] - mean[j];
      const float normalized = centered / std::sqrt(var[j] + eps);
      output[i * C + j] = gamma[j] * normalized + beta[j];
    }
  }
  return output;
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kEps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

std::vector<float> make_patterned_data(int n, int period, float scale, float bias) {
  std::vector<float> values(n);
  for (int i = 0; i < n; ++i) {
    const int centered = (i % period) - (period / 2);
    values[i] = centered * scale + bias;
  }
  return values;
}

bool run_case(const std::string& name,
              const std::vector<float>& input,
              const std::vector<float>& gamma,
              const std::vector<float>& beta,
              int N,
              int C,
              float eps) {
  const size_t input_bytes = static_cast<size_t>(N) * static_cast<size_t>(C) * sizeof(float);
  const size_t param_bytes = static_cast<size_t>(C) * sizeof(float);
  const std::vector<float> expected = batch_norm_reference(input, gamma, beta, N, C, eps);
  std::vector<float> actual(N * C, 0.0f);

  float* d_input = nullptr;
  float* d_gamma = nullptr;
  float* d_beta = nullptr;
  float* d_output = nullptr;
  auto cleanup = [&]() {
    if (d_input != nullptr) cudaFree(d_input);
    if (d_gamma != nullptr) cudaFree(d_gamma);
    if (d_beta != nullptr) cudaFree(d_beta);
    if (d_output != nullptr) cudaFree(d_output);
  };

  if (cudaMalloc(&d_input, input_bytes) != cudaSuccess ||
      cudaMalloc(&d_gamma, param_bytes) != cudaSuccess ||
      cudaMalloc(&d_beta, param_bytes) != cudaSuccess ||
      cudaMalloc(&d_output, input_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }
  if (cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_gamma, gamma.data(), param_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_beta, beta.data(), param_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemset(d_output, 0, input_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": device upload failed\n";
    cleanup();
    return false;
  }

  solve(d_input, d_gamma, d_beta, d_output, N, C, eps);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_output, input_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution failed\n";
    cleanup();
    return false;
  }

  for (int i = 0; i < N * C; ++i) {
    if (!almost_equal(actual[i], expected[i])) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << i
                << ", expected=" << expected[i] << ", actual=" << actual[i] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

float time_solve(bool optimized,
                 const float* d_input,
                 const float* d_gamma,
                 const float* d_beta,
                 float* d_output,
                 int N,
                 int C,
                 int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    if (optimized) {
      solve(d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
    } else {
      solve_baseline_impl(d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
    }
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    if (optimized) {
      solve(d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
    } else {
      solve_baseline_impl(d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
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

float time_tiled_variant(const float* d_input,
                         const float* d_gamma,
                         const float* d_beta,
                         float* d_output,
                         int N,
                         int C,
                         int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    solve_tiled_impl(d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    solve_tiled_impl(d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iterations);
}

template <int kThreads, int kUnroll>
float time_candidate_variant(const float* d_input,
                             const float* d_gamma,
                             const float* d_beta,
                             float* d_output,
                             int N,
                             int C,
                             int iterations) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; ++i) {
    batch_norm_optimized_kernel<kThreads, kUnroll><<<C, kThreads>>>(
        d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i) {
    batch_norm_optimized_kernel<kThreads, kUnroll><<<C, kThreads>>>(
        d_input, d_gamma, d_beta, d_output, N, C, 1e-5f);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / static_cast<float>(iterations);
}

bool run_benchmark_case(int N, int C) {
  const size_t input_bytes = static_cast<size_t>(N) * static_cast<size_t>(C) * sizeof(float);
  const size_t param_bytes = static_cast<size_t>(C) * sizeof(float);
  const std::vector<float> input = make_patterned_data(N * C, 251, 0.03125f, -0.5f);
  const std::vector<float> gamma = make_patterned_data(C, 67, 0.01f, 1.25f);
  const std::vector<float> beta = make_patterned_data(C, 71, 0.02f, -0.25f);

  float* d_input = nullptr;
  float* d_gamma = nullptr;
  float* d_beta = nullptr;
  float* d_output = nullptr;
  if (cudaMalloc(&d_input, input_bytes) != cudaSuccess ||
      cudaMalloc(&d_gamma, param_bytes) != cudaSuccess ||
      cudaMalloc(&d_beta, param_bytes) != cudaSuccess ||
      cudaMalloc(&d_output, input_bytes) != cudaSuccess) {
    std::cerr << "[BENCH] allocation failed for N=" << N << ", C=" << C << '\n';
    return false;
  }

  cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma.data(), param_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta.data(), param_bytes, cudaMemcpyHostToDevice);

  const int iterations = C <= 64 ? 500 : 200;
  const float baseline_ms = time_solve(false, d_input, d_gamma, d_beta, d_output, N, C, iterations);
  const float optimized_ms = time_solve(true, d_input, d_gamma, d_beta, d_output, N, C, iterations);
  const float tiled_ms = time_tiled_variant(d_input, d_gamma, d_beta, d_output, N, C, iterations);
  const float t128_u4 = time_candidate_variant<128, 4>(d_input, d_gamma, d_beta, d_output, N, C, iterations);
  const float t128_u8 = time_candidate_variant<128, 8>(d_input, d_gamma, d_beta, d_output, N, C, iterations);
  const float t256_u4 = time_candidate_variant<256, 4>(d_input, d_gamma, d_beta, d_output, N, C, iterations);
  const float t256_u8 = time_candidate_variant<256, 8>(d_input, d_gamma, d_beta, d_output, N, C, iterations);

  std::cout << "[BENCH] N=" << std::setw(5) << N << " C=" << std::setw(4) << C
            << " baseline=" << std::fixed << std::setprecision(4) << baseline_ms << " ms"
            << " optimized=" << optimized_ms << " ms"
            << " speedup=" << baseline_ms / optimized_ms << "x\n";
  std::cout << "        tiled=" << tiled_ms << " speedup=" << baseline_ms / tiled_ms << "x\n";
  std::cout << "        variants: 128x4=" << t128_u4
            << " 128x8=" << t128_u8
            << " 256x4=" << t256_u4
            << " 256x8=" << t256_u8 << '\n';

  cudaFree(d_input);
  cudaFree(d_gamma);
  cudaFree(d_beta);
  cudaFree(d_output);
  return true;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int N = 3;
    const int C = 2;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> gamma = {1.0f, 1.0f};
    const std::vector<float> beta = {0.0f, 0.0f};
    passed += run_case("sample_case_unit_scale_shift", input, gamma, beta, N, C, 1e-5f) ? 1 : 0;
  }

  {
    ++total;
    const int N = 2;
    const int C = 2;
    const std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> gamma = {2.0f, 0.5f};
    const std::vector<float> beta = {1.0f, -1.0f};
    passed += run_case("sample_case_affine_transform", input, gamma, beta, N, C, 1e-5f) ? 1 : 0;
  }

  {
    ++total;
    const int N = 1;
    const int C = 1;
    const std::vector<float> input = {7.0f};
    const std::vector<float> gamma = {0.5f};
    const std::vector<float> beta = {-2.0f};
    passed += run_case("minimum_size_single_feature", input, gamma, beta, N, C, 1e-5f) ? 1 : 0;
  }

  {
    ++total;
    const int N = 2;
    const int C = 3;
    const std::vector<float> input = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f};
    const std::vector<float> gamma = {1.0f, 1.0f, 1.0f};
    const std::vector<float> beta = {0.0f, 0.0f, 0.0f};
    passed += run_case("two_rows_three_channels_negative_values", input, gamma, beta, N, C, 1e-5f)
                  ? 1
                  : 0;
  }

  {
    ++total;
    const int N = 64;
    const int C = 17;
    const std::vector<float> input = make_patterned_data(N * C, 29, 0.125f, -0.75f);
    const std::vector<float> gamma = make_patterned_data(C, 11, 0.05f, 1.5f);
    const std::vector<float> beta = make_patterned_data(C, 13, 0.1f, -0.5f);
    passed += run_case("larger_patterned_case", input, gamma, beta, N, C, 1e-5f) ? 1 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
  if (passed == total) {
    run_benchmark_case(5000, 32);
    run_benchmark_case(5000, 128);
    run_benchmark_case(5000, 512);
    run_benchmark_case(5000, 1024);
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

通常把 NCHW 张量摊平成一维数组，grid 覆盖所有元素。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

从 flat index 还原 channel 是关键。NCHW 中 `c = (idx / (H * W)) % C`。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
__global__ void batch_norm_baseline_kernel(const float* input,
                                           const float* gamma,
                                           const float* beta,
                                           float* output,
                                           int N,
                                           int C,
                                           float eps) {
  __shared__ float mean;
  __shared__ float variance;

  int channel = blockIdx.x;
  int tid = threadIdx.x;

  float thread_sum = 0.0f;
  for (int row = tid; row < N; row += blockDim.x) {
    thread_sum += input[row * C + channel];
  }

  float sum = block_reduce_sum(thread_sum);
  if (tid == 0) mean = sum / float(N);
  __syncthreads();

  float thread_sq_sum = 0.0f;
  for (int row = tid; row < N; row += blockDim.x) {
    float diff = input[row * C + channel] - mean;
    thread_sq_sum += diff * diff;
  }

  float sq_sum = block_reduce_sum(thread_sq_sum);
  if (tid == 0) variance = sq_sum / float(N);
  __syncthreads();

  float inv_std = rsqrtf(variance + eps);
  for (int row = tid; row < N; row += blockDim.x) {
    int index = row * C + channel;
    output[index] = gamma[channel] * ((input[index] - mean) * inv_std) + beta[channel];
  }
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。从 flat index 还原 channel 是关键。NCHW 中 `c = (idx / (H * W)) % C`。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。

### 版本 2

先看代码：

```cpp
__global__ void accumulate_stats_tiled_kernel(const float* input,
                                              float* sum,
                                              float* sumsq,
                                              int N,
                                              int C) {
  __shared__ float shared_sum[kStatsThreadRows][kTileChannels];
  __shared__ float shared_sumsq[kStatsThreadRows][kTileChannels];

  int channel = blockIdx.x * blockDim.x + threadIdx.x;
  int row_begin = blockIdx.y * kStatsRowsPerBlock;
  int row_end = min(row_begin + kStatsRowsPerBlock, N);

  float local_sum = 0.0f;
  float local_sumsq = 0.0f;
  if (channel < C) {
    for (int row = row_begin + threadIdx.y; row < row_end; row += blockDim.y) {
      float v = input[row * C + channel];
      local_sum += v;
      local_sumsq += v * v;
    }
  }

  shared_sum[threadIdx.y][threadIdx.x] = local_sum;
  shared_sumsq[threadIdx.y][threadIdx.x] = local_sumsq;
  __syncthreads();

  if (threadIdx.y == 0 && channel < C) {
    float total_sum = 0.0f, total_sumsq = 0.0f;
    for (int i = 0; i < kStatsThreadRows; ++i) {
      total_sum += shared_sum[i][threadIdx.x];
      total_sumsq += shared_sumsq[i][threadIdx.x];
    }
    atomicAdd(sum + channel, total_sum);
    atomicAdd(sumsq + channel, total_sumsq);
  }
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。从 flat index 还原 channel 是关键。NCHW 中 `c = (idx / (H * W)) % C`。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。

## 六、性能和可改方向

这题我会优先看这些点：

- 统计量按 channel 复用，访问 gamma/beta/mean/var 时 cache 友好。
- `rsqrtf` 是常见写法。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- epsilon 加在 var 上再开方。
- NCHW 和 NHWC 的 channel 位置完全不同。
- 不要把 batch 维当 channel。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Batch Normalization** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
