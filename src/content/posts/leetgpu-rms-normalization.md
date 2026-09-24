---
title: 'RMS 归一化'
published: 2026-07-08
description: 'RMS 归一化 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
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
sourceLink: 'https://leetgpu.com/challenges/rms-normalization'
---

刷 LeetGPU 的时候，**RMS 归一化** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/rms-normalization](https://leetgpu.com/challenges/rms-normalization)

英文名可以先记一下：**RMS Normalization**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、RMS Normalization

RMSNorm 和 LayerNorm 很像，但它不减均值，只按均方根缩放。LLM 里很常见。

这题的核心关系可以写成：`y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
float ss = 0.0f;
for (int i = 0; i < hidden; ++i) ss += x[i] * x[i];
float scale = rsqrtf(ss / hidden + eps);
for (int i = 0; i < hidden; ++i) y[i] = x[i] * scale * weight[i];
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace cg = cooperative_groups;

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kVecWidth = 4;
constexpr int kMaxFallbackBlocks = 128;

__device__ float g_sum_sq;
__device__ float g_scale;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int kThreads>
__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[kThreads / kWarpSize];

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    value = (threadIdx.x < kThreads / kWarpSize) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

template <int kThreads>
__launch_bounds__(kThreads)
__global__ void rms_cooperative_kernel(const float* __restrict__ input,
                                       float gamma,
                                       float beta,
                                       float* __restrict__ output,
                                       int N,
                                       float eps) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int vec_count = N / kVecWidth;

    if (blockIdx.x == 0 && tid == 0) {
        g_sum_sq = 0.0f;
    }
    grid.sync();

    float local_sum = 0.0f;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        const float value = input[idx];
        local_sum += value * value;
    }

    const float block_sum = block_reduce_sum<kThreads>(local_sum);
    if (tid == 0) {
        atomicAdd(&g_sum_sq, block_sum);
    }
    grid.sync();

    if (blockIdx.x == 0 && tid == 0) {
        g_scale = rsqrtf(g_sum_sq / static_cast<float>(N) + eps) * gamma;
    }
    grid.sync();

    const float scale = g_scale;
    float4* output4 = reinterpret_cast<float4*>(output);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        output4[vec_idx] = make_float4(
            fmaf(v.x, scale, beta),
            fmaf(v.y, scale, beta),
            fmaf(v.z, scale, beta),
            fmaf(v.w, scale, beta));
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        output[idx] = fmaf(input[idx], scale, beta);
    }
}

template <int kThreads>
__global__ void rms_sum_kernel(const float* __restrict__ input, int N) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_count = N / kVecWidth;

    float local_sum = 0.0f;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        local_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        const float value = input[idx];
        local_sum += value * value;
    }

    const float block_sum = block_reduce_sum<kThreads>(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&g_sum_sq, block_sum);
    }
}

__global__ void rms_finalize_scale_kernel(int N, float gamma, float eps) {
    g_scale = rsqrtf(g_sum_sq / static_cast<float>(N) + eps) * gamma;
}

template <int kThreads>
__global__ void rms_normalize_kernel(const float* __restrict__ input,
                                     float beta,
                                     float* __restrict__ output,
                                     int N) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_count = N / kVecWidth;
    const float scale = g_scale;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    for (int vec_idx = global_tid; vec_idx < vec_count; vec_idx += stride) {
        const float4 v = input4[vec_idx];
        output4[vec_idx] = make_float4(
            fmaf(v.x, scale, beta),
            fmaf(v.y, scale, beta),
            fmaf(v.z, scale, beta),
            fmaf(v.w, scale, beta));
    }

    for (int idx = vec_count * kVecWidth + global_tid; idx < N; idx += stride) {
        output[idx] = fmaf(input[idx], scale, beta);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    if (N <= 0) {
        return;
    }

    int blocks = (N + kBlockSize - 1) / kBlockSize;
    if (blocks < 1) {
        blocks = 1;
    }

    int sm_count = 0;
    int cooperative_supported = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&cooperative_supported, cudaDevAttrCooperativeLaunch, 0);

    if (cooperative_supported) {
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, rms_cooperative_kernel<kBlockSize>, kBlockSize, 0);
        const int cooperative_limit = sm_count * max_blocks_per_sm;
        if (cooperative_limit > 0) {
            if (blocks > cooperative_limit) {
                blocks = cooperative_limit;
            }
            void* args[] = {const_cast<float**>(&input), &gamma, &beta, &output, &N, &eps};
            if (cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(rms_cooperative_kernel<kBlockSize>),
                    blocks,
                    kBlockSize,
                    args,
                    0,
                    nullptr) == cudaSuccess) {
                return;
            }
            cudaGetLastError();
        }
    }

    if (blocks > kMaxFallbackBlocks) {
        blocks = kMaxFallbackBlocks;
    }
    float zero = 0.0f;
    cudaMemcpyToSymbol(g_sum_sq, &zero, sizeof(float));
    rms_sum_kernel<kBlockSize><<<blocks, kBlockSize>>>(input, N);
    rms_finalize_scale_kernel<<<1, 1>>>(N, gamma, eps);
    rms_normalize_kernel<kBlockSize><<<blocks, kBlockSize>>>(input, beta, output, N);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-4f;
constexpr float kSentinel = 123456.0f;

std::vector<float> rms_reference(const std::vector<float>& input, float gamma, float beta,
                                 float eps) {
    const int n = static_cast<int>(input.size());
    float sum_sq = 0.0f;
    for (float v : input) {
        sum_sq += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / n + eps);

    std::vector<float> expected(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        expected[i] = input[i] * inv_rms * gamma + beta;
    }
    return expected;
}

bool almost_equal(float a, float b) {
    return std::fabs(a - b) <= kEps;
}

std::vector<float> make_patterned_data(int n, int period, float scale, float bias) {
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i) {
        const int centered = (i % period) - (period / 2);
        values[i] = centered * scale + bias;
    }
    return values;
}

bool run_case(const std::string& name, const std::vector<float>& input, float gamma, float beta,
              float eps) {
    const int n = static_cast<int>(input.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const std::vector<float> expected = rms_reference(input, gamma, beta, eps);
    std::vector<float> actual(n, 0.0f);
    const std::vector<float> sentinel(n, kSentinel);

    float* d_input = nullptr;
    float* d_output = nullptr;

    auto cleanup = [&]() {
        if (d_input != nullptr) cudaFree(d_input);
        if (d_output != nullptr) cudaFree(d_output);
    };

    if (cudaMalloc(&d_input, bytes) != cudaSuccess || cudaMalloc(&d_output, bytes) != cudaSuccess) {
        std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
        cleanup();
        return false;
    }

    bool ok = true;
    ok = ok && (cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    ok = ok && (cudaMemcpy(d_output, sentinel.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    if (!ok) {
        std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed\n";
        cleanup();
        return false;
    }

    solve(d_input, gamma, beta, d_output, n, eps);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "[FAIL] " << name << ": kernel execution or cudaMemcpy D2H failed\n";
        cleanup();
        return false;
    }

    bool unchanged = true;
    for (int i = 0; i < n; ++i) {
        if (!almost_equal(actual[i], kSentinel)) {
            unchanged = false;
            break;
        }
    }
    if (unchanged) {
        std::cerr << "[FAIL] " << name << ": output buffer was never updated by solve\n";
        cleanup();
        return false;
    }

    for (int i = 0; i < n; ++i) {
        if (!almost_equal(actual[i], expected[i])) {
            std::cerr << "[FAIL] " << name << ": mismatch at index " << i
                      << ", expected=" << expected[i] << ", actual=" << actual[i] << '\n';
            cleanup();
            return false;
        }
    }

    cleanup();
    std::cout << "[PASS] " << name << '\n';
    return true;
}

}  // namespace

int main() {
    int passed = 0;
    int total = 0;

    {
        ++total;
        const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        passed += run_case("sample_case_n4", input, 1.0f, 0.0f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {1.0f, 2.0f, 3.0f};
        passed += run_case("sample_case_n3", input, 1.0f, 0.0f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {5.0f};
        passed += run_case("single_element_with_scale_and_shift", input, 2.0f, -1.5f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {0.0f, 0.0f, 0.0f, 0.0f};
        passed += run_case("all_zero_input", input, 1.25f, 3.0f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = {-3.0f, -1.0f, 2.0f, 4.0f, -2.0f, 1.0f};
        passed += run_case("mixed_signs_nontrivial_affine", input, 0.75f, -0.25f, 1e-5f) ? 1 : 0;
    }

    {
        ++total;
        const std::vector<float> input = make_patterned_data(257, 17, 0.5f, -1.0f);
        passed += run_case("patterned_large_input", input, 1.1f, 0.2f, 1e-5f) ? 1 : 0;
    }

    std::cout << "Summary: " << passed << "/" << total << " cases passed\n";
    return passed == total ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

一般一个 block 处理一行 hidden state，先归约平方和，再并行写回。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程在一行里 stride 访问 hidden 维，先贡献局部平方和，拿到 scale 后再处理自己负责的元素。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
namespace {

constexpr int kBlockSize = 256;

__global__ void rms_sum_squares(const float* input, double* partial_sums, int N) {
  __shared__ double shared[kBlockSize];

  int tid = threadIdx.x;
  int global_tid = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  double local_sum = 0.0;
  for (int idx = global_tid; idx < N; idx += stride) {
    double value = static_cast<double>(input[idx]);
    local_sum += value * value;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] += shared[tid + offset];
    __syncthreads();
  }

  if (tid == 0) partial_sums[blockIdx.x] = shared[0];
}

__global__ void rms_reduce_partial_sums(const double* partial_sums, double* total_sum, int count) {
  __shared__ double shared[kBlockSize];

  int tid = threadIdx.x;
  double local_sum = 0.0;
  for (int idx = tid; idx < count; idx += blockDim.x) {
    local_sum += partial_sums[idx];
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] += shared[tid + offset];
    __syncthreads();
  }

  if (tid == 0) total_sum[0] = shared[0];
}

__global__ void rms_normalize(const float* input, float gamma, float beta, float* output, int N,
                              float inv_rms) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx] * inv_rms * gamma + beta;
  }
}

}  // namespace
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程在一行里 stride 访问 hidden 维，先贡献局部平方和，拿到 scale 后再处理自己负责的元素。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。

### 版本 2

先看代码：

```cpp
template <int kThreads>
__global__ void rms_cooperative_kernel(const float* input,
                                       float gamma,
                                       float beta,
                                       float* output,
                                       int N,
                                       float eps) {
  cg::grid_group grid = cg::this_grid();
  float local_sum = 0.0f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    local_sum += input[i] * input[i];
  }

  float block_sum = block_reduce_sum<kThreads>(local_sum);
  if (threadIdx.x == 0) atomicAdd(&g_sum_sq, block_sum);
  grid.sync();

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    g_scale = rsqrtf(g_sum_sq / float(N) + eps) * gamma;
  }
  grid.sync();

  float scale = g_scale;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    output[i] = fmaf(input[i], scale, beta);
  }
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程在一行里 stride 访问 hidden 维，先贡献局部平方和，拿到 scale 后再处理自己负责的元素。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。

## 六、性能和可改方向

这题我会优先看这些点：

- 第一阶段是归约，第二阶段是逐元素缩放。
- weight 是按 hidden 维读取的，访问通常连续。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 平方和要除以 hidden size。
- eps 在 rsqrt 前。
- scale 对同一行所有元素共享。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **RMS Normalization** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
