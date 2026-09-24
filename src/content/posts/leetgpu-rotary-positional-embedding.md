---
title: '旋转位置编码'
published: 2026-07-08
description: '旋转位置编码 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 中等题
  - Transformer
  - 深度学习算子
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/rotary-positional-embedding'
---

刷 LeetGPU 的时候，**旋转位置编码** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/rotary-positional-embedding](https://leetgpu.com/challenges/rotary-positional-embedding)

英文名可以先记一下：**Rotary Positional Embedding**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Rotary Positional Embedding

RoPE 不是把位置向量加上去，而是对偶奇维组成的二维向量做旋转。它在 LLM 里非常常见。

这题的核心关系可以写成：`y0 = x0 * cos - x1 * sin; y1 = x0 * sin + x1 * cos`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
for each token and dim pair:
    y0 = x0 * cos_pos - x1 * sin_pos;
    y1 = x0 * sin_pos + x1 * cos_pos;
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

__device__ __forceinline__ float4 rope_low4(const float4& q_low,
                                            const float4& q_high,
                                            const float4& c_low,
                                            const float4& s_low) {
  return make_float4(
      fmaf(q_low.x, c_low.x, -q_high.x * s_low.x),
      fmaf(q_low.y, c_low.y, -q_high.y * s_low.y),
      fmaf(q_low.z, c_low.z, -q_high.z * s_low.z),
      fmaf(q_low.w, c_low.w, -q_high.w * s_low.w));
}

__device__ __forceinline__ float4 rope_high4(const float4& q_low,
                                             const float4& q_high,
                                             const float4& c_high,
                                             const float4& s_high) {
  return make_float4(
      fmaf(q_high.x, c_high.x, q_low.x * s_high.x),
      fmaf(q_high.y, c_high.y, q_low.y * s_high.y),
      fmaf(q_high.z, c_high.z, q_low.z * s_high.z),
      fmaf(q_high.w, c_high.w, q_low.w * s_high.w));
}

template <int kHalf>
__global__ void rope_pairs_vec4_128(const float* __restrict__ Q,
                                    const float* __restrict__ cos,
                                    const float* __restrict__ sin,
                                    float* __restrict__ output,
                                    int M) {
  constexpr int kWidth = 4;
  constexpr int kD = kHalf * 2;
  constexpr int kVecsPerRow = kHalf / kWidth;

  const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_vecs = M * kVecsPerRow;
  if (vec_idx >= total_vecs) {
    return;
  }

  const int row = vec_idx / kVecsPerRow;
  const int vec_col = vec_idx - row * kVecsPerRow;
  const int base = row * kD + vec_col * kWidth;

  const float4 q_low = reinterpret_cast<const float4*>(Q + base)[0];
  const float4 q_high = reinterpret_cast<const float4*>(Q + base + kHalf)[0];
  const float4 c_low = reinterpret_cast<const float4*>(cos + base)[0];
  const float4 c_high = reinterpret_cast<const float4*>(cos + base + kHalf)[0];
  const float4 s_low = reinterpret_cast<const float4*>(sin + base)[0];
  const float4 s_high = reinterpret_cast<const float4*>(sin + base + kHalf)[0];

  reinterpret_cast<float4*>(output + base)[0] = rope_low4(q_low, q_high, c_low, s_low);
  reinterpret_cast<float4*>(output + base + kHalf)[0] = rope_high4(q_low, q_high, c_high, s_high);
}

__global__ void rope_pairs_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ cos,
                                  const float* __restrict__ sin,
                                  float* __restrict__ output,
                                  int M,
                                  int D) {
  const size_t half = static_cast<size_t>(D >> 1);
  const size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total_pairs = static_cast<size_t>(M) * half;
  if (pair_idx >= total_pairs) {
    return;
  }

  const size_t row = pair_idx / half;
  const size_t col = pair_idx - row * half;
  const size_t base = row * static_cast<size_t>(D) + col;
  const size_t high = base + half;

  const float q_low = Q[base];
  const float q_high = Q[high];
  output[base] = fmaf(q_low, cos[base], -q_high * sin[base]);
  output[high] = fmaf(q_high, cos[high], q_low * sin[high]);
}

}  // namespace

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
  if (M <= 0 || D <= 0) {
    return;
  }

  constexpr int kThreads = 256;
  if (D == 128) {
    constexpr int kHalf = 64;
    constexpr int kVecsPerRow = kHalf / 4;
    const int total_vecs = M * kVecsPerRow;
    const int blocks = (total_vecs + kThreads - 1) / kThreads;
    rope_pairs_vec4_128<kHalf><<<blocks, kThreads>>>(Q, cos, sin, output, M);
    return;
  }

  const size_t total_pairs = static_cast<size_t>(M) * static_cast<size_t>(D >> 1);
  const int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
  rope_pairs_kernel<<<blocks, kThreads>>>(Q, cos, sin, output, M, D);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-5f;

std::vector<float> rope_reference(const std::vector<float>& q,
                                  const std::vector<float>& cos,
                                  const std::vector<float>& sin,
                                  int M,
                                  int D) {
  std::vector<float> expected(M * D, 0.0f);
  const int half = D / 2;
  for (int row = 0; row < M; ++row) {
    const int base = row * D;
    for (int col = 0; col < D; ++col) {
      const int pair_col = (col < half) ? (col + half) : (col - half);
      const float rotated = (col < half) ? -q[base + pair_col] : q[base + pair_col];
      expected[base + col] = q[base + col] * cos[base + col] + rotated * sin[base + col];
    }
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

bool run_case(const std::string& name,
              const std::vector<float>& q,
              const std::vector<float>& cos,
              const std::vector<float>& sin,
              int M,
              int D) {
  const size_t bytes = static_cast<size_t>(M) * static_cast<size_t>(D) * sizeof(float);
  const std::vector<float> expected = rope_reference(q, cos, sin, M, D);
  std::vector<float> actual(M * D, 0.0f);

  float* d_q = nullptr;
  float* d_cos = nullptr;
  float* d_sin = nullptr;
  float* d_out = nullptr;

  auto cleanup = [&]() {
    if (d_q != nullptr) cudaFree(d_q);
    if (d_cos != nullptr) cudaFree(d_cos);
    if (d_sin != nullptr) cudaFree(d_sin);
    if (d_out != nullptr) cudaFree(d_out);
  };

  if (cudaMalloc(&d_q, bytes) != cudaSuccess ||
      cudaMalloc(&d_cos, bytes) != cudaSuccess ||
      cudaMalloc(&d_sin, bytes) != cudaSuccess ||
      cudaMalloc(&d_out, bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  bool ok = true;
  ok = ok && (cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_cos, cos.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_sin, sin.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);

  if (!ok) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed\n";
    cleanup();
    return false;
  }

  solve(d_q, d_cos, d_sin, d_out, M, D);
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution or cudaMemcpy D2H failed\n";
    cleanup();
    return false;
  }

  for (int i = 0; i < M * D; ++i) {
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

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int M = 2;
    const int D = 4;
    const std::vector<float> q = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    const std::vector<float> cos = {
        1.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    const std::vector<float> sin = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    passed += run_case("sample_case_identity_and_rotation", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int D = 2;
    const std::vector<float> q = {2.5f, -4.0f};
    const std::vector<float> cos = {0.0f, 0.0f};
    const std::vector<float> sin = {1.0f, 1.0f};
    passed += run_case("smallest_even_dimension", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int D = 6;
    const std::vector<float> q = {
        1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f,
    };
    const std::vector<float> cos = {
        0.5f, -1.0f, 0.25f, 2.0f, -0.5f, 1.5f,
    };
    const std::vector<float> sin = {
        1.0f, 0.5f, -1.0f, 0.25f, 2.0f, -0.5f,
    };
    passed += run_case("mixed_signs_and_coefficients", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 3;
    const int D = 4;
    const std::vector<float> q = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, -2.0f, 5.0f, 6.0f,
        0.5f, -0.5f, 1.5f, -1.5f,
    };
    const std::vector<float> cos = {
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };
    const std::vector<float> sin = {
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        -0.5f, 1.0f, -1.0f, 0.5f,
    };
    passed += run_case("multiple_rows_independent_positions", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 2;
    const int D = 8;
    const std::vector<float> q = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        -8.0f, -7.0f, -6.0f, -5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    };
    const std::vector<float> cos = {
        1.0f, 0.5f, 0.0f, -0.5f, -1.0f, 1.5f, -1.5f, 2.0f,
        2.0f, -2.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, -0.25f,
    };
    const std::vector<float> sin = {
        -1.0f, 1.0f, -0.5f, 0.5f, 1.0f, -1.0f, 1.5f, -1.5f,
        0.25f, 0.75f, -0.25f, -0.75f, 1.25f, -1.25f, 0.1f, -0.1f,
    };
    passed += run_case("larger_dimension_stress_pattern", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 2;
    const int D = 128;
    const std::vector<float> q = make_patterned_data(M * D, 19, 0.125f, -0.25f);
    const std::vector<float> cos = make_patterned_data(M * D, 23, 0.05f, 0.5f);
    const std::vector<float> sin = make_patterned_data(M * D, 29, -0.04f, -0.1f);
    passed += run_case("specialized_path_d128", q, cos, sin, M, D) ? 1 : 0;
  }

  {
    ++total;
    const int M = 1;
    const int D = 2048;
    const std::vector<float> q = make_patterned_data(M * D, 31, 0.03125f, 0.75f);
    const std::vector<float> cos = make_patterned_data(M * D, 27, -0.02f, 0.3f);
    const std::vector<float> sin = make_patterned_data(M * D, 25, 0.015f, -0.45f);
    passed += run_case("generic_large_dimension", q, cos, sin, M, D) ? 1 : 0;
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

线程可以负责一个维度对，也可以负责一个元素。grid 需要覆盖 token、head 和 rotary 维度。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

由 token 位置找到 sin/cos 行，由 dim pair 找到旋转频率，再读相邻偶奇两个元素。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
__global__ void ropeToken(float* Q, float* cos, float* sin, float* output, int M, int D) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float token[];
  int offset = bid * D;

  token[tid] = Q[offset + tid];
  __syncthreads();

  output[offset + tid] = token[tid] * cos[offset + tid];
  if (tid >= D / 2) {
    output[offset + tid] += token[tid - D / 2] * sin[offset + tid];
  } else {
    output[offset + tid] -= token[tid + D / 2] * sin[offset + tid];
  }
}

extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
  ropeToken<<<M, D, D * sizeof(float)>>>(Q, cos, sin, output, M, D);
  cudaDeviceSynchronize();
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。由 token 位置找到 sin/cos 行，由 dim pair 找到旋转频率，再读相邻偶奇两个元素。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。

### 版本 2

先看代码：

```cpp
__global__ void rope_pairs_kernel(const float* Q,
                                  const float* cos,
                                  const float* sin,
                                  float* output,
                                  int M,
                                  int D) {
  size_t half = D >> 1;
  size_t pair_idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_pairs = size_t(M) * half;
  if (pair_idx >= total_pairs) return;

  size_t row = pair_idx / half;
  size_t col = pair_idx - row * half;
  size_t base = row * size_t(D) + col;
  size_t high = base + half;

  float q_low = Q[base];
  float q_high = Q[high];
  output[base] = fmaf(q_low, cos[base], -q_high * sin[base]);
  output[high] = fmaf(q_high, cos[high], q_low * sin[high]);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。由 token 位置找到 sin/cos 行，由 dim pair 找到旋转频率，再读相邻偶奇两个元素。

## 六、性能和可改方向

这题我会优先看这些点：

- RoPE 计算轻，重点是连续读写。
- 如果 Q/K 都要旋转，可以合并处理减少 launch。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 偶数维和奇数维必须成对。
- 只旋转 rotary_dim 范围内的维度。
- sin/cos 的索引别用错 token 位置。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Rotary Positional Embedding** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
