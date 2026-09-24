---
title: '向量点积'
published: 2026-07-08
description: '向量点积 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 中等题
  - 归约
  - Warp 优化
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/dot-product'
---

刷 LeetGPU 的时候，**向量点积** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/dot-product](https://leetgpu.com/challenges/dot-product)

英文名可以先记一下：**Dot Product**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Dot Product

点积可以看成向量乘法加一次归约。它比 vector add 多了一步合并，因此很适合接在 reduction 后面学习。

这题的核心关系可以写成：`output = sum(A[i] * B[i])`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i)
    sum += A[i] * B[i];
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
#include <iostream>
#include <string>
#include <vector>
#endif

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void dot_product_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ result,
                                   int N) {
  float thread_sum = 0.0f;
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = global_thread; i < N; i += stride) {
    thread_sum = fmaf(A[i], B[i], thread_sum);
  }

  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  thread_sum = warp_reduce_sum(thread_sum);

  if (lane == 0) {
    warp_sums[warp] = thread_sum;
  }
  __syncthreads();

  if (warp == 0) {
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    float block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      atomicAdd(result, block_sum);
    }
  }
}

}  // namespace

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
  if (N <= 0) {
    return;
  }

  constexpr int kThreads = 256;
  constexpr int kMaxBlocks = 4096;
  int blocks = (N + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }

  cudaMemset(result, 0, sizeof(float));
  dot_product_kernel<<<blocks, kThreads>>>(A, B, result, N);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kAbsEps = 1e-5f;
constexpr float kRelEps = 1e-4f;

float reference_dot_product(const std::vector<float>& a, const std::vector<float>& b) {
  double sum = 0.0;
  for (int i = 0; i < static_cast<int>(a.size()); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(sum);
}

bool almost_equal(float actual, float expected) {
  const float diff = std::fabs(actual - expected);
  if (diff <= kAbsEps) {
    return true;
  }
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return diff <= scale * kRelEps;
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
              const std::vector<float>& a,
              const std::vector<float>& b) {
  const int n = static_cast<int>(a.size());
  const size_t vector_bytes = static_cast<size_t>(n) * sizeof(float);
  const float expected = reference_dot_product(a, b);
  float actual = 0.0f;

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_result = nullptr;

  auto cleanup = [&]() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_result != nullptr) cudaFree(d_result);
  };

  if (cudaMalloc(&d_a, vector_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed for A\n";
    cleanup();
    return false;
  }
  if (cudaMalloc(&d_b, vector_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed for B\n";
    cleanup();
    return false;
  }
  if (cudaMalloc(&d_result, sizeof(float)) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed for result\n";
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_a, a.data(), vector_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed for A\n";
    cleanup();
    return false;
  }
  if (cudaMemcpy(d_b, b.data(), vector_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed for B\n";
    cleanup();
    return false;
  }

  const float poison = 12345.0f;
  if (cudaMemcpy(d_result, &poison, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed for result init\n";
    cleanup();
    return false;
  }

  solve(d_a, d_b, d_result, n);

  if (cudaGetLastError() != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel launch failed\n";
    cleanup();
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaDeviceSynchronize failed\n";
    cleanup();
    return false;
  }
  if (cudaMemcpy(&actual, d_result, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy D2H failed for result\n";
    cleanup();
    return false;
  }

  if (!almost_equal(actual, expected)) {
    std::cerr << "[FAIL] " << name << ": expected=" << expected
              << ", actual=" << actual << '\n';
    cleanup();
    return false;
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
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    passed += run_case("example_1_basic_positive_values", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {0.5f, 1.5f, 2.5f};
    const std::vector<float> b = {2.0f, 3.0f, 4.0f};
    passed += run_case("example_2_fractional_inputs", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-3.5f};
    const std::vector<float> b = {2.0f};
    passed += run_case("single_element_minimum_n", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-2.0f, 4.0f, -1.5f, 3.0f, 0.25f};
    const std::vector<float> b = {8.0f, -0.5f, 6.0f, 1.5f, -4.0f};
    passed += run_case("mixed_signs_and_cancellation", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = make_patterned_data(4097, 29, 0.125f, -1.0f);
    const std::vector<float> b = make_patterned_data(4097, 17, -0.25f, 0.5f);
    passed += run_case("larger_deterministic_reference_case", a, b) ? 1 : 0;
  }

  std::cout << passed << "/" << total << " cases passed\n";
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

线程并行计算局部乘积，然后 block 内归约，最后写出局部和或原子加到全局输出。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程 `idx` 读取 `A[idx]` 和 `B[idx]`，得到乘积后参与归约。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
__device__ __forceinline__ float warp_reduce_sum(float x) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
  float thread_sum = 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < N; i += stride) {
    thread_sum = fmaf(A[i], B[i], thread_sum);
  }

  __shared__ float warp_sums[32];
  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  thread_sum = warp_reduce_sum(thread_sum);

  if (lane == 0) warp_sums[warp] = thread_sum;
  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) atomicAdd(result, block_sum);
  }
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 读取 `A[idx]` 和 `B[idx]`，得到乘积后参与归约。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。

## 六、性能和可改方向

这题我会优先看这些点：

- 连续读取 A/B 很重要。
- 不要让所有线程直接 atomicAdd 一个地址，除非 N 很小。
- 可用 float4 提高读取吞吐。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 输出结果要初始化。
- 浮点归约顺序不同会导致微小误差。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Dot Product** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
