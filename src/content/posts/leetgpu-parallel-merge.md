---
title: '并行归并两个有序数组'
published: 2026-07-08
description: '并行归并两个有序数组 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 中等题
  - 并行算法
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/parallel-merge'
---

刷 LeetGPU 的时候，**并行归并两个有序数组** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/parallel-merge](https://leetgpu.com/challenges/parallel-merge)

英文名可以先记一下：**Parallel Merge**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Parallel Merge

并行归并比普通逐元素题难一些，因为串行 merge 的当前位置依赖前面走了多少步。GPU 写法要先把输出区间切开。

这题的核心关系可以写成：`merge sorted A and sorted B into sorted output`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
int i = 0, j = 0, k = 0;
while (i < n && j < m)
    output[k++] = A[i] <= B[j] ? A[i++] : B[j++];
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kItemsPerThread = 8;
constexpr int kTileItems = kBlockThreads * kItemsPerThread;

template <typename T>
__device__ __forceinline__ int merge_path_partition(const T* a,
                                                    int a_count,
                                                    const T* b,
                                                    int b_count,
                                                    int diag) {
  int low = max(0, diag - b_count);
  int high = min(diag, a_count);

  while (low < high) {
    const int mid = (low + high) >> 1;
    const int j = diag - mid;

    if (j > 0 && mid < a_count && b[j - 1] >= a[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

__global__ void parallel_merge_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      int m,
                                      int n) {
  __shared__ float tile[kTileItems];

  const int total = m + n;
  const int block_diag = blockIdx.x * kTileItems;
  if (block_diag >= total) {
    return;
  }

  const int tile_count = min(kTileItems, total - block_diag);
  const int block_a_start = merge_path_partition(a, m, b, n, block_diag);
  const int block_a_end = merge_path_partition(a, m, b, n, block_diag + tile_count);
  const int block_b_start = block_diag - block_a_start;
  const int block_b_end = block_diag + tile_count - block_a_end;
  const int tile_a_count = block_a_end - block_a_start;
  const int tile_b_count = block_b_end - block_b_start;

  float* tile_a = tile;
  float* tile_b = tile + tile_a_count;

  for (int idx = threadIdx.x; idx < tile_a_count; idx += blockDim.x) {
    tile_a[idx] = a[block_a_start + idx];
  }
  for (int idx = threadIdx.x; idx < tile_b_count; idx += blockDim.x) {
    tile_b[idx] = b[block_b_start + idx];
  }
  __syncthreads();

  const int thread_diag = min(threadIdx.x * kItemsPerThread, tile_count);
  const int thread_diag_end = min(thread_diag + kItemsPerThread, tile_count);

  const int thread_a_start =
      merge_path_partition(tile_a, tile_a_count, tile_b, tile_b_count, thread_diag);
  const int thread_a_end =
      merge_path_partition(tile_a, tile_a_count, tile_b, tile_b_count, thread_diag_end);
  const int thread_b_start = thread_diag - thread_a_start;
  const int thread_b_end = thread_diag_end - thread_a_end;

  int a_idx = thread_a_start;
  int b_idx = thread_b_start;
  float out[kItemsPerThread];

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const bool active = thread_diag + item < thread_diag_end;
    const bool take_a =
        active &&
        (b_idx >= thread_b_end || (a_idx < thread_a_end && tile_a[a_idx] <= tile_b[b_idx]));
    out[item] = take_a ? tile_a[a_idx++] : (active ? tile_b[b_idx++] : 0.0f);
  }

  const int out_base = block_diag + thread_diag;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    if (thread_diag + item < tile_count) {
      c[out_base + item] = out[item];
    }
  }
}

}  // namespace

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  const int total = M + N;
  if (total == 0) {
    return;
  }

  const int blocks = (total + kTileItems - 1) / kTileItems;
  parallel_merge_kernel<<<blocks, kBlockThreads>>>(A, B, C, M, N);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kEps = 1e-6f;
constexpr float kSentinel = 123456.0f;

std::vector<float> merge_reference(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> expected;
  expected.reserve(a.size() + b.size());

  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      expected.push_back(a[i++]);
    } else {
      expected.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    expected.push_back(a[i++]);
  }
  while (j < b.size()) {
    expected.push_back(b[j++]);
  }
  return expected;
}

std::vector<float> make_sorted_data(int count, float start, float step, int wobble_mod) {
  std::vector<float> values(count);
  for (int idx = 0; idx < count; ++idx) {
    values[idx] = start + step * static_cast<float>(idx) +
                  0.001f * static_cast<float>(idx % wobble_mod);
  }
  return values;
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kEps;
}

bool run_case(const std::string& name,
              const std::vector<float>& a,
              const std::vector<float>& b) {
  const int m = static_cast<int>(a.size());
  const int n = static_cast<int>(b.size());
  const size_t bytes_a = static_cast<size_t>(m) * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(n) * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(m + n) * sizeof(float);

  const std::vector<float> expected = merge_reference(a, b);
  std::vector<float> actual(m + n, 0.0f);
  const std::vector<float> sentinel(m + n, kSentinel);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  auto cleanup = [&]() {
    if (d_a != nullptr) cudaFree(d_a);
    if (d_b != nullptr) cudaFree(d_b);
    if (d_c != nullptr) cudaFree(d_c);
  };

  if (cudaMalloc(&d_a, bytes_a) != cudaSuccess ||
      cudaMalloc(&d_b, bytes_b) != cudaSuccess ||
      cudaMalloc(&d_c, bytes_c) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  bool ok = true;
  ok = ok && (cudaMemcpy(d_a, a.data(), bytes_a, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_b, b.data(), bytes_b, cudaMemcpyHostToDevice) == cudaSuccess);
  ok = ok && (cudaMemcpy(d_c, sentinel.data(), bytes_c, cudaMemcpyHostToDevice) == cudaSuccess);
  if (!ok) {
    std::cerr << "[FAIL] " << name << ": cudaMemcpy H2D failed\n";
    cleanup();
    return false;
  }

  solve(d_a, d_b, d_c, m, n);
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_c, bytes_c, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution or cudaMemcpy D2H failed\n";
    cleanup();
    return false;
  }

  bool unchanged = true;
  for (float value : actual) {
    if (!almost_equal(value, kSentinel)) {
      unchanged = false;
      break;
    }
  }
  if (unchanged) {
    std::cerr << "[FAIL] " << name << ": output buffer was never updated by solve\n";
    cleanup();
    return false;
  }

  for (int idx = 0; idx < m + n; ++idx) {
    if (!almost_equal(actual[idx], expected[idx])) {
      std::cerr << "[FAIL] " << name << ": mismatch at index " << idx
                << ", expected=" << expected[idx] << ", actual=" << actual[idx] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

bool run_large_case() {
  const std::vector<float> a = make_sorted_data(1 << 18, -1000.0f, 0.5f, 7);
  const std::vector<float> b = make_sorted_data(1 << 18, -999.75f, 0.5f, 11);
  return run_case("large_balanced_interleaved_case", a, b);
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const std::vector<float> a = {1.0f, 3.0f, 5.0f, 7.0f};
    const std::vector<float> b = {2.0f, 4.0f, 6.0f, 8.0f};
    passed += run_case("sample_case_interleaved_even_lengths", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-1.0f, 1.0f, 3.0f};
    const std::vector<float> b = {2.0f};
    passed += run_case("sample_case_single_tail_insert", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {1.5f};
    const std::vector<float> b = {-2.5f};
    passed += run_case("minimum_sizes_one_element_each", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-4.0f, -1.0f, 2.0f, 2.0f};
    const std::vector<float> b = {-3.0f, 2.0f, 5.0f};
    passed += run_case("duplicates_and_negative_values", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {-10.0f, -9.0f, -8.0f};
    const std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    passed += run_case("one_array_entirely_before_the_other", a, b) ? 1 : 0;
  }

  {
    ++total;
    const std::vector<float> a = {0.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> b = {0.0f, 0.0f, 2.0f};
    passed += run_case("repeated_equal_values", a, b) ? 1 : 0;
  }

  {
    ++total;
    passed += run_large_case() ? 1 : 0;
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

常见思路是让线程或 block 负责一段输出区间，通过二分查找确定这段区间在 A/B 中的起点。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程拿到输出位置 k 后，需要知道有多少元素来自 A、有多少来自 B。这个分界可以通过 rank 或 merge path 算出来。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
#define BLOCK_SIZE 256

__device__ inline int binary_search(const float* B, int N, float k, bool is_upper) {
  int l = 0, r = N;
  while (l < r) {
    int mid = (l + r) / 2;
    if ((!is_upper && B[mid] >= k) || (is_upper && B[mid] > k)) {
      r = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

__global__ void asc_merge_A(const float* A, const float* B, float* C, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  int countB = binary_search(B, N, A[idx], false);
  C[countB + idx] = A[idx];
}

__global__ void asc_merge_B(const float* A, const float* B, float* C, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int countA = binary_search(A, M, B[idx], true);
  C[countA + idx] = B[idx];
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  asc_merge_A<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(A, B, C, M, N);
  asc_merge_B<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(A, B, C, M, N);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程拿到输出位置 k 后，需要知道有多少元素来自 A、有多少来自 B。这个分界可以通过 rank 或 merge path 算出来。

### 版本 2

先看代码：

```cpp
template <typename T>
__device__ __forceinline__ int merge_path_partition(const T* a, int m, const T* b, int n, int diag) {
  int low = max(0, diag - n);
  int high = min(diag, m);
  while (low < high) {
    int mid = (low + high) >> 1;
    int j = diag - mid;
    if (j > 0 && mid < m && b[j - 1] >= a[mid]) low = mid + 1;
    else high = mid;
  }
  return low;
}

__global__ void parallel_merge_kernel(const float* a, const float* b, float* c, int m, int n) {
  __shared__ float tile[kTileItems];
  int block_diag = blockIdx.x * kTileItems;
  int total = m + n;
  if (block_diag >= total) return;

  int tile_count = min(kTileItems, total - block_diag);
  int a0 = merge_path_partition(a, m, b, n, block_diag);
  int a1 = merge_path_partition(a, m, b, n, block_diag + tile_count);
  int b0 = block_diag - a0;
  int b1 = block_diag + tile_count - a1;

  // load tile_a / tile_b to shared memory
  // each thread merges its own diagonal slice
  // write back contiguous outputs
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程拿到输出位置 k 后，需要知道有多少元素来自 A、有多少来自 B。这个分界可以通过 rank 或 merge path 算出来。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。

## 六、性能和可改方向

这题我会优先看这些点：

- 直接一线程一个输出但每次二分，容易写对但开销不小。
- block 级分段后顺序 merge 小片段，可能更快。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 相等元素的稳定性规则要一致。
- 分段边界不能重复写或漏写。
- 二分条件错一位就会整体错位。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Parallel Merge** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
