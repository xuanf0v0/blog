---
title: '一维卷积'
published: 2026-07-08
description: '一维卷积 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 简单题
  - 卷积
  - Stencil
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/1d-convolution'
---

刷 LeetGPU 的时候，**一维卷积** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/1d-convolution](https://leetgpu.com/challenges/1d-convolution)

英文名可以先记一下：**1D Convolution**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、1D Convolution

一维卷积就是滑动窗口点积。相比向量加法，它多了一层窗口循环，也更容易引出向量化、共享内存和常量内存这些优化话题。

这题的核心关系可以写成：`output[i] = sum(input[i + j] * kernel[j])`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
int outN = N - K + 1;
for (int i = 0; i < outN; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < K; ++j)
        sum += input[i + j] * kernel[j];
    output[i] = sum;
}
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    #pragma unroll
    for(int i = 0; i < kernel_size; i++){
        float tempI = input[idx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[idx] = temp;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    //cudaDeviceSynchronize();
}
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

输出长度是 `N - K + 1`，所以 grid 覆盖的是输出元素数量，而不是输入 N。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程 `idx` 负责 `output[idx]`，对应输入窗口起点也是 `idx`，因此窗口内访问是 `input[idx + j]`。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
#include <cuda_runtime.h>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    #pragma unroll
    for(int i = 0; i < kernel_size; i++){
        float tempI = input[idx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[idx] = temp;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    //cudaDeviceSynchronize();
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 负责 `output[idx]`，对应输入窗口起点也是 `idx`，因此窗口内访问是 `input[idx + j]`。

### 版本 2

先看代码：

```cpp
#include <cuda_runtime.h>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    #pragma unroll
    for(int i = 0; i < kernel_size / 4; i++){
        int p = i * 4;
        float4 tempI = make_float4(input[idx + p], input[idx + p + 1], input[idx + p + 2], input[idx + p + 3]);
        float4 tempK = make_float4(kernel[p], kernel[p + 1], kernel[p + 2], kernel[p + 3]);
        temp += tempI.x * tempK.x + tempI.y * tempK.y + tempI.z * tempK.z + tempI.w * tempK.w ;
    }
    for(int i = kernel_size & ~3; i < kernel_size; i++ ){
        float tempI = input[idx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[idx] = temp;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    //cudaDeviceSynchronize();
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 负责 `output[idx]`，对应输入窗口起点也是 `idx`，因此窗口内访问是 `input[idx + j]`。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

### 版本 3

先看代码：

```cpp
#include <cuda_runtime.h>

__global__ void conv1d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    const int out_start = blockIdx.x * blockDim.x;                  // 本 block 负责的输出起点（全局）
    const int out_end   = min(out_start + blockDim.x, out_size);    // 本 block 负责的输出终点（开区间）

    // 共享内存布局：[ 输入切片 (blockDim.x + K - 1) | kernel (K) ]
    extern __shared__ float smem[];
    float* s_in = smem;
    float* s_k  = smem + (blockDim.x + kernel_size - 1);

    // 这个 tile 对应的输入起点
    const int tile_in_start = out_start; // valid 卷积时输入对齐输出
    const int tile_in_len   = min(blockDim.x + kernel_size - 1, input_size - tile_in_start);

    // 1) cooperative load: 输入切片 -> shared
    #pragma unroll
    for (int t = threadIdx.x; t < tile_in_len; t += blockDim.x) {
        s_in[t] = input[tile_in_start + t];
    }
    #pragma unroll
    // 2) cooperative load: kernel -> shared
    for (int t = threadIdx.x; t < kernel_size; t += blockDim.x) {
        s_k[t] = kernel[t];
    }
    __syncthreads();

    // 3) 计算本线程的输出（若在有效范围内）
    const int out_idx = out_start + threadIdx.x;
    if (out_idx < out_end) {
        // 在共享内存上做点积：s_in[threadIdx.x + j] * s_k[j]
        float acc = 0.f;

        // 手动 4-步展开（避免 float4 对齐问题，且适用于任意对齐的起点）
        int j = 0;
        #pragma unroll
        for (; j + 3 < kernel_size; j += 4) {
            float a0 = s_in[threadIdx.x + j + 0];
            float a1 = s_in[threadIdx.x + j + 1];
            float a2 = s_in[threadIdx.x + j + 2];
            float a3 = s_in[threadIdx.x + j + 3];
            float b0 = s_k[j + 0];
            float b1 = s_k[j + 1];
            float b2 = s_k[j + 2];
            float b3 = s_k[j + 3];
            acc += a0*b0 + a1*b1 + a2*b2 + a3*b3;
        }
        for (; j < kernel_size; ++j) {
            acc += s_in[threadIdx.x + j] * s_k[j];
        }

        output[out_idx] = acc;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    if (out_size <= 0) return;

    const int threadsPerBlock = 1024;
    const int blocksPerGrid   = (out_size + threadsPerBlock - 1) / threadsPerBlock;

    // 共享内存大小 = 输入切片 + kernel
    const size_t shmem_bytes =
        (threadsPerBlock + kernel_size - 1 + kernel_size) * sizeof(float);

    conv1d_shared_kernel<<<std::min(148*22,blocksPerGrid), threadsPerBlock, shmem_bytes>>>(
        input, kernel, output, input_size, kernel_size);

}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 负责 `output[idx]`，对应输入窗口起点也是 `idx`，因此窗口内访问是 `input[idx + j]`。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

### 版本 4

先看代码：

```cpp
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_THREADS 256
#define THREAD_OUT    8
#define MAX_K         4096

__constant__ float ck[MAX_K];

template<int BYTES>
__device__ __forceinline__ void cp_async_smem_gmem(void* smem_dst, const void* gmem_src) {
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst)); // 32-bit shared addr
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
        "r"(saddr), "l"(gmem_src), "n"(BYTES)
    );
}

__global__ void conv1d_bk_cpasync_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    if (out_size <= 0) return;

    const int out_per_blk = BLOCK_THREADS * THREAD_OUT;
    const int out_blk_beg = blockIdx.x * out_per_blk;
    if (out_blk_beg >= out_size) return;

    // tile 输入覆盖 [out_blk_beg, out_blk_beg + out_per_blk + K - 1)
    const int tile_in_start = out_blk_beg;
    const int stage_elems   = out_per_blk + kernel_size - 1;
    const int tile_in_len   = max(0, min(stage_elems, input_size - tile_in_start));

    extern __shared__ float smem[];
    float* s_in = smem;

    // ===== 全覆盖装载：float4 协同 + 尾部标量 =====
    const int vec_elems  = (tile_in_len & ~3);     // 4 对齐部分
    const int vec_chunks = (vec_elems >> 2);       // float4 个数

    // 关键修复：所有线程覆盖所有 chunk（不再按 warp 残缺分段）
    for (int i = threadIdx.x; i < vec_chunks; i += BLOCK_THREADS) {
        const char* gptr = reinterpret_cast<const char*>(input + tile_in_start) + i * 16;
        char*       sptr = reinterpret_cast<char*>(s_in)                               + i * 16;
        cp_async_smem_gmem<16>(sptr, gptr);
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);

    // 剩余 <4 元素用标量补齐
    for (int t = vec_elems + threadIdx.x; t < tile_in_len; t += BLOCK_THREADS) {
        s_in[t] = input[tile_in_start + t];
    }
    __syncthreads();

    // ===== 计算：每线程 8 输出 =====
    const int tlocal  = threadIdx.x * THREAD_OUT;
    const int out_g0  = out_blk_beg + tlocal;
    const int n_valid = max(0, min(THREAD_OUT, out_size - out_g0));
    if (n_valid <= 0) return;

    float acc[THREAD_OUT] = {0};

    int j = 0;
    for (; j + 3 < kernel_size; j += 4) {
        float k0 = ck[j+0], k1 = ck[j+1], k2 = ck[j+2], k3 = ck[j+3];
        #pragma unroll
        for (int o = 0; o < THREAD_OUT; ++o) {
            float a0 = s_in[tlocal + o + j + 0];
            float a1 = s_in[tlocal + o + j + 1];
            float a2 = s_in[tlocal + o + j + 2];
            float a3 = s_in[tlocal + o + j + 3];
            acc[o] += a0*k0 + a1*k1 + a2*k2 + a3*k3;  // 让编译器自由做 FMA/ILP
        }
    }
    for (; j < kernel_size; ++j) {
        float kj = ck[j];
        #pragma unroll
        for (int o = 0; o < THREAD_OUT; ++o) {
            acc[o] += s_in[tlocal + o + j] * kj;
        }
    }

    #pragma unroll
    for (int o = 0; o < THREAD_OUT; ++o) {
        if (o < n_valid) output[out_g0 + o] = acc[o];
    }
}

// ... 下面仍是同一份完整实现，文章此处省略中间细节，保留完整代码请看原题解仓库。
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 负责 `output[idx]`，对应输入窗口起点也是 `idx`，因此窗口内访问是 `input[idx + j]`。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

## 六、性能和可改方向

这题我会优先看这些点：

- 相邻输出窗口高度重叠，这是 shared memory tile 的机会。
- kernel 对所有线程相同，可以考虑 constant memory。
- K 比较小时 unroll/float4 会有收益，K 大时重点变成数据复用。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- N 小于 K 时输出长度非法。
- shared memory tile 长度要算成 `blockDim.x + K - 1`。
- 向量化后尾部元素不能漏。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **1D Convolution** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
