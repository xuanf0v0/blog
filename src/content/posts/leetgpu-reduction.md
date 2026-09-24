---
title: '并行归约求和'
published: 2026-07-08
description: '并行归约求和 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
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
sourceLink: 'https://leetgpu.com/challenges/reduction'
---

刷 LeetGPU 的时候，**并行归约求和** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/reduction](https://leetgpu.com/challenges/reduction)

英文名可以先记一下：**Reduction**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Reduction

归约求和是 CUDA 经典题。它不是一个线程写一个输出，而是很多输入最终合成一个标量。

这题的核心关系可以写成：`output = sum(input[i])`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) {
    sum += input[i];
}
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024
__global__ void reduction(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //1. copy to shared
    //2. shared reduction
    //3. warp reduction
    //4. block sum
    __shared__ float sharedMem[BLOCK_SIZE];
    //拷到共享内存（越界用0），不能提前return
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float sum = sharedMem[threadIdx.x];          
        unsigned mask = 0xffffffff;
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
        if(threadIdx.x == 0){
            atomicAdd(output, sum);
        }
    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    reduction<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE ,BLOCK_SIZE>>>(input, output, N);
}
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

通常每个 block 先处理一段输入，block 内用 shared memory 或 warp shuffle 求局部和，再把局部和写到全局。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程先读自己的元素或 stride 循环读多个元素，然后参与 block 内归约。最后通常只有 `threadIdx.x == 0` 写出 block 结果。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024
__global__ void reduction(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //1. copy to shared
    //2. shared reduction
    //3. warp reduction
    //4. block sum
    __shared__ float sharedMem[BLOCK_SIZE];
    //拷到共享内存（越界用0），不能提前return
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float sum = sharedMem[threadIdx.x];          
        unsigned mask = 0xffffffff;
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
        if(threadIdx.x == 0){
            atomicAdd(output, sum);
        }
    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    reduction<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE ,BLOCK_SIZE>>>(input, output, N);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程先读自己的元素或 stride 循环读多个元素，然后参与 block 内归约。最后通常只有 `threadIdx.x == 0` 写出 block 结果。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。

### 版本 2

先看代码：

```cpp
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define FULL_MASK 0xffffffffu

__inline__ __device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

template<int BS>
__global__ void reduce_limit_blocks(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int n) {
    __shared__ float smem[BS];
    const int tid   = threadIdx.x;
    const int bid   = blockIdx.x;
    const int gsize = gridDim.x * (BS * 2);   // 2x 展开
    int i = bid * (BS * 2) + tid;

    // 线程本地累加（两元素/线程）
    float sum = 0.f;
    if (i < n)              sum += x[i];
    if (i + BS < n)         sum += x[i + BS];

    // grid-stride 回合也按 2x 展开
    for (i += gsize; i < n; i += gsize) {
        sum += x[i];
        if (i + BS < n) sum += x[i + BS];
    }

    smem[tid] = sum;
    __syncthreads();

    // block 内规约（到 32 停）
    #pragma unroll
    for (int s = BS / 2; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // warp 规约
    if (tid < 32) {
        float v = smem[tid];
        v = warp_reduce_sum(v);
        if (tid == 0) atomicAdd(out, v);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // 设备信息
    int dev = 0, sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

    // 自然网格（两元素/线程）
    int natural = (N + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);

    // 关键：限制但别太小。经验：SM * 6~10 比较稳
    int grid = sm * 8;
    if (natural < grid) grid = natural;
    if (grid < 1) grid = 1;

    reduce_limit_blocks<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(input, output, N);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程先读自己的元素或 stride 循环读多个元素，然后参与 block 内归约。最后通常只有 `threadIdx.x == 0` 写出 block 结果。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。

### 版本 3

先看代码：

```cpp
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256   // 可试 128/192/320 做 AB
#endif

#define FULL_MASK 0xffffffffu

__inline__ __device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

template<int BS>
__global__ void reduce_fast(const float* __restrict__ x,
                            float* __restrict__ out,
                            int n) {
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp_id  = tid >> 5;             // 0..(BS/32-1)
    const int warps    = BS / 32;

    float sum = 0.f;

    // -------- 向量化主循环：float4 + 2× 展开 => 每迭代处理 8 个元素/线程 --------
    const bool aligned16 = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0);
    if (aligned16) {
        const int n4 = n >> 2;                 // 以 float4 计
        const float4* __restrict__ v = reinterpret_cast<const float4*>(x);

        int j = blockIdx.x * BS + tid;         // 每线程一个向量索引
        const int stride4 = gridDim.x * BS;

        // 每回合拿两组：j 与 j+stride4（手动 2× 展开）
        for (; j < n4; j += (stride4 << 1)) {
            float4 a = v[j];
            sum += a.x + a.y + a.z + a.w;

            int j2 = j + stride4;
            if (j2 < n4) {
                float4 b = v[j2];
                sum += b.x + b.y + b.z + b.w;
            }
        }

        // 处理 4 对齐剩余的尾巴（0~3 个），仍用 grid-stride
        int base = n4 << 2;
        for (int i = base + blockIdx.x * BS + tid; i < n; i += gridDim.x * BS) {
            sum += x[i];
        }
    } else {
        // 非 16B 对齐：回退到标量 2× 展开
        const int gsize = gridDim.x * (BS * 2);
        int i = blockIdx.x * (BS * 2) + tid;
        if (i < n)       sum += x[i];
        if (i + BS < n)  sum += x[i + BS];
        for (i += gsize; i < n; i += gsize) {
            sum += x[i];
            if (i + BS < n) sum += x[i + BS];
        }
    }

    // -------- 纯 warp 规约 + 仅一次同步 --------
    sum = warp_reduce_sum(sum);

    __shared__ float smem[BS / 32];            // 每个 warp 一个槽
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();                           // 仅此一次

    // 由 warp0 做最后规约
    float block_sum = 0.f;
    if (warp_id == 0) {
        block_sum = (tid < warps) ? smem[lane] : 0.f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) atomicAdd(out, block_sum);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int dev = 0, sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

    // grid 适度加大以饱和带宽。经验：SM * 10~12 往往比 *8 更满
    int natural = (N + (BLOCK_SIZE * 8) - 1) / (BLOCK_SIZE * 8); // 估算向量化吞吐

// ... 下面仍是同一份完整实现，文章此处省略中间细节，保留完整代码请看原题解仓库。
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程先读自己的元素或 stride 循环读多个元素，然后参与 block 内归约。最后通常只有 `threadIdx.x == 0` 写出 block 结果。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

## 六、性能和可改方向

这题我会优先看这些点：

- 第一版可以 atomicAdd，但性能一般。
- shared memory 归约是基础优化。
- warp shuffle 可以减少 shared memory 和同步开销。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 每轮 stride 的活跃线程条件容易写错。
- 浮点加法顺序变了，结果可能有细小差异。
- `__syncthreads()` 不能随便删。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Reduction** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
