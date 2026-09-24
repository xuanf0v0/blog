---
title: '图像颜色反转'
published: 2026-07-08
description: '图像颜色反转 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 简单题
  - 逐元素算子
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/color-inversion'
---

刷 LeetGPU 的时候，**图像颜色反转** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/color-inversion](https://leetgpu.com/challenges/color-inversion)

英文名可以先记一下：**Color Inversion**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Color Inversion

图像反色属于典型逐元素题。每个像素通道独立处理，GPU 非常适合把整张图摊平成一维数组后并行计算。

这题的核心关系可以写成：`output[i] = 255 - input[i]`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
for (int i = 0; i < width * height * channels; ++i) {
    output[i] = 255 - input[i];
}
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= width * height) return;
    image[idx * 4]      = 255 - image[idx * 4];
    image[idx * 4 + 1]  = 255 - image[idx * 4 + 1];
    image[idx * 4 + 2]  = 255 - image[idx * 4 + 2];
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    //cudaDeviceSynchronize();
}
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

把图像所有通道看成连续数组，总工作量是 `width * height * channels`，一维 grid 就能覆盖。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程 `idx` 对应一个通道值。只要 `idx` 小于总元素数，就读 input[idx] 并写 output[idx]。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= width * height) return;
    image[idx * 4]      = 255 - image[idx * 4];
    image[idx * 4 + 1]  = 255 - image[idx * 4 + 1];
    image[idx * 4 + 2]  = 255 - image[idx * 4 + 2];
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    //cudaDeviceSynchronize();
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 对应一个通道值。只要 `idx` 小于总元素数，就读 input[idx] 并写 output[idx]。

### 版本 2

先看代码：

```cpp
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= width * height) return;
    auto flip24 = [](int32_t x) -> int32_t {
    // XOR 后低 24 位翻转，高 8 位不变
        return x ^ 0xFFFFFF;
    };
    int temp = flip24(reinterpret_cast<int32_t*>(image)[idx]);
    reinterpret_cast<int32_t*>(image)[idx] = temp;
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 对应一个通道值。只要 `idx` 小于总元素数，就读 input[idx] 并写 output[idx]。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

### 版本 3

先看代码：

```cpp
#define BLOCK_SIZE 256
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > width * height / 4) return;
    auto flip24 = [](int32_t x) -> int32_t {
    // XOR 后低 24 位翻转，高 8 位不变
        return x ^ 0xFFFFFF;
    };
    if(idx == width * height / 4){
        if(width * height % 4 == 0)
            return;
        #pragma unroll
        for(int i = idx * 4; i < width * height; i++){
            int temp = flip24(reinterpret_cast<int32_t*>(image)[i]);
            reinterpret_cast<int32_t*>(image)[i] = temp;
        }
        return;
    }
    auto temp = reinterpret_cast<int4*>(image)[idx];
    //__syncthreads();
    reinterpret_cast<int4*>(image)[idx] = make_int4(flip24(temp.x), flip24(temp.y), 
    flip24(temp.z), flip24(temp.w));
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 对应一个通道值。只要 `idx` 小于总元素数，就读 input[idx] 并写 output[idx]。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

### 版本 4

先看代码：

```cpp
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// 低 24 位取反（RGB），A 不变
__device__ __forceinline__ uint32_t flip24(uint32_t x) { return x ^ 0x00FFFFFFu; }

// 128-bit streaming load（L1 no-allocate + L2 128B），失败则退化为普通加载
__device__ __forceinline__ uint4 ld_stream128(const uint4* ptr) {
#if __CUDA_ARCH__ >= 900  // Hopper/Blackwell
    uint4 v;
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr));
    return v;
#else
    return *ptr;
#endif
}

// 128-bit streaming store
__device__ __forceinline__ void st_stream128(uint4* ptr, const uint4& v) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};\n" ::
        "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
#else
    *ptr = v;
#endif
}

__global__ void invert_rgba_prefetch(uint32_t* __restrict__ data, size_t num_pixels)
{
    // 以 4 像素（16B）为一组
    size_t n4 = num_pixels >> 2;
    uint4* __restrict__ p4 = reinterpret_cast<uint4*>(data);

    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    // 软件流水线：先预取一组
    if (tid < n4) {
        size_t j  = tid;
        uint4 r0  = ld_stream128(p4 + j);
        j += stride;

        // 主体：每轮预取下一组 r1，同时处理并写回 r0
        for (; j < n4; j += stride) {
            uint4 r1 = ld_stream128(p4 + j);

            r0.x = flip24(r0.x);
            r0.y = flip24(r0.y);
            r0.z = flip24(r0.z);
            r0.w = flip24(r0.w);
            st_stream128(p4 + (j - stride), r0);

            r0 = r1; // 滑动窗口
        }

        // 收尾：把最后一组 r0 写回
        r0.x = flip24(r0.x);
        r0.y = flip24(r0.y);
        r0.z = flip24(r0.z);
        r0.w = flip24(r0.w);
        st_stream128(p4 + (j - stride), r0);
    }

    // 处理余数（不足 4 像素）——单线程避免竞争
    if ((num_pixels & 3) && tid == 0) {
        for (size_t k = (n4 << 2); k < num_pixels; ++k) {
            data[k] = flip24(data[k]);
        }
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    const size_t N = (size_t)width * (size_t)height;
    uint32_t* data = reinterpret_cast<uint32_t*>(image);

    int sm = 148; //cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    const int threads = BLOCK_SIZE;
    size_t n4 = N >> 2;

    int blocks;
    if (n4 == 0) blocks = 1;

// ... 下面仍是同一份完整实现，文章此处省略中间细节，保留完整代码请看原题解仓库。
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程 `idx` 对应一个通道值。只要 `idx` 小于总元素数，就读 input[idx] 并写 output[idx]。
- 这里开始做向量化，思路和参考文里调 block size 一样：先让代码跑通，再针对访存吞吐做小步尝试。

## 六、性能和可改方向

这题我会优先看这些点：

- 这题几乎就是内存带宽测试。
- 如果按 uchar4/RGBA 或 float4 做向量化，能减少指令数量。
- 图像类题要关注数据类型，uint8 和 float 的写法不一样。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- RGB/RGBA 的 channels 不同，不要随手写死。
- 如果输入是 unsigned char，`255 - input` 的类型转换要清楚。
- 总元素数别漏乘 channels。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Color Inversion** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
