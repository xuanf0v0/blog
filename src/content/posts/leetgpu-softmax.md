---
title: 'Softmax 归一化'
published: 2026-07-08
description: 'Softmax 归一化 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
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
sourceLink: 'https://leetgpu.com/challenges/softmax'
---

刷 LeetGPU 的时候，**Softmax 归一化** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/softmax](https://leetgpu.com/challenges/softmax)

英文名可以先记一下：**Softmax**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Softmax

softmax 是这批题里比较重要也比较麻烦的算子。它要先求最大值，再求指数和，最后归一化，天然包含多次归约。

这题的核心关系可以写成：`output[i] = exp(input[i] - max) / sum(exp(input[j] - max))`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
float maxv = max(input);
float sum = 0.0f;
for (int i = 0; i < N; ++i) sum += expf(input[i] - maxv);
for (int i = 0; i < N; ++i) output[i] = expf(input[i] - maxv) / sum;
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define UPPER_DIV(A, B) ((A + B - 1) / B)
__global__ void block_max_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //same as reduction, but get max instead of sum
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] = fmaxf(sharedMem[threadIdx.x], sharedMem[threadIdx.x + stride]);
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockMax = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockMax = fmaxf(blockMax, __shfl_down_sync(0xffffffff, blockMax, stride));
        }
        if(threadIdx.x == 0){
            output[blockIdx.x] = blockMax;
        }
    }  
}
__global__ void softmax_kernel(const float* input, float* output, float* inputMax, double* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? __expf(input[idx] - inputMax[0]) : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockSum = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockSum += __shfl_down_sync(0xffffffff, blockSum, stride);
        }
        if(threadIdx.x == 0){
            atomicAdd(expSum, blockSum);
        }
    }
}
__global__ void softmax_tail_kernel(const float* input, float* output, float* inputMax, double* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N ) return;
    auto sum = *expSum;
    output[idx] = __expf(input[idx] - inputMax[0]) / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int blocksPerGrid = UPPER_DIV(N, BLOCK_SIZE);
    double *expSum;
    float *blockMax;
    cudaStream_t asyncStream;
    cudaStreamCreate(&asyncStream);
    cudaMallocAsync(&expSum, sizeof(double), asyncStream);
    cudaMallocAsync(&blockMax, sizeof(float) * blocksPerGrid, asyncStream);
    block_max_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, blockMax, N); 
    int blocks = UPPER_DIV(blocksPerGrid, BLOCK_SIZE);
    for(; blocks > 1 ; blocks = UPPER_DIV(blocks, BLOCK_SIZE)){
        block_max_kernel<<<blocks, BLOCK_SIZE, 0, asyncStream>>>(blockMax, blockMax, N);    
    }
    softmax_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, output, blockMax, expSum, N);
    softmax_tail_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, output, blockMax, expSum, N);
    cudaFreeAsync(expSum, asyncStream);
    cudaFreeAsync(blockMax, asyncStream);
    cudaStreamDestroy(asyncStream);
}
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

如果 N 不大，可以一个 block 干完整行；如果 N 很大，就要多 kernel 做 partial max、global max、partial sum、normalize。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

线程以 stride 方式遍历输入，先贡献局部 max，再贡献局部 sum，最后再写 output。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define UPPER_DIV(A, B) ((A + B - 1) / B)
__global__ void block_max_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //same as reduction, but get max instead of sum
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] = fmaxf(sharedMem[threadIdx.x], sharedMem[threadIdx.x + stride]);
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockMax = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockMax = fmaxf(blockMax, __shfl_down_sync(0xffffffff, blockMax, stride));
        }
        if(threadIdx.x == 0){
            output[blockIdx.x] = blockMax;
        }
    }  
}
__global__ void softmax_kernel(const float* input, float* output, float* inputMax, double* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? __expf(input[idx] - inputMax[0]) : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockSum = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockSum += __shfl_down_sync(0xffffffff, blockSum, stride);
        }
        if(threadIdx.x == 0){
            atomicAdd(expSum, blockSum);
        }
    }
}
__global__ void softmax_tail_kernel(const float* input, float* output, float* inputMax, double* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N ) return;
    auto sum = *expSum;
    output[idx] = __expf(input[idx] - inputMax[0]) / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int blocksPerGrid = UPPER_DIV(N, BLOCK_SIZE);
    double *expSum;
    float *blockMax;
    cudaStream_t asyncStream;
    cudaStreamCreate(&asyncStream);
    cudaMallocAsync(&expSum, sizeof(double), asyncStream);
    cudaMallocAsync(&blockMax, sizeof(float) * blocksPerGrid, asyncStream);
    block_max_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, blockMax, N); 
    int blocks = UPPER_DIV(blocksPerGrid, BLOCK_SIZE);
    for(; blocks > 1 ; blocks = UPPER_DIV(blocks, BLOCK_SIZE)){
        block_max_kernel<<<blocks, BLOCK_SIZE, 0, asyncStream>>>(blockMax, blockMax, N);    
    }
    softmax_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, output, blockMax, expSum, N);
    softmax_tail_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, output, blockMax, expSum, N);
    cudaFreeAsync(expSum, asyncStream);
    cudaFreeAsync(blockMax, asyncStream);
    cudaStreamDestroy(asyncStream);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程以 stride 方式遍历输入，先贡献局部 max，再贡献局部 sum，最后再写 output。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。
- host 侧还申请了中间 buffer，说明这题不是一个 kernel 直接结束，而是多个阶段串起来。buffer 生命周期要和计算图顺序一致。

### 版本 2

先看代码：

```cpp
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024
#define UPPER_DIV(A, B) ((A + B - 1) / B)
__device__ float atomic_max_float(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //same as reduction, but get max instead of sum
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] = fmaxf(sharedMem[threadIdx.x], sharedMem[threadIdx.x + stride]);
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockMax = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockMax = fmaxf(blockMax, __shfl_down_sync(0xffffffff, blockMax, stride));
        }
        if(threadIdx.x == 0){
            atomic_max_float(output, blockMax);
        }
    }  
}
__global__ void softmax_kernel(const float* input, float* output, float* inputMax, float* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? __expf(input[idx] - *inputMax) : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockSum = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockSum += __shfl_down_sync(0xffffffff, blockSum, stride);
        }
        if(threadIdx.x == 0){
            atomicAdd(expSum, blockSum);
        }
    }
}
__global__ void softmax_tail_kernel(const float* input, float* output, float* inputMax, float* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N ) return;
    auto sum = *expSum;
    output[idx] = __expf(input[idx] - *inputMax) / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int blocksPerGrid = UPPER_DIV(N, BLOCK_SIZE);
    float *expSum;
    float *inputMax;
    cudaMallocAsync(&expSum, sizeof(float), cudaStreamDefault);
    cudaMallocAsync(&inputMax, sizeof(float), cudaStreamDefault);
    max_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, cudaStreamDefault>>>(input, inputMax, N); 
    softmax_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, cudaStreamDefault>>>(input, output, inputMax, expSum, N);
    softmax_tail_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, cudaStreamDefault>>>(input, output, inputMax, expSum, N);
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。线程以 stride 方式遍历输入，先贡献局部 max，再贡献局部 sum，最后再写 output。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。
- 这里用了原子操作，原因是多个线程会更新同一个结果。原子能保正确，但竞争多的时候也会拖慢。
- host 侧还申请了中间 buffer，说明这题不是一个 kernel 直接结束，而是多个阶段串起来。buffer 生命周期要和计算图顺序一致。

## 六、性能和可改方向

这题我会优先看这些点：

- 减最大值是数值稳定，不是可选优化。
- block 内规约要用 shared memory 或 warp shuffle。
- 跨 block 归约会引入额外 kernel 或 atomic。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- 直接 expf(input[i]) 可能溢出。
- sum buffer 启动前要置零。
- max 初值不要用 0，负数输入会错，应该用 `-FLT_MAX` 或 `-inf`。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Softmax** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
