---
title: '矩阵乘法'
published: 2026-07-08
description: '矩阵乘法 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 简单题
  - 矩阵计算
  - 全局内存
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/matrix-multiplication'
---

刷 LeetGPU 的时候，**矩阵乘法** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/matrix-multiplication](https://leetgpu.com/challenges/matrix-multiplication)

英文名可以先记一下：**Matrix Multiplication**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、Matrix Multiplication

矩阵乘法是 CUDA 入门后第一个真正需要二维 grid 的题。C = A * B，A 是 M x N，B 是 N x K，输出 C 是 M x K。

这题的核心关系可以写成：`C[row, col] = sum(A[row, i] * B[i, col])`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
for (int row = 0; row < M; ++row)
  for (int col = 0; col < K; ++col) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
      sum += A[row * N + i] * B[i * K + col];
    C[row * K + col] = sum;
  }
```

CPU 写法的好处是直观，坏处也很明显：大量元素只能一轮一轮处理。GPU 版本要做的，就是把这些彼此独立的工作拆给很多线程。

## 二、CUDA 代码先跑起来

LeetGPU 现在给到的参数基本都是已经在 GPU 上申请好的 device pointer，也就是说我们不用在题解里写 `cudaMalloc`、`cudaMemcpy`、`cudaFree` 这些模板代码，重点都落在 kernel 怎么写。

<details>
<summary>点击展开本题主要 CUDA 实现</summary>

```cpp
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < K){
        float sum = 0.0f;
        #pragma unroll
        for(int r = 0; r < N; r++){
            sum += A[row * N + r] * B[r * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

</details>

先不要急着优化。我的习惯是先确认三件事：

- kernel launch 的 grid/block 覆盖了全部输出。
- 每个线程能算出自己负责的逻辑位置。
- 所有读写前都有边界保护，最后一个 block 不会越界。

## 三、launch 配置怎么理解

这里用二维 block 比一维更自然。一般让 x 方向对应输出列 col，y 方向对应输出行 row，例如 `dim3 block(16, 16)`，grid 分别对 K 和 M 向上取整。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

`col = blockIdx.x * blockDim.x + threadIdx.x`，`row = blockIdx.y * blockDim.y + threadIdx.y`。这个顺序有利于相邻线程访问相邻列，写 C 时也更规整。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < K){
        float sum = 0.0f;
        #pragma unroll
        for(int r = 0; r < N; r++){
            sum += A[row * N + r] * B[r * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

这段代码我会这样读：

- 这里用了二维或三维 launch。相比一维数组题，多出来的维度不是装饰，而是在表达输出张量本身的行、列、head 或 channel。
- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。`col = blockIdx.x * blockDim.x + threadIdx.x`，`row = blockIdx.y * blockDim.y + threadIdx.y`。这个顺序有利于相邻线程访问相邻列，写 C 时也更规整。

## 六、性能和可改方向

这题我会优先看这些点：

- 朴素版本每个线程算一个 C 元素，容易写对。
- 如果 row/col 映射反了，逻辑可能仍对，但访存合并会变差，参考文里这种写法甚至会超时。
- 进一步优化通常用 shared memory tile 或 Tensor Core/CUTLASS。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- A 的行步长是 N，B/C 的行步长是 K，别混。
- 二维边界要同时判断 row 和 col。
- 不要只看代码结果对不对，矩阵乘法的访存顺序会极大影响速度。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **Matrix Multiplication** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
