---
title: 'CUDA 实现 Max Pooling：从线程映射到 4D 索引展开'
published: 2026-07-07
description: '逐行拆解一个 NCHW 布局下的 CUDA Max Pooling 算子，理解输出尺寸、线程坐标、padding 边界和 4D 到 1D 的内存索引。'
image: '../../assets/images/cover.avif'
tags:
  - CUDA
  - GPU
  - 深度学习算子
  - 并行编程
category: CUDA
lang: zh-CN
---

Max Pooling 是卷积神经网络里非常常见的下采样算子。它的工作方式并不复杂：在输入特征图上滑动一个窗口，每个窗口只保留最大值。但真正把它写成 CUDA kernel 时，需要同时处理输出尺寸、线程映射、padding、边界检查和线性内存索引。

这篇文章完整回放下面这个 CUDA 实现。默认输入和输出都采用 `NCHW` 布局：

- `N`：batch size
- `C`：channel 数
- `H`：输入高度
- `W`：输入宽度

输入张量可以看成 `input[n][c][h][w]`，但在 GPU 内存里它实际是一段连续的一维数组。

## 完整代码

先把完整实现放在这里，后面再逐段拆开。

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
__global__ void maxpool_kernel(const float* input, float* output, int N, int C, int H, int W,
                                int kernel_size, int stride, int padding) {

    float max_v = -1e38f;

    // 1: 计算输出尺寸
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // 2: 计算当前线程位置
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;

    // 3: 计算输入起始位置
    int h_start = h * stride - padding;
    int w_start = w * stride - padding;

    // 4: 边界检查
    if (n >= N || c >= C || w >= W_out || h >= H_out) {
        return;
    }

    // 5: 从左上角开始遍历窗口
    for (int kh = 0; kh < kernel_size; kh++) {
        int h_idx = h_start + kh;
        if (h_idx < 0 || h_idx >= H)
            continue;

        for (int kw = 0; kw < kernel_size; kw++) {
            int w_idx = w_start + kw;
            if (w_idx < 0 || w_idx >= W)
                continue;

            int input_idx = ((n * C + c) * H + h_idx) * W + w_idx;
            float val = input[input_idx];

            if (val > max_v) {
                max_v = val;
            }
        }
    }

    // 6: 写入输出
    int output_idx = ((n * C + c) * H_out + h) * W_out + w;
    output[output_idx] = max_v;
}

extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {

    // 计算输出尺寸
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // 设置 CUDA 网格和块
    dim3 blockDim(16, 16);
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                 (H_out + blockDim.y - 1) / blockDim.y,
                 N * C);

    // 启动内核
    maxpool_kernel<<<gridDim, blockDim>>>(input, output, N, C, H, W, kernel_size, stride, padding);

    // 同步
    cudaDeviceSynchronize();
}
```

## 1. Kernel 函数负责什么

```cpp
__global__ void maxpool_kernel(const float* input, float* output, int N, int C, int H, int W,
                                int kernel_size, int stride, int padding)
```

`__global__` 表示这是一个 CUDA kernel。它由 CPU 端发起调用，但实际在 GPU 上执行。

参数可以分成三组：

- `input` 和 `output`：GPU 端内存指针，也就是 device pointer
- `N`、`C`、`H`、`W`：输入张量形状
- `kernel_size`、`stride`、`padding`：池化窗口大小、滑动步长和 padding 大小

这个 kernel 的设计是：一个 CUDA 线程负责计算一个输出元素。也就是说，一个线程对应 `output[n][c][h][w]` 中的一个位置。

## 2. 先给最大值一个足够小的初始值

```cpp
float max_v = -1e38f;
```

Max Pooling 要在窗口内找最大值，所以需要一个初始值。这里用 `-1e38f` 作为非常小的 float 值。

之后每读到一个有效输入元素，就执行：

```cpp
if (val > max_v) {
    max_v = val;
}
```

这样遍历完整个窗口后，`max_v` 就是当前池化窗口里的最大值。

## 3. 输出尺寸怎么计算

```cpp
int H_out = (H + 2 * padding - kernel_size) / stride + 1;
int W_out = (W + 2 * padding - kernel_size) / stride + 1;
```

池化窗口不是只放一次，而是在输入特征图上滑动。以高度方向为例：

```text
输出高度 = (输入高度 + 上下 padding - 池化核大小) / stride + 1
```

对应代码就是：

```cpp
H_out = (H + 2 * padding - kernel_size) / stride + 1;
```

宽度方向同理：

```cpp
W_out = (W + 2 * padding - kernel_size) / stride + 1;
```

举个例子，如果输入是 `H = 5, W = 5`，`kernel_size = 2`，`stride = 2`，`padding = 0`，那么：

```text
H_out = (5 + 0 - 2) / 2 + 1 = 2
W_out = (5 + 0 - 2) / 2 + 1 = 2
```

输出就是 `2 x 2`。

如果 `padding = 1`，那么：

```text
H_out = (5 + 2 - 2) / 2 + 1 = 3
W_out = (5 + 2 - 2) / 2 + 1 = 3
```

padding 会让可滑动区域变大，所以输出尺寸也可能变大。

## 4. 线程坐标如何映射到输出坐标

```cpp
int w = blockDim.x * blockIdx.x + threadIdx.x;
int h = blockDim.y * blockIdx.y + threadIdx.y;
int c = blockIdx.z % C;
int n = blockIdx.z / C;
```

这一段是整个 kernel 的核心：把 CUDA 的线程编号转换成输出张量的四维坐标。

### x 方向负责输出列

```cpp
int w = blockDim.x * blockIdx.x + threadIdx.x;
```

`blockIdx.x` 表示当前 block 在 x 方向是第几个 block，`threadIdx.x` 表示当前线程在 block 内部的 x 坐标。

所以：

```text
全局 w 坐标 = 每个 block 的 x 方向线程数 * block 的 x 编号 + 线程在 block 内的 x 编号
```

它对应输出特征图的列号。

### y 方向负责输出行

```cpp
int h = blockDim.y * blockIdx.y + threadIdx.y;
```

同理，`h` 对应输出特征图的行号。

### z 方向同时承载 batch 和 channel

```cpp
int c = blockIdx.z % C;
int n = blockIdx.z / C;
```

`gridDim.z` 被设置成 `N * C`，所以 z 方向的每一层对应一个 `(n, c)` 组合。

假设 `N = 2`，`C = 3`，那么 `blockIdx.z` 的含义是：

| `blockIdx.z` | `n = blockIdx.z / C` | `c = blockIdx.z % C` |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 1 | 0 |
| 4 | 1 | 1 |
| 5 | 1 | 2 |

这样就用一个三维 CUDA grid 覆盖了四维输出张量：`N x C x H_out x W_out`。

## 5. 从输出位置反推输入窗口左上角

```cpp
int h_start = h * stride - padding;
int w_start = w * stride - padding;
```

输出里的一个点，对应输入上的一个池化窗口。窗口左上角的位置由输出坐标、stride 和 padding 决定。

如果没有 padding，输出位置 `(h, w)` 对应的输入起点就是：

```text
h_start = h * stride
w_start = w * stride
```

加入 padding 后，逻辑上输入周围多了一圈空白区域。为了把这个逻辑坐标映射回真实输入，需要减掉 padding：

```text
h_start = h * stride - padding
w_start = w * stride - padding
```

例如 `stride = 2, padding = 1`：

| 输出坐标 `(h, w)` | 输入窗口起点 `(h_start, w_start)` |
| --- | --- |
| `(0, 0)` | `(-1, -1)` |
| `(0, 1)` | `(-1, 1)` |
| `(1, 0)` | `(1, -1)` |
| `(1, 1)` | `(1, 1)` |

出现负数不是错误，它表示窗口的一部分落在 padding 区域里。后面的边界检查会跳过这些无效位置。

## 6. 为什么需要输出边界检查

```cpp
if (n >= N || c >= C || w >= W_out || h >= H_out) {
    return;
}
```

CUDA block 通常按固定大小分配，比如这里每个 block 是 `16 x 16`。但输出尺寸不一定刚好是 16 的倍数。

例如 `W_out = 30`，x 方向需要：

```cpp
(W_out + blockDim.x - 1) / blockDim.x
```

也就是：

```text
(30 + 16 - 1) / 16 = 45 / 16 = 2
```

两个 block 一共能覆盖 `32` 个 x 坐标，其中 `w = 30` 和 `w = 31` 超出了真实输出宽度。因此 kernel 内部必须检查：

```cpp
w >= W_out
```

高度方向同理。`n >= N` 和 `c >= C` 是为了保证 batch 和 channel 不越界。

## 7. 遍历池化窗口

```cpp
for (int kh = 0; kh < kernel_size; kh++) {
    int h_idx = h_start + kh;
    if (h_idx < 0 || h_idx >= H)
        continue;

    for (int kw = 0; kw < kernel_size; kw++) {
        int w_idx = w_start + kw;
        if (w_idx < 0 || w_idx >= W)
            continue;

        int input_idx = ((n * C + c) * H + h_idx) * W + w_idx;
        float val = input[input_idx];

        if (val > max_v) {
            max_v = val;
        }
    }
}
```

`kh` 和 `kw` 是池化窗口内部的偏移。

如果 `kernel_size = 3`，那么窗口内部会访问这些相对位置：

```text
(0, 0)  (0, 1)  (0, 2)
(1, 0)  (1, 1)  (1, 2)
(2, 0)  (2, 1)  (2, 2)
```

真实输入坐标由窗口起点加上偏移得到：

```cpp
int h_idx = h_start + kh;
int w_idx = w_start + kw;
```

因为 padding 可能让窗口落到输入外面，所以访问前必须判断：

```cpp
if (h_idx < 0 || h_idx >= H)
    continue;
```

以及：

```cpp
if (w_idx < 0 || w_idx >= W)
    continue;
```

这两段判断保证只读取真实输入范围内的数据。

## 8. 4D 坐标如何转换成 1D 内存地址

```cpp
int input_idx = ((n * C + c) * H + h_idx) * W + w_idx;
float val = input[input_idx];
```

这是最容易写错的一行。

虽然逻辑上输入是四维的：

```text
input[n][c][h_idx][w_idx]
```

但实际内存是一维连续数组。`NCHW` 布局的展开顺序是：

```text
先选 batch，再选 channel，再选行，再选列
```

公式是：

```text
index = ((n * C + c) * H + h) * W + w
```

可以分步理解：

```text
n * C + c
```

先定位到第 `n` 个 batch 里的第 `c` 个 channel。

```text
(n * C + c) * H + h
```

再定位到这个 channel 里的第 `h` 行。

```text
((n * C + c) * H + h) * W + w
```

最后定位到这一行里的第 `w` 列。

所以代码里的：

```cpp
int input_idx = ((n * C + c) * H + h_idx) * W + w_idx;
```

就是在读取当前窗口内的一个真实输入元素。

## 9. 写回输出

```cpp
int output_idx = ((n * C + c) * H_out + h) * W_out + w;
output[output_idx] = max_v;
```

输出也是 `NCHW` 布局，只不过高度和宽度换成了 `H_out`、`W_out`。

所以输出索引公式是：

```text
output_idx = ((n * C + c) * H_out + h) * W_out + w
```

当前线程只负责一个输出元素，因此最后直接写入：

```cpp
output[output_idx] = max_v;
```

到这里，一个线程的工作就完成了。

## 10. Host 端如何启动 kernel

```cpp
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
```

`solve` 是 CPU 端入口函数。它负责计算输出尺寸、配置 CUDA grid/block，并启动 kernel。

`extern "C"` 的作用是关闭 C++ name mangling，让外部系统可以用稳定的 C 风格函数名找到 `solve`。

### 再算一次输出尺寸

```cpp
int H_out = (H + 2 * padding - kernel_size) / stride + 1;
int W_out = (W + 2 * padding - kernel_size) / stride + 1;
```

host 端也需要输出尺寸，因为它要根据输出大小决定启动多少线程。

### 每个 block 使用 16 x 16 个线程

```cpp
dim3 blockDim(16, 16);
```

这里每个 block 有：

```text
16 * 16 = 256
```

个线程。每个线程计算一个输出位置，所以一个 block 最多覆盖输出平面上的 `16 x 16` 个点。

### gridDim 覆盖整个输出张量

```cpp
dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
             (H_out + blockDim.y - 1) / blockDim.y,
             N * C);
```

x 方向覆盖输出宽度：

```cpp
(W_out + blockDim.x - 1) / blockDim.x
```

y 方向覆盖输出高度：

```cpp
(H_out + blockDim.y - 1) / blockDim.y
```

这是一种常见的向上取整写法。它保证即使 `W_out` 或 `H_out` 不是 block 尺寸的整数倍，也能启动足够多的线程。

z 方向覆盖所有 batch 和 channel：

```cpp
N * C
```

因此，整个 grid 覆盖的是：

```text
N * C 个输出平面，每个平面大小为 H_out * W_out
```

### 启动 kernel 并同步

```cpp
maxpool_kernel<<<gridDim, blockDim>>>(input, output, N, C, H, W, kernel_size, stride, padding);
cudaDeviceSynchronize();
```

第一行启动 GPU kernel。第二行等待 GPU 执行完成。

如果没有 `cudaDeviceSynchronize()`，CPU 端可能在 kernel 还没执行完时就继续往下走。对于在线评测、教学代码或需要马上读取结果的场景，同步可以让行为更明确。

## 11. 用一个小例子串起来

假设输入是：

```text
N = 1
C = 1
H = 4
W = 4
kernel_size = 2
stride = 2
padding = 0
```

输入矩阵是：

```text
1   2   5   6
3   4   7   8
9   10  13  14
11  12  15  16
```

输出尺寸：

```text
H_out = (4 + 0 - 2) / 2 + 1 = 2
W_out = (4 + 0 - 2) / 2 + 1 = 2
```

输出位置 `(h = 0, w = 0)` 的窗口起点是：

```text
h_start = 0 * 2 - 0 = 0
w_start = 0 * 2 - 0 = 0
```

窗口覆盖：

```text
1  2
3  4
```

最大值是 `4`。

输出位置 `(h = 0, w = 1)` 的窗口起点是：

```text
h_start = 0
w_start = 1 * 2 = 2
```

窗口覆盖：

```text
5  6
7  8
```

最大值是 `8`。

最终输出是：

```text
4   8
12  16
```

这正是 Max Pooling 的结果。

## 12. 这份实现的关键点

这段 CUDA Max Pooling 代码可以总结成一条主线：

```text
一个线程 -> 一个输出元素 -> 一个池化窗口 -> 一个最大值
```

具体步骤是：

1. 根据输入尺寸、padding、kernel size 和 stride 算出输出尺寸。
2. 用 `blockIdx` 和 `threadIdx` 算出当前线程对应的输出坐标 `(n, c, h, w)`。
3. 用输出坐标反推输入窗口左上角 `(h_start, w_start)`。
4. 遍历窗口里的每个元素，跳过 padding 造成的越界位置。
5. 用 `((n * C + c) * H + h) * W + w` 把 4D 坐标转换成 1D 地址。
6. 找到最大值后写入对应的输出地址。

理解这几个步骤后，再看卷积、平均池化、归约类算子时，思路会清晰很多：先确定“一个线程负责什么”，再确定“线程坐标如何映射到数据坐标”，最后把多维坐标正确展开成线性内存地址。
