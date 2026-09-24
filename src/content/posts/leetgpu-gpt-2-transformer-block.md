---
title: 'GPT-2 Transformer 块'
published: 2026-07-08
description: 'GPT-2 Transformer 块 LeetGPU CUDA 刷题记录：参考博客园教程风格，记录题意、launch 配置、索引推导、代码实现、优化思路和踩坑点。'
image: '../../assets/images/cover.avif'
tags:
  - LeetGPU
  - CUDA
  - GPU 编程
  - 困难题
  - Transformer
  - 深度学习算子
category: LeetGPU
lang: zh-CN
sourceLink: 'https://leetgpu.com/challenges/gpt-2-transformer-block'
---

刷 LeetGPU 的时候，**GPT-2 Transformer 块** 属于很适合单独拎出来复盘的一题。它不像只贴答案那么无聊，真正有价值的是看清楚：题目数据怎么摆、CUDA 怎么启动线程、每个线程到底算哪一块，以及后面还能往哪里改。

题目链接：[https://leetgpu.com/challenges/gpt-2-transformer-block](https://leetgpu.com/challenges/gpt-2-transformer-block)

英文名可以先记一下：**GPT-2 Transformer Block**。后面看官方题目、CUDA sample 或 Triton tutorial，经常还是英文名更容易搜到资料。

## 一、GPT-2 Transformer Block

GPT-2 block 不是单个 kernel 难，而是整条计算图容易乱：LayerNorm、QKV、Attention、Projection、Residual、MLP 都要接对。

这题的核心关系可以写成：`LN1 -> Attention -> Residual -> LN2 -> MLP -> Residual`。先把这个关系想清楚，再去看 CUDA 代码，就不会被 `blockIdx`、`threadIdx` 这些名字绕晕。

如果用 CPU 串行写，大概就是下面这种感觉：

```cpp
ln1 = layer_norm(x)
attn = attention(linear_qkv(ln1))
res1 = x + linear(attn)
ln2 = layer_norm(res1)
output = res1 + mlp(ln2)
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
#include <cstddef>
#include <vector>

#ifndef ONLINE_JUDGE
#include <iostream>
#include <string>
#endif

constexpr int kDModel = 768;
constexpr int kNumHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kFfnDim = 3072;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kApproxCoeff = 0.044715f;
constexpr size_t kGamma1Offset = 0;
constexpr size_t kBeta1Offset = 768;
constexpr size_t kWqkvOffset = 1536;
constexpr size_t kBqkvOffset = 1771008;
constexpr size_t kWAttnOffset = 1773312;
constexpr size_t kBAttnOffset = 2363136;
constexpr size_t kGamma2Offset = 2363904;
constexpr size_t kBeta2Offset = 2364672;
constexpr size_t kWfcOffset = 2365440;
constexpr size_t kBfcOffset = 4724736;
constexpr size_t kWProjOffset = 4727808;
constexpr size_t kBProjOffset = 7087104;
constexpr size_t kWeightsSize = 7087872;
constexpr float kCompareEps = 2e-4f;

__global__ void layerNorm(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len || threadIdx.x != 0) {
    return;
  }

  const float* gamma = weights + kGamma1Offset;
  const float* beta = weights + kBeta1Offset;
  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  float* output_row = output + static_cast<size_t>(row) * kDModel;

  float mean = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    mean += input_row[i];
  }
  mean /= static_cast<float>(kDModel);

  float var = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    const float diff = input_row[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(kDModel);

  const float inv_std = rsqrtf(var + kLayerNormEps);
  for (int i = 0; i < kDModel; ++i) {
    output_row[i] = ((input_row[i] - mean) * inv_std) * gamma[i] + beta[i];
  }
}

__global__ void layerNorm2(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len || threadIdx.x != 0) {
    return;
  }

  const float* gamma = weights + kGamma2Offset;
  const float* beta = weights + kBeta2Offset;
  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  float* output_row = output + static_cast<size_t>(row) * kDModel;

  float mean = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    mean += input_row[i];
  }
  mean /= static_cast<float>(kDModel);

  float var = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    const float diff = input_row[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(kDModel);

  const float inv_std = rsqrtf(var + kLayerNormEps);
  for (int i = 0; i < kDModel; ++i) {
    output_row[i] = ((input_row[i] - mean) * inv_std) * gamma[i] + beta[i];
  }
}

__global__ void qkv(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len) {
    return;
  }

  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  const float* w_qkv = weights + kWqkvOffset;
  const float* b_qkv = weights + kBqkvOffset;
  float* output_row = output + static_cast<size_t>(row) * (kDModel * 3);

  for (int out_col = threadIdx.x; out_col < kDModel * 3; out_col += blockDim.x) {
    float sum = b_qkv[out_col];
    for (int in_col = 0; in_col < kDModel; ++in_col) {
      sum += input_row[in_col] * w_qkv[in_col * (kDModel * 3) + out_col];
    }
    output_row[out_col] = sum;
  }
}

__global__ void attn(const float* qkv, float* output, const float* weights, int seq_len) {
  (void)weights;
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= seq_len || head >= kNumHeads || lane >= kHeadDim) {
    return;
  }

  extern __shared__ float scores[];

  const int row_base = row * (kDModel * 3);
  const int q_base = row_base + head * kHeadDim;

  if (lane == 0) {
    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
      const int k_base = j * (kDModel * 3) + kDModel + head * kHeadDim;
      float dot = 0.0f;
      for (int d = 0; d < kHeadDim; ++d) {
        dot += qkv[q_base + d] * qkv[k_base + d];
      }
      const float score = dot / sqrtf(static_cast<float>(kHeadDim));
      scores[j] = score;
      max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
      scores[j] = expf(scores[j] - max_score);
      denom += scores[j];
    }

    const float inv_denom = 1.0f / denom;
    for (int j = 0; j < seq_len; ++j) {
      scores[j] *= inv_denom;
    }
  }
  __syncthreads();

  float acc = 0.0f;
  for (int j = 0; j < seq_len; ++j) {
    const int v_base = j * (kDModel * 3) + 2 * kDModel + head * kHeadDim;
    acc += scores[j] * qkv[v_base + lane];
  }
  output[row * kDModel + head * kHeadDim + lane] = acc;
}

__global__ void linear_bias(const float* input,
                            float* output,
                            const float* weight,
                            const float* bias,
                            int rows,
                            int in_dim,
                            int out_dim) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= out_dim) {
    return;
  }

  float sum = bias != nullptr ? bias[col] : 0.0f;
  const float* input_row = input + static_cast<size_t>(row) * in_dim;
  for (int k = 0; k < in_dim; ++k) {
    sum += input_row[k] * weight[k * out_dim + col];
  }
  output[static_cast<size_t>(row) * out_dim + col] = sum;
}

__device__ float gelu(float x) {
  const float cubic = x * x * x;
  const float inner = kSqrt2OverPi * (x + kApproxCoeff * cubic);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void ffn(const float* x, float* output, const float* weights, int seq_len) {
  const int row = blockIdx.x;
  if (row >= seq_len) {
    return;
  }

  __shared__ float up[kFfnDim];

  const float* input_row = x + static_cast<size_t>(row) * kDModel;
  const float* wfc = weights + kWfcOffset;
  const float* bfc = weights + kBfcOffset;
  const float* wproj = weights + kWProjOffset;
  const float* bproj = weights + kBProjOffset;
  float* output_row = output + static_cast<size_t>(row) * kDModel;

  for (int hidden = threadIdx.x; hidden < kFfnDim; hidden += blockDim.x) {
    float sum = bfc[hidden];
    for (int i = 0; i < kDModel; ++i) {
      sum += input_row[i] * wfc[i * kFfnDim + hidden];
    }
    up[hidden] = gelu(sum);
  }
  __syncthreads();

  for (int out_col = threadIdx.x; out_col < kDModel; out_col += blockDim.x) {
    float sum = bproj[out_col];
    for (int hidden = 0; hidden < kFfnDim; ++hidden) {
      sum += up[hidden] * wproj[hidden * kDModel + out_col];
    }
    output_row[out_col] = sum;
  }
}

__global__ void add_residual(const float* a, const float* b, float* output, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = a[idx] + b[idx];
  }
}

namespace {

std::vector<float> gpt2_block_reference(const std::vector<float>& x,
                                        const std::vector<float>& weights,
                                        int seq_len);

}  // namespace

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
  if (x == nullptr || output == nullptr || weights == nullptr || seq_len <= 0) {
    return;
  }

  const size_t token_count = static_cast<size_t>(seq_len) * kDModel;
  const size_t qkv_count = static_cast<size_t>(seq_len) * kDModel * 3;
  const size_t bytes = token_count * sizeof(float);
  const size_t qkv_bytes = qkv_count * sizeof(float);

  float* ln1 = nullptr;
  float* qkv_buf = nullptr;
  float* attn_concat = nullptr;
  float* attn_proj = nullptr;
  float* residual1 = nullptr;
  float* ln2 = nullptr;
  float* ff2 = nullptr;

  auto cleanup = [&]() {
    if (ln1 != nullptr) cudaFree(ln1);
    if (qkv_buf != nullptr) cudaFree(qkv_buf);
    if (attn_concat != nullptr) cudaFree(attn_concat);
    if (attn_proj != nullptr) cudaFree(attn_proj);
    if (residual1 != nullptr) cudaFree(residual1);
    if (ln2 != nullptr) cudaFree(ln2);
    if (ff2 != nullptr) cudaFree(ff2);
  };

  if (cudaMalloc(&ln1, bytes) != cudaSuccess ||
      cudaMalloc(&qkv_buf, qkv_bytes) != cudaSuccess ||
      cudaMalloc(&attn_concat, bytes) != cudaSuccess ||
      cudaMalloc(&attn_proj, bytes) != cudaSuccess ||
      cudaMalloc(&residual1, bytes) != cudaSuccess ||
      cudaMalloc(&ln2, bytes) != cudaSuccess ||
      cudaMalloc(&ff2, bytes) != cudaSuccess) {
    cleanup();
    return;
  }

  layerNorm<<<seq_len, 1>>>(x, ln1, weights, seq_len);
  qkv<<<seq_len, 256>>>(ln1, qkv_buf, weights, seq_len);
  attn<<<dim3(seq_len, kNumHeads), kHeadDim, static_cast<size_t>(seq_len) * sizeof(float)>>>(
      qkv_buf, attn_concat, weights, seq_len);

  const dim3 linear_grid((kDModel + 255) / 256, seq_len);
  linear_bias<<<linear_grid, 256>>>(
      attn_concat,
      attn_proj,
      weights + kWAttnOffset,
      weights + kBAttnOffset,
      seq_len,
      kDModel,
      kDModel);

  add_residual<<<static_cast<int>((token_count + 255) / 256), 256>>>(x, attn_proj, residual1,
                                                                      static_cast<int>(token_count));

  layerNorm2<<<seq_len, 1>>>(residual1, ln2, weights, seq_len);
  ffn<<<seq_len, 256>>>(ln2, ff2, weights, seq_len);

  add_residual<<<static_cast<int>((token_count + 255) / 256), 256>>>(residual1, ff2, output,
                                                                      static_cast<int>(token_count));

  cleanup();
}

#ifndef ONLINE_JUDGE
namespace {

float gelu_tanh(float x) {
  const float cubic = x * x * x;
  const float inner = kSqrt2OverPi * (x + kApproxCoeff * cubic);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

bool almost_equal(float a, float b) {
  return std::fabs(a - b) <= kCompareEps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

std::vector<float> make_patterned_data(size_t n, int period, float scale, float bias) {
  std::vector<float> values(n);
  for (size_t i = 0; i < n; ++i) {
    const int centered = static_cast<int>(i % static_cast<size_t>(period)) - (period / 2);
    values[i] = centered * scale + bias;
  }
  return values;
}

void layer_norm_row(const float* input,
                    const float* gamma,
                    const float* beta,
                    float* output) {
  float mean = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    mean += input[i];
  }
  mean /= static_cast<float>(kDModel);

  float var = 0.0f;
  for (int i = 0; i < kDModel; ++i) {
    const float diff = input[i] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(kDModel);

  const float inv_std = 1.0f / std::sqrt(var + kLayerNormEps);
  for (int i = 0; i < kDModel; ++i) {
    output[i] = ((input[i] - mean) * inv_std) * gamma[i] + beta[i];
  }
}

void matmul_add_bias(const float* input,
                     int rows,
                     int in_dim,
                     const float* weights,
                     int out_dim,
                     const float* bias,
                     float* output) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < out_dim; ++c) {
      float sum = bias != nullptr ? bias[c] : 0.0f;
      for (int k = 0; k < in_dim; ++k) {
        sum += input[r * in_dim + k] * weights[k * out_dim + c];
      }
      output[r * out_dim + c] = sum;
    }
  }
}

std::vector<float> gpt2_block_reference(const std::vector<float>& x,
                                        const std::vector<float>& weights,
                                        int seq_len) {
  const float* gamma1 = weights.data() + kGamma1Offset;
  const float* beta1 = weights.data() + kBeta1Offset;
  const float* wqkv = weights.data() + kWqkvOffset;
  const float* bqkv = weights.data() + kBqkvOffset;
  const float* wattn = weights.data() + kWAttnOffset;
  const float* battn = weights.data() + kBAttnOffset;
  const float* gamma2 = weights.data() + kGamma2Offset;
  const float* beta2 = weights.data() + kBeta2Offset;
  const float* wfc = weights.data() + kWfcOffset;
  const float* bfc = weights.data() + kBfcOffset;
  const float* wproj = weights.data() + kWProjOffset;
  const float* bproj = weights.data() + kBProjOffset;

  std::vector<float> ln1(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (int row = 0; row < seq_len; ++row) {
    layer_norm_row(x.data() + row * kDModel, gamma1, beta1, ln1.data() + row * kDModel);
  }

  std::vector<float> qkv(static_cast<size_t>(seq_len) * kDModel * 3, 0.0f);
  matmul_add_bias(ln1.data(), seq_len, kDModel, wqkv, kDModel * 3, bqkv, qkv.data());

  std::vector<float> attn_concat(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<float> scores(seq_len * seq_len, 0.0f);
  std::vector<float> weights_softmax(seq_len * seq_len, 0.0f);

  for (int head = 0; head < kNumHeads; ++head) {
    for (int i = 0; i < seq_len; ++i) {
      float max_score = -INFINITY;
      for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
          const int q_idx = i * (kDModel * 3) + head * kHeadDim + d;
          const int k_idx = j * (kDModel * 3) + kDModel + head * kHeadDim + d;
          dot += qkv[q_idx] * qkv[k_idx];
        }
        const float score = dot * scale;
        scores[i * seq_len + j] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      for (int j = 0; j < seq_len; ++j) {
        const float weight = std::exp(scores[i * seq_len + j] - max_score);
        weights_softmax[i * seq_len + j] = weight;
        denom += weight;
      }

      for (int d = 0; d < kHeadDim; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
          const int v_idx = j * (kDModel * 3) + 2 * kDModel + head * kHeadDim + d;
          acc += (weights_softmax[i * seq_len + j] / denom) * qkv[v_idx];
        }
        attn_concat[i * kDModel + head * kHeadDim + d] = acc;
      }
    }
  }

  std::vector<float> attn_proj(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  matmul_add_bias(attn_concat.data(), seq_len, kDModel, wattn, kDModel, battn, attn_proj.data());

  std::vector<float> residual1(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (size_t i = 0; i < residual1.size(); ++i) {
    residual1[i] = x[i] + attn_proj[i];
  }

  std::vector<float> ln2(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (int row = 0; row < seq_len; ++row) {
    layer_norm_row(residual1.data() + row * kDModel, gamma2, beta2, ln2.data() + row * kDModel);
  }

  std::vector<float> ff1(static_cast<size_t>(seq_len) * kFfnDim, 0.0f);
  matmul_add_bias(ln2.data(), seq_len, kDModel, wfc, kFfnDim, bfc, ff1.data());
  for (float& value : ff1) {
    value = gelu_tanh(value);
  }

  std::vector<float> ff2(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  matmul_add_bias(ff1.data(), seq_len, kFfnDim, wproj, kDModel, bproj, ff2.data());

  std::vector<float> output(static_cast<size_t>(seq_len) * kDModel, 0.0f);
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = residual1[i] + ff2[i];
  }
  return output;
}

bool run_case(const std::string& name,
              const std::vector<float>& x,
              const std::vector<float>& weights,
              int seq_len) {
  const size_t x_bytes = x.size() * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(seq_len) * kDModel * sizeof(float);
  const size_t weights_bytes = weights.size() * sizeof(float);
  const std::vector<float> expected = gpt2_block_reference(x, weights, seq_len);
  std::vector<float> actual(expected.size(), 0.0f);

  float* d_x = nullptr;
  float* d_out = nullptr;
  float* d_weights = nullptr;
  auto cleanup = [&]() {
    if (d_x != nullptr) cudaFree(d_x);
    if (d_out != nullptr) cudaFree(d_out);
    if (d_weights != nullptr) cudaFree(d_weights);
  };

  if (cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
      cudaMalloc(&d_out, out_bytes) != cudaSuccess ||
      cudaMalloc(&d_weights, weights_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": cudaMalloc failed\n";
    cleanup();
    return false;
  }

  if (cudaMemcpy(d_x, x.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_weights, weights.data(), weights_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemset(d_out, 0, out_bytes) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": device upload failed\n";
    cleanup();
    return false;
  }

  solve(d_x, d_out, d_weights, seq_len);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(actual.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "[FAIL] " << name << ": kernel execution failed\n";
    cleanup();
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!almost_equal(actual[i], expected[i])) {
      std::cerr << "[FAIL] " << name << ": mismatch at flat index " << i
                << ", expected=" << expected[i]
                << ", actual=" << actual[i] << '\n';
      cleanup();
      return false;
    }
  }

  cleanup();
  std::cout << "[PASS] " << name << '\n';
  return true;
}

std::vector<float> make_weights_pattern(float scale) {
  std::vector<float> weights(kWeightsSize, 0.0f);
  for (size_t i = 0; i < weights.size(); ++i) {
    const int centered = static_cast<int>(i % 17) - 8;
    weights[i] = centered * scale;
  }
  return weights;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int seq_len = 1;
    const std::vector<float> x =
        make_patterned_data(static_cast<size_t>(seq_len) * kDModel, 19, 0.01f, -0.02f);
    const std::vector<float> weights = make_weights_pattern(1e-4f);
    passed += run_case("single_token_reference", x, weights, seq_len) ? 1 : 0;
  }

  {
    ++total;
    const int seq_len = 2;
    const std::vector<float> x =
        make_patterned_data(static_cast<size_t>(seq_len) * kDModel, 23, 0.02f, 0.01f);
    const std::vector<float> weights = make_weights_pattern(8e-5f);
    passed += run_case("two_token_reference", x, weights, seq_len) ? 1 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
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

这里会连续 launch 多个 kernel。LayerNorm 按 token 行，linear 按矩阵元素，attention 按 query/head，residual 按 flat token_count。

参考文里反复提到 `threadsPerBlock`，这个值确实值得试。比如 128、256、512、1024 都可能出现不同表现；不是线程越多越快，因为寄存器、shared memory、warp 调度和访存形态都会影响最终结果。

一般写法是：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
```

这里的 `(work_items + threadsPerBlock - 1) / threadsPerBlock` 就是向上取整。多出来的线程会空转，所以 kernel 内部一定要有边界判断。

## 四、索引和地址怎么推

重点不是某一个 idx，而是每个中间 buffer 的 shape：`seq_len * hidden`、`seq_len * 3hidden`、`seq_len * heads * head_dim`。

CUDA 题大多数 bug 都藏在这里：不是公式不会，而是坐标换算成线性地址时乘错了步长。我的建议是每次都写出“逻辑坐标 -> 内存地址”的关系，再动手写代码。

## 五、代码版本回放

参考文的写法很好的一点是不会只给最终答案，而是会把尝试过程说出来。下面也按这个方式看：先跑通，再看能不能改。

### 版本 1

先看代码：

```cpp
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
  // allocate ln1 / qkv / attn / residual / ln2 / ffn buffers

  layerNorm<<<seq_len, 1>>>(x, ln1, weights, seq_len);
  qkv<<<seq_len, 256>>>(ln1, qkv_buf, weights, seq_len);

  attn<<<dim3(seq_len, kNumHeads), kHeadDim,
         static_cast<size_t>(seq_len) * sizeof(float)>>>(qkv_buf, attn_concat, weights, seq_len);

  linear_bias<<<linear_grid, 256>>>(
      attn_concat, attn_proj, weights + kWAttnOffset, weights + kBAttnOffset,
      seq_len, kDModel, kDModel);

  add_residual<<<(token_count + 255) / 256, 256>>>(x, attn_proj, residual1, token_count);
  layerNorm2<<<seq_len, 1>>>(residual1, ln2, weights, seq_len);
  ffn<<<seq_len, 256>>>(ln2, ff2, weights, seq_len);
  add_residual<<<(token_count + 255) / 256, 256>>>(residual1, ff2, output, token_count);
}
```

这段代码我会这样读：

- 这里用了二维或三维 launch。相比一维数组题，多出来的维度不是装饰，而是在表达输出张量本身的行、列、head 或 channel。

### 版本 2

先看代码：

```cpp
__global__ void attn(const float* qkv, float* output, const float* weights, int seq_len) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= seq_len || head >= kNumHeads || lane >= kHeadDim) return;

  extern __shared__ float scores[];

  if (lane == 0) {
    // compute full score row
    // softmax over seq_len
  }
  __syncthreads();

  float acc = 0.0f;
  for (int j = 0; j < seq_len; ++j) {
    acc += scores[j] * V(row=j, head, lane);
  }
  output[row * kDModel + head * kHeadDim + lane] = acc;
}
```

这段代码我会这样读：

- 索引这块可以按参考文里的思路看：先由 block 编号和 thread 编号算出逻辑位置。重点不是某一个 idx，而是每个中间 buffer 的 shape：`seq_len * hidden`、`seq_len * 3hidden`、`seq_len * heads * head_dim`。
- 出现 shared memory 说明代码已经不是单纯读全局内存了。通常是因为相邻线程会复用同一批数据，先搬进 block 内共享内存能减少重复读取。
- 只要用了 shared memory 并且后续线程要读别人写的数据，`__syncthreads()` 就不能随便删。它是 block 内的栅栏。

## 六、性能和可改方向

这题我会优先看这些点：

- 这一版更像把模型结构跑通。
- 优化方向是融合 kernel、减少中间全局内存落地、优化 attention。
- 权重 packed offset 必须严格对齐。

LeetGPU 的百分位有时候会受测试数据、GPU 架构、平台负载影响，所以不要只盯一个数字。更靠谱的做法是先根据访存模式判断瓶颈，再小步调整 block size、向量化、shared memory 或归约方式。

如果换成 PyTorch，很多题可能就是一行 API；如果换成 Triton，代码通常比 CUDA 少一些，尤其是矩阵和块级操作。但这篇主要还是看 CUDA，因为 CUDA 能把线程、内存和同步这些底层细节暴露得最清楚。

## 七、容易踩的坑

- offset 错一个，后面全错。
- residual 的位置不能随便调。
- cudaMalloc 失败要清理已分配 buffer。
- `cudaDeviceSynchronize()` 在 OJ 里方便暴露错误，但真实工程里不要无脑同步。
- 能删边界判断不代表应该删，除非你非常确定数据范围和 launch 刚好整除。

## 八、小结

这题最值得记住的不是某一行固定代码，而是这个顺序：先从 **GPT-2 Transformer Block** 的输入输出关系出发，确定输出空间，再设计 grid/block，最后把线程编号翻译成内存地址。

写 CUDA 很像写一篇严谨的账本：每个线程从哪里来、去哪儿、读什么、写什么，都要算得清清楚楚。算清楚以后再谈优化，才不会变成玄学调参。
