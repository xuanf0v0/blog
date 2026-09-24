# LeetGPU 技术博客批量重构实施计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将余下 22 篇 LeetGPU 文章重构为与向量加法样稿一致的中文 CUDA 教程。

**架构：** 每篇仅修改 Markdown 正文，保留 frontmatter、`sourceLink` 和原有题目代码的算法意图。按题型分为三个无重叠文件组；每篇添加 TL;DR、读者前提、问题/数据布局、实现、集中推导、优化限制、常见错误与小结。主线程最后运行内容集合校验。

**技术栈：** Astro 内容集合、Markdown、CUDA C++、中文技术写作规范。

---

## 共同约束

- 不修改 YAML frontmatter 字段或题目链接。
- 不捏造基准、平台百分位、硬件结果或外部来源。
- 删除「先不要急着优化」「参考文里」及重复的版本回放叙述。
- 代码围栏使用 `cpp`；只有在原文能支持时才保留或改写代码。
- 对优化说明写明前置条件与不适用场景，例如边界、对齐、同步、数值稳定性或内存开销。
- 不提交文章：它们已在用户的暂存集合中。

### 任务 1：重构基础与逐元素算子文章

**文件：**
- 修改：`src/content/posts/leetgpu-color-inversion.md`
- 修改：`src/content/posts/leetgpu-count-2d-array-element.md`
- 修改：`src/content/posts/leetgpu-count-array-element.md`
- 修改：`src/content/posts/leetgpu-leaky-relu.md`
- 修改：`src/content/posts/leetgpu-matrix-copy.md`
- 修改：`src/content/posts/leetgpu-rainbow-table.md`
- 修改：`src/content/posts/leetgpu-relu.md`
- 修改：`src/content/posts/leetgpu-reverse-array.md`

- [x] 为每篇建立和样稿一致的教程结构，围绕其真实索引或数据布局解释基础 kernel。
- [x] 保留原有代码意图，去除重复段落与无依据的性能断言。
- [x] 对每篇运行 Markdown 结构检查，确认存在 `TL;DR`、读者前提、优化限制、常见错误与小结。

### 任务 2：重构规约、卷积与矩阵文章

**文件：**
- 修改：`src/content/posts/leetgpu-1d-convolution.md`
- 修改：`src/content/posts/leetgpu-dot-product.md`
- 修改：`src/content/posts/leetgpu-matrix-multiplication.md`
- 修改：`src/content/posts/leetgpu-matrix-transpose.md`
- 修改：`src/content/posts/leetgpu-parallel-merge.md`
- 修改：`src/content/posts/leetgpu-reduction.md`
- 修改：`src/content/posts/leetgpu-softmax.md`

- [x] 为每篇建立教程结构；将 shape、步长、线程映射、同步或数值稳定性放入集中推导章节。
- [x] 只保留能解释算法的代表性代码版本；明确 shared memory、bank conflict、归约顺序或数值误差等限制条件。
- [x] 对每篇运行 Markdown 结构检查，确认存在 `TL;DR`、读者前提、优化限制、常见错误与小结。

### 任务 3：重构归一化、量化与 Transformer 文章

**文件：**
- 修改：`src/content/posts/leetgpu-batch-normalization.md`
- 修改：`src/content/posts/leetgpu-gpt-2-transformer-block.md`
- 修改：`src/content/posts/leetgpu-int8-quantized-matmul.md`
- 修改：`src/content/posts/leetgpu-multi-head-attention.md`
- 修改：`src/content/posts/leetgpu-rms-normalization.md`
- 修改：`src/content/posts/leetgpu-rotary-positional-embedding.md`
- 修改：`src/content/posts/leetgpu-weight-dequantization.md`

- [x] 为每篇建立教程结构；明确张量 shape、权重布局、并行维度与中间 buffer 的责任。
- [x] 保留代表性代码并删除重复版本回放；对精度、数值稳定性、内存占用、序列长度和硬件依赖写出限制条件。
- [x] 对每篇运行 Markdown 结构检查，确认存在 `TL;DR`、读者前提、优化限制、常见错误与小结。

### 任务 4：统一验证

**文件：**
- 验证：`src/content/posts/leetgpu-*.md`

- [x] 运行 PowerShell 检查，确保 23 篇文章均含 `TL;DR`、`适用读者与前置知识`、`常见错误` 和 `小结`，且没有移除的套话。
- [x] 运行 `git diff --check -- src/content/posts`，确认无空白错误。
- [x] 运行 `pnpm check`，确认 Astro 内容集合与类型校验退出码为 0。
