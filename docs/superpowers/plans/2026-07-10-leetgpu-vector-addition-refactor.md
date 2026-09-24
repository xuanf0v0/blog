# LeetGPU 向量加法样稿重构实施计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 LeetGPU 向量加法文章重构为结构清晰、事实克制且面向 CUDA 初学者的教程。

**架构：** 只重写 Markdown 正文，保留 frontmatter、题目链接和原有两段 CUDA 示例。用一处集中的索引推导替代各代码版本的重复叙述，并为 `float4` 优化明确列出对齐、尾部与性能限制。

**技术栈：** Astro 内容集合、Markdown、中文技术写作规范。

---

### 任务 1：重构向量加法文章正文

**文件：**
- 修改：`src/content/posts/leetgpu-vector-addition.md`
- 参考：`docs/superpowers/specs/2026-07-10-leetgpu-article-refactor-design.md`

- [x] **步骤 1：运行基线结构检查**

运行：

```powershell
$article = Get-Content -Raw 'src/content/posts/leetgpu-vector-addition.md'
@('**TL;DR：**', '## 适用读者与前置知识', '## 优化版本：float4') |
  ForEach-Object { if ($article -notmatch [regex]::Escape($_)) { "MISSING: $_" } }
```

预期：输出 3 个 `MISSING` 项，证明旧文章未符合已批准的结构。

- [x] **步骤 2：按规格重写正文**

保留 YAML frontmatter、题目链接、基础 kernel 和 `float4` 示例。将正文组织为：TL;DR、适用读者与前置知识、问题与数据布局、基础实现、索引与 launch 推导、优化版本、常见错误、小结。删除重复的版本回放和元叙述；不得添加未测量的性能数字或外部引用。

- [x] **步骤 3：运行结构和内容检查**

运行：

```powershell
$article = Get-Content -Raw 'src/content/posts/leetgpu-vector-addition.md'
$required = @('**TL;DR：**', '## 适用读者与前置知识', '## 问题与数据布局', '## 基础实现', '## 索引与 launch 推导', '## 优化版本：float4', '## 常见错误', '## 小结')
foreach ($section in $required) { if ($article -notmatch [regex]::Escape($section)) { throw "Missing section: $section" } }
if (($article | Select-String -AllMatches -Pattern 'int i = blockDim\.x \* blockIdx\.x \+ threadIdx\.x;').Matches.Count -gt 1) { throw 'Global index derivation is duplicated' }
if ($article -match '先不要急着优化|参考文里') { throw 'Removed boilerplate remains' }
if ($article -notmatch '16 字节对齐|16-byte alignment') { throw 'float4 alignment constraint is missing' }
Write-Output 'PASS: article satisfies the approved tutorial structure.'
```

预期：输出 `PASS`。

### 任务 2：验证内容集合解析

**文件：**
- 验证：`src/content/posts/leetgpu-vector-addition.md`

- [x] **步骤 1：运行 Astro 内容校验**

运行：

```powershell
pnpm check
```

预期：命令退出码为 0，且没有与 `leetgpu-vector-addition.md` 相关的 Markdown 或内容集合错误。

- [x] **步骤 2：检查定向差异**

运行：

```powershell
git diff --check -- src/content/posts/leetgpu-vector-addition.md
git diff -- src/content/posts/leetgpu-vector-addition.md
```

预期：无空白错误，且差异仅涉及该文章的正文重构。
