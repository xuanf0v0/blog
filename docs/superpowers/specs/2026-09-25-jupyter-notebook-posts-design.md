# Jupyter Notebook 文章支持 — 设计

日期：2026-09-25
状态：待评审

## 1. 背景与目标

博客目前的文章只有 `.md` / `.mdx`，由 `src/content.config.ts` 中的 `glob()` loader
加载。目标：让 `src/content/posts/**/*.ipynb` 与手写 markdown 文章**完全等价**地渲染 ——
同样的 URL 规则、同样的代码块样式（expressive-code 的行号/语言徽章/折叠/复制按钮）、
同样的 KaTeX、mermaid、PlantUML、callouts、图片优化与 LQIP、同样的归档/标签/分类/相关文章/RSS。

不执行 notebook。渲染 notebook 中已保存的输出。

## 2. 范围

### 做

- 解析 `.ipynb`，渲染 markdown 单元格、代码单元格与其全部输出
- 元数据：notebook `metadata` 打底，首单元格 YAML 覆盖
- 输出类型：`stream` / `execute_result` / `display_data` / `error`
- 富输出：`text/plain`、`text/html`、`image/png|jpeg|svg+xml`、`text/latex`、`application/json`
- ANSI SGR 颜色转 HTML span
- cell `attachments`（Jupyter 里粘贴的图片）
- base64 图片抽取为真实资源文件，接入 Astro 图片优化与 LQIP
- dev 下热更新（改 notebook 即重渲染，含新增/删除）

### 不做（YAGNI）

- 构建期执行 notebook（需要 Python/内核环境）
- Colab / Binder 按钮、下载 `.ipynb` 按钮
- MyST 交叉引用、自动章节编号、定理/练习环境
- notebook 内的交互式控件（ipywidgets）

## 3. 关键决策与依据

| 决策 | 依据 |
|---|---|
| 生成 `.md` 后交给官方 `glob()`，不自己写 loader 逻辑 | `glob()` 的 `load()` 开头快照 `store.keys()`，结尾 `untouchedEntries.forEach((id) => store.delete(id))`（`node_modules/astro/dist/content/loaders/glob.js:225`）。若「包装 glob 再追加 entry」，追加的 entry 会在下次同步被当作 stale 删除后重建，dev 下会闪 404；且需复刻 MDX 的 `getRenderFunction` / `assetImports`。 |
| 生成物与手写文章同目录、同名 | 路由用 `removeFileExtension(entry.id)` 生成 URL（`src/pages/posts/[...slug].astro:31`），而 `id` 来自文件相对路径。同目录同名的生成物天然得到正确的 id/slug/URL，无需覆写 `generateId`（避免复刻 `getContentEntryIdAndSlug` 的 slugify 与 `/index` 规则）。 |
| 不注册自定义 entry type | `settings.contentEntryTypes` 硬编码为 `[markdownContentEntryType]`（`node_modules/astro/dist/core/config/settings.js:37`），无公开注册 API。因此 `glob({ pattern: "**/*.{md,mdx,ipynb}" })` 只会对 `.ipynb` 报 "No entry type found"。 |
| 生成物不入库，用自动维护的 `.gitignore` 管理 | 生成物与手写 md 同名同目录，无法用全局模式区分；由生成器在被忽略目录内维护一份按文件名列举的 `.gitignore`，同时充当清理清单。 |
| 构建顺序：生成 notebook 必须在 LQIP 之前 | `scripts/generate-lqips.ts:80` 扫描 `{src,public}/**/*.{png,jpg,jpeg,webp,avif}`；notebook 配图若晚于该脚本生成，将永远没有 LQIP。`pnpm build` 现为 icons → lqips → astro build → pagefind，需插入生成步骤。 |
| 不执行 notebook 也能拿到完整样式 | `ctx.renderMarkdown()` → `config.markdown.processor`（`node_modules/astro/dist/content/content-layer.js:107`），即项目 `astro.config.mjs` 里那条 unified 管线；expressive-code 通过向 `markdownProcessor.options.rehypePlugins` 注入生效（`node_modules/astro-expressive-code/dist/index.js:501-502`）。因此 notebook 的 markdown 与代码块自动获得全部既有能力。 |

## 4. 架构与数据流

```
foo.ipynb ──(转换器)──▶ src/content/posts/foo.md ──(官方 glob)──▶ content collection
                              │                                        │
                              │                                  现有 unified 管线
                              │                                  (KaTeX / expressive-code /
                              │                                   mermaid / callouts / …)
                              ▼                                        ▼
                  src/content/posts/.gitignore                    <Content />
                   （自动维护的产物清单）
```

`src/assets/notebooks/<slug>/<hash>.<ext>` ◀── 转换器抽取的 base64 图片；
正文以 `../../assets/notebooks/<slug>/<hash>.png` 相对路径引用（与现有
`../../assets/images/cover.avif` 写法一致），从而被 Astro 图片优化与 LQIP 覆盖。

## 5. 模块设计

新增目录 `src/utils/notebook/`，每个模块单一职责：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `parseNotebook.ts` | 读文件、`JSON.parse`、校验 `nbformat` 与 `cells` 结构；返回类型化对象 | 无 |
| `resolveMetadata.ts` | 合并 notebook `metadata` 与首单元格 YAML；校验必填字段 | `@astrojs/markdown-remark` 的 `parseFrontmatter` |
| `cellToMarkdown.ts` | markdown 单元格 → markdown；code 单元格 → 围栏代码块 + 输出 | `outputToMarkdown` |
| `outputToMarkdown.ts` | 输出项 → markdown/HTML 片段 | `ansi.ts`、`extractAssets` |
| `ansi.ts` | SGR 序列 → `<span style>`；未知序列丢弃 | 无 |
| `extractAssets.ts` | base64 → 文件（内容 hash 命名、幂等） | `node:crypto`、`node:fs` |
| `convertNotebook.ts` | 编排上述模块：`convertNotebook(ipynbPath, { assetsDir })` → `{ frontmatter, markdown }`；图片落地是过程中的副作用（幂等），编排层不重复实现 | 上述全部 |

新增 `src/loaders/notebook-posts.ts`：

```ts
export function notebookPosts(options?: { base?: string }): Loader
```

`load(ctx)` 的步骤：

1. 调 `generateNotebooks()` 同步生成物（见 §9）
2. `await inner.load(ctx)`，`inner = glob({ pattern: "**/*.{md,mdx}", base })`
3. 注册 `*.ipynb` 监听（**只注册一次**，`load` 每次同步都会被调用）

新增 `scripts/generate-notebooks.ts`：调用同一个生成函数，供命令行与构建脚本使用。

## 6. 元数据规范

优先级：**首单元格 YAML > `metadata.blog` > 默认值**。

notebook 原生元数据（Jupyter → Edit → Edit Notebook Metadata）：

```json
{
  "metadata": {
    "blog": {
      "title": "Softmax 归一化",
      "published": "2026-07-08",
      "tags": ["CUDA", "LeetGPU"],
      "category": "LeetGPU",
      "description": "...",
      "draft": false,
      "pinned": false,
      "comment": true
    }
  }
}
```

首单元格（必须是第一个单元格且为 markdown，内容以 `---` 开头）：

````markdown
---
title: Softmax 归一化
published: 2026-07-08
tags: [CUDA, LeetGPU]
---

正文从这里开始……
````

字段与 `src/content.config.ts` 的 posts schema 一一对应，可取值为
`title` / `published` / `updated` / `draft` / `description` / `image` / `tags` /
`category` / `lang` / `pinned` / `author` / `sourceLink` / `licenseName` /
`licenseUrl` / `comment` / `password` / `passwordHint`。

默认值：`title` 缺失时回退为文件名（去扩展名、连字符转空格，确定性可复现）；
其余字段沿用 schema 默认值。

**`published` 必须显式声明**，不提供任何自动兜底：`git` 不保留文件 mtime，
以 mtime 或构建时间兜底会让同一篇 notebook 在不同机器与 CI 上得到不同日期，
进而让文章排序与 RSS 漂移。缺失或非法日期按 §10 报错。

被消费的首单元格**不进入正文**。

## 7. 转换规则

### 单元格

| 类型 | 处理 |
|---|---|
| `markdown` | `source` 按行拼接后原样输出；`attachments` 中的图片先在正文定位替换为抽取后的路径，未能定位的追加到文末 |
| `code` | 围栏代码块（语言见下）+ 每条输出；`execution_count` 不输出 |
| `raw` | 跳过 |

代码块语言：`metadata.language_info.name` → `metadata.kernelspec.language` → `python`。
做小写归一与常见别名映射（`python3`→`python`、`c++`→`cpp`、`c#`→`csharp`）。

### 输出

| 输出类型 | MIME | 处理 |
|---|---|---|
| `stream` | — | 文本 → markdown 块（见下方 ANSI 规则）；`stderr` 额外包 `<div class="nb-stderr">` |
| `execute_result` / `display_data` | `text/html` | 原样 HTML，包 `<div class="nb-output-html">` |
| 同上 | `image/png` `image/jpeg` | 抽为资源文件，输出 `![...](相对路径)` |
| 同上 | `image/svg+xml` | 原样内联（SVG 无需优化，也避免 sharp 处理） |
| 同上 | `text/latex` | `$$...$$`，交给现有 remark-math + KaTeX |
| 同上 | `application/json` | 围栏 `json` 代码块 |
| 同上 | `text/plain` | 文本 → markdown 块（见下方 ANSI 规则） |
| `error` | — | 文本 → markdown 块，内容为 `ename: evalue` + `traceback` |

同一输出含多种 MIME 时按上表自上而下取第一个可用者。

**ANSI 与容器选择**：文本输出的容器取决于是否含 ANSI 转义序列。
含 `\u001b[` 时用 `<pre class="nb-output">` 原生 HTML 包裹，由 `ansi.ts` 把 SGR
序列转成 `<span style>` —— **不能**用围栏代码块，否则 `<span>` 会被当作字面量转义、颜色丢失。
不含 ANSI 时改用围栏 `text` 代码块，从而复用 expressive-code 的样式与复制按钮。
需样式的输出统一下述包裹层：HTML 输出包 `<div class="nb-output-html">`，
`stderr` 流包 `<div class="nb-stderr">`。

二者均落在正文 `.custom-md` 作用域内，因此可直接复用现有基础设施 —— 表格的横向滚动由
`src/layouts/Layout.astro:841` 的自动包裹处理，图片由 Fancybox 的 `.custom-md img`
选择器接管，无需重复实现。

## 8. 资源抽取

- 落地路径：`src/assets/notebooks/<slug>/<sha256 前 16 位>.<ext>`
- 命名基于**内容 hash**：同一图片重复出现只落一次；重跑得到相同文件名，天然幂等
- 已存在同名文件则跳过写入（避免触发 Vite 重载）
- `<slug>` 为 notebook 相对 posts 目录的路径去扩展名（子目录 notebook 保留层级）
- 目录整体写进 `.gitignore`：`/src/assets/notebooks/`

## 9. 幂等、监听与清理

**生成器**（`generateNotebooks`）：

1. glob `src/content/posts/**/*.ipynb`
2. 对每个 notebook 转换；结果与已存在的目标文件逐字节比较，**仅在内容变化时写入**
   （避免 Vite 重载循环）
3. 目标路径与手写文件冲突（目标已存在且不在产物清单中）→ 报错并跳过该文件
4. 清理：产物清单中存在、但本轮不再生成的 `.md` → 删除
5. 重写 `src/content/posts/.gitignore`（含头部注释「自动生成，请勿手改」+ 产物相对路径清单）；
   清单为空时删除该文件

**dev 热更新**：loader 通过 `ctx.watcher` 监听 `src/content/posts` 下的 `.ipynb`：

- `change` / `add` / `unlink` → 重新生成 → glob 自身的 watcher 接住生成的 `.md`
  （`glob.js:264-270` 同时监听 `change`/`add`/`unlink`，因此新增 notebook 无需重启 dev）
- 监听器只注册一次（`load` 每次同步都会执行），用一个模块级 WeakSet 按 `ctx` 去重
- 防抖：连续事件合并为一次生成

**写文件引发的重载**：`astro.config.mjs` 的 `vite.server.watch.ignored` 增加
`**/src/content/posts/.gitignore`（该文件每次都被重写；内容不变时不写可进一步降噪）。

## 10. 错误处理

原则：单个 notebook 出错只跳过该文件并报明原因，**不中断构建**。

| 情况 | 行为 |
|---|---|
| JSON 解析失败 | `[notebooks] foo.ipynb: 解析失败 —— <原始错误>` |
| 缺 `nbformat` / `cells` 非数组 | 报错并跳过 |
| 首个单元格是 YAML 但解析失败 | 报错并跳过，提示格式 |
| `published` 非法日期 | 报错并跳过 |
| 与手写 `.md` 撞名 | 报错并跳过，提示二选一 |
| 文件名以下划线开头 | 报错并跳过 —— content 目录下 `_` 前缀是特殊语义（`hasUnderscoreBelowContentDirectoryPath`），生成物一律不得落入该命名空间 |
| 未知 MIME | 回退 `text/plain`；都没有则输出一句占位说明 |
| 未知单元格类型 | 跳过 |

schema 校验（`parseData`）失败由 Astro 抛出，不做额外包装。

## 11. 集成点（改动清单）

新增：

- `src/utils/notebook/{parseNotebook,resolveMetadata,cellToMarkdown,outputToMarkdown,ansi,extractAssets,convertNotebook}.ts`
- `src/loaders/notebook-posts.ts`
- `scripts/generate-notebooks.ts`
- `src/content/posts/.gitignore`（由生成器维护）

修改：

- `src/content.config.ts` — posts 的 loader 从 `glob({...})` 换成 `notebookPosts({ base: "./src/content/posts" })`
- `astro.config.mjs` — `vite.server.watch.ignored` 增加 `**/src/content/posts/.gitignore`
  （该文件由生成器重写；生成的 `.md` 不忽略，它们是热更新的数据来源）
- `package.json` — 新增 `"notebooks": "npx tsx scripts/generate-notebooks.ts"`；
  `build` 在 `generate-lqips.ts` 与 `astro build` 之间插入该步骤，最终为：
  `node scripts/generate-icons.js && npx tsx scripts/generate-notebooks.ts && npx tsx scripts/generate-lqips.ts && astro build && npx tsx scripts/subset-fonts.ts && pagefind --site dist`
- `.gitignore` — 增加 `/src/assets/notebooks/`
- 样式：`src/styles/markdown.css`（由 `src/layouts/MainGridLayout.astro:42` 引入，正文作用域为 `.custom-md`）
  内补充 `.nb-output-html` 与 `.nb-stderr` 的最小样式
- `docs/writing-guide.md` — 增加 notebook 文章写法一节

## 12. 验证计划

样例 notebook（临时，验证后删除；文件名不得以下划线开头）：

1. `verify-nb-a.ipynb` — markdown 单元格含 KaTeX 公式与 mermaid 图；代码单元格用于验证代码块形态
   （行号、语言徽章、复制按钮、双主题配色）
2. `verify-nb-b.ipynb` — 代码单元格含 base64 PNG 输出、stdout、HTML 表格（pandas 风格）、`error` 输出
3. `verify-nb-c.ipynb` — ANSI 彩色 stdout 与彩色 traceback

验证项：

- `pnpm dev` 下访问 `/posts/verify-nb-a/` 等三条路由均 200，正文含公式渲染结果、
  mermaid 容器、expressive-code 的 `frame`/行号/复制按钮 DOM
- 图片被抽到 `src/assets/notebooks/` 且页面 `<img>` 指向优化后的资源（非 base64 data URI）
- 生成的 `.md` 出现在 `src/content/posts/`，且被 `.gitignore` 忽略（`git status` 干净）
- 热更新：改动 notebook 后页面自动更新；新增 notebook 无需重启 dev
- `pnpm build` 通过；`src/constants/lqips.json` 含 notebook 图片条目
- 错误路径：构造一个坏 notebook，确认报错清晰且其余文章照常构建

## 13. 风险与已知限制

- **与 Astro 版本的耦合**：本设计依赖 `glob()` 的 stale 清理与 watcher 行为（已在 7.0.2 源码中核实）。
  升级 Astro major 时需回归验证 §12。
- **生成物出现在 `src/`**：依赖幂等写入 + watcher ignore 避免重载循环；若未来 Vite 变更
  watch 语义需重新评估。
- **notebook 的 `text/html` 输出原样注入**：属于作者自有内容，与 markdown 中写原始 HTML 同等信任级；
  不做 sanitize。若将来支持外部投稿需重新评估。
- **大输出**：包含大体积 base64（如 10MB 图像）的 notebook 会让首次转换变慢；
  输出未做尺寸裁剪。
- **`_` 前缀路径在 content 下是特殊语义**（`hasUnderscoreBelowContentDirectoryPath`），
  因此生成物一律不使用 `_` 前缀目录。
