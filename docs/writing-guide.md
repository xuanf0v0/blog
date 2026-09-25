# 写作指南

这份文档说明如何在这个博客里新增、组织和维护文章。

---

## 1. 新文章放在哪里

所有公开文章都放在：

```text
src/content/blog/
```

支持两种格式：

- Markdown：`.md`
- MDX：`.mdx`

如果只是普通文章，优先用 `.md`。
如果需要嵌入 Astro 组件或更灵活的内容，使用 `.mdx`。

---

## 2. 文件命名建议

建议用英文 slug 命名文件，例如：

```text
my-first-post.md
blog-structure-notes.md
writing-workflow.mdx
```

好处：
- URL 更稳定
- 兼容性更好
- 不容易因为路径编码出现问题

---

## 3. 一篇文章的最小模板

```md
---
title: '文章标题'
description: '文章摘要'
pubDate: '2026-07-02'
tags:
  - 标签1
  - 标签2
draft: false
language: 'zh-CN'
---

## 开头

正文开始。
```

---

## 4. 推荐写法

### 普通文章
适合：
- 学习笔记
- 工具记录
- 短文
- 经验总结

```yaml
featured: false
```

### 置顶文章
适合：
- 博客入口文
- 代表作
- 长期希望首页展示的内容

```yaml
featured: true
```

### 系列文章
适合：
- 教程
- 项目复盘
- 连载内容

```yaml
series: '博客搭建记录'
seriesOrder: 1
```

---

## 5. 标签怎么写

建议原则：

- 控制在 2~5 个标签
- 优先使用已有标签
- 避免同义词重复
- 尽量不要一篇文章十几个标签

例如：

```yaml
tags:
  - Astro
  - 博客
  - 写作
```

---

## 6. 草稿怎么处理

如果文章还不准备发布，可以写：

```yaml
draft: true
```

这样它不会出现在：
- 博客列表
- 标签页
- 系列页
- 归档页
- RSS

---

## 7. 什么时候用 MDX

当你需要：
- 在正文里插入 Astro 组件
- 做更复杂的展示
- 放交互按钮或提示块

才建议用 `.mdx`。

普通文章优先 `.md`，更简单稳定。

---

## 8. 用 Jupyter Notebook 写文章

除了 `.md` / `.mdx`，也可以直接把 `.ipynb` 放进：

```text
src/content/posts/
```

文件名就是 URL：`softmax.ipynb` → `/posts/softmax/`。
构建时会先把它转成同名的 `.md`，再按普通文章的流程加载，所以你只需要维护 notebook 本身。

### 元数据怎么写

两种写法任选其一。

写法一，首单元格 YAML：把 notebook 的第一个单元格设为 Markdown，内容从 `---` 开始——

```yaml
---
title: '用 Softmax 做多分类'
published: 2026-07-02
tags:
  - 机器学习
---
```

这和普通 `.md` 文章的 frontmatter 完全一致。首单元格里 frontmatter 之后的正文会**保留**为文章开头，不会丢掉。

写法二，notebook 元数据：在 Jupyter 里点 `Edit → Edit Notebook Metadata`，加入 `blog` 字段——

```json
{
  "blog": {
    "title": "用 Softmax 做多分类",
    "published": "2026-07-02",
    "tags": ["机器学习"]
  }
}
```

两种都写时，**首单元格 YAML 优先**，按字段覆盖 `metadata.blog`。

可用字段与普通文章一致（`title`、`description`、`tags`、`category`、`lang`、`draft`、`pinned`、`comment`、`password` 等）。其中 **`published` 必填**，两种写法里至少要有一处给出，格式为 `YYYY-MM-DD`，不要带时分秒；缺失时会报错并跳过这个 notebook，**不会**拿文件修改时间兜底——这是刻意的，避免换台机器日期就漂移。`title` 可以省略，省略时按文件名推导。

### 几条约束

- 文件名**不能以下划线开头**：`src/content/posts/` 下 `_` 前缀是保留语义，会被报错跳过。
- **不要**手写与 notebook 同名的 `.md`（例如在 `softmax.ipynb` 旁边再放一个 `softmax.md`）：会因撞名报错并跳过该 notebook。
- 生成物不入库：`src/content/posts/<name>.md` 与抽出的图片（`src/assets/notebooks/`）都由 `.gitignore` 自动维护，**不要手改，也不要提交**。要改内容就改 notebook。
- 改完 notebook 保存即可，dev server 会自动重新生成，无需重启。

### 会渲染什么，不会做什么

已保存的输出会按原样渲染：

- stdout / stderr 流，报错回溯也保留 ANSI 颜色
- `text/html` 输出，例如 pandas 表格
- base64 内嵌图片：会抽成真实文件放进 `src/assets/notebooks/`，因此照常参与图片优化与 LQIP，也能点击放大
- Markdown 单元格与代码单元格（代码按 notebook 的语言高亮）

但**不会执行 notebook**：构建时只读取其中已经保存的输出，不会跑代码。所以发布前请在本地把要展示的结果跑出来并保存。

---

## 9. 写作建议

建议每篇文章至少包含：

1. 标题
2. 一句摘要
3. 2~3 个小节标题
4. 清晰的结尾或下一步

这样：
- 搜索结果更好看
- TOC 更有用
- 文章结构更清晰

---

## 10. 发布前检查

新增文章后，建议执行：

```bash
npm run dev
```

本地确认：
- 文章页正常打开
- 标签显示正常
- 系列顺序正常
- TOC 正常生成
- 搜索页能搜到内容（需 build 后）

如果要验证搜索索引，执行：

```bash
npm run build
```

---

## 11. 推荐内容类型

适合这个博客的内容方向：

- 技术笔记
- 项目复盘
- 写作与知识管理
- 工具与效率系统
- 系列教程
- 长期更新型清单
