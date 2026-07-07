# YPC 的博客

一个基于 **Astro** 构建的中文个人博客项目，支持：

- 中文首页与关于页
- 博客文章列表
- 标签分类
- 系列文章
- 时间归档
- RSS
- sitemap
- 深色模式
- Pagefind 站内搜索
- 文章目录（TOC）
- 置顶文章

---

## 1. 项目简介

这个项目的目标是提供一个适合长期写作和知识沉淀的个人博客骨架。

目前它已经具备比较完整的内容组织能力，既适合写普通博客，也适合做个人知识库。

---

## 2. 主要功能

### 内容组织
- 文章列表页：`/blog`
- 标签页：`/tags`
- 系列页：`/series`
- 归档页：`/archives`
- 搜索页：`/search`

### 阅读体验
- 深色模式切换
- 文章目录导航
- 阅读进度条
- 返回顶部

### 分发与发现
- RSS：`/rss.xml`
- sitemap：构建时自动生成
- SEO 基础元信息
- Pagefind 站内搜索索引

---

## 3. 本地开发

在项目根目录运行：

```bash
npm install
npm run dev
```

启动后访问：

```text
http://localhost:4321/
```

---

## 4. 生产构建

```bash
npm run build
```

这个命令会执行两件事：

1. Astro 构建静态站点到 `dist/`
2. Pagefind 为 `dist/` 生成搜索索引

构建完成后，可预览：

```bash
npm run preview
```

---

## 5. 项目结构

```text
D:\blog
├── README.md
├── docs/
│   ├── content-model.md
│   ├── writing-guide.md
│   └── deployment.md
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   ├── content/
│   │   └── blog/
│   ├── layouts/
│   └── pages/
├── astro.config.mjs
├── package.json
└── tsconfig.json
```

---

## 6. 内容写在哪里

公开文章写在：

```text
src/content/blog/
```

项目说明文档写在：

```text
README.md
```

更详细的内部维护文档写在：

```text
docs/
```

---

## 7. 如何新增文章

在 `src/content/blog/` 下新建一个 Markdown 或 MDX 文件，例如：

```text
src/content/blog/my-new-post.md
```

然后写 frontmatter 和正文。

详细字段说明见：

- `docs/content-model.md`
- `docs/writing-guide.md`

---

## 8. 当前支持的文章增强字段

常用 frontmatter 字段包括：

- `title`
- `description`
- `pubDate`
- `updatedDate`
- `heroImage`
- `tags`
- `featured`
- `series`
- `seriesOrder`
- `draft`
- `language`

具体含义见：

```text
docs/content-model.md
```

---

## 9. 维护建议

如果你后续准备长期使用这个博客，建议优先继续完善：

- 替换首页与 About 的占位内容
- 换成真实社交链接
- 加评论系统（例如 Giscus）
- 部署到 Vercel 或 Cloudflare Pages
- 补一批真实文章

---

## 10. 相关文档

- 内容模型：`docs/content-model.md`
- 写作指南：`docs/writing-guide.md`
- 部署说明：`docs/deployment.md`
