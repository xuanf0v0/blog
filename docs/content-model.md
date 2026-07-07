# 内容模型说明

本文档说明博客文章支持的 frontmatter 字段。

文章内容目录：

```text
src/content/blog/
```

支持文件类型：

- `.md`
- `.mdx`

---

## 基础示例

```yaml
---
title: '从零搭起我的博客工作台'
description: '这是一篇介绍博客定位与未来规划的文章。'
pubDate: '2026-07-01'
updatedDate: '2026-07-02'
heroImage: '../../assets/blog-placeholder-1.jpg'
tags:
  - 博客
  - Astro
  - 开始
featured: true
series: '博客搭建记录'
seriesOrder: 1
draft: false
language: 'zh-CN'
---
```

---

## 字段说明

### `title`
文章标题。

```yaml
title: '文章标题'
```

### `description`
文章摘要，会用于：
- 博客列表页
- SEO description
- RSS description
- 搜索结果摘要

```yaml
description: '一句简短摘要'
```

### `pubDate`
发布日期。

```yaml
pubDate: '2026-07-01'
```

### `updatedDate`
可选，文章更新时间。

```yaml
updatedDate: '2026-07-02'
```

### `heroImage`
可选，文章头图。

```yaml
heroImage: '../../assets/blog-placeholder-1.jpg'
```

### `tags`
标签数组，用于标签页聚合。

```yaml
tags:
  - 博客
  - Astro
  - 写作
```

### `featured`
是否置顶。

- `true`：在首页/博客列表优先展示
- `false`：普通文章

```yaml
featured: true
```

### `series`
可选，所属系列名称。

```yaml
series: '博客搭建记录'
```

### `seriesOrder`
可选，该文章在系列中的顺序。

```yaml
seriesOrder: 2
```

### `draft`
是否草稿。

- `true`：不会出现在正式页面和 RSS 中
- `false`：正常发布

```yaml
draft: false
```

### `language`
文章语言标记。

```yaml
language: 'zh-CN'
```

### `canonicalURL`
可选，自定义 canonical URL。

```yaml
canonicalURL: 'https://example.com/blog/my-post/'
```

---

## 使用建议

### 什么时候加 `featured`
适合：
- 首页想长期展示的文章
- 博客入口文章
- 系列总览或代表作

### 什么时候加 `series`
适合：
- 教程
- 项目复盘
- 连载主题
- 阶段性知识整理

### `tags` 怎么控制
建议：
- 每篇文章 2~5 个标签即可
- 不要过度细分
- 尽量保持标签命名稳定

---

## 当前页面如何消费这些字段

- `/blog`：使用 `featured`、`tags`、`series`
- `/tags`：使用 `tags`
- `/series`：使用 `series`、`seriesOrder`
- `/archives`：使用 `pubDate`
- `/rss.xml`：使用 `title`、`description`、`pubDate`、`tags`
