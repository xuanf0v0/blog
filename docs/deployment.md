# 部署说明

这份文档说明这个博客项目如何本地构建，以及后续适合部署到哪些平台。

---

## 1. 本地开发

安装依赖：

```bash
npm install
```

启动开发环境：

```bash
npm run dev
```

默认地址：

```text
http://localhost:4321/
```

---

## 2. 生产构建

执行：

```bash
npm run build
```

当前 build 脚本会：

1. 执行 Astro 静态构建
2. 输出到 `dist/`
3. 运行 Pagefind
4. 在 `dist/pagefind/` 生成搜索索引

---

## 3. 本地预览构建结果

```bash
npm run preview
```

适合在正式部署前确认：
- 页面路径是否正常
- 搜索资源是否正常
- RSS / sitemap 是否生成
- 样式是否一致

---

## 4. 推荐部署平台

### Vercel
优点：
- 配置简单
- 对 Astro 支持好
- 适合个人博客

一般构建命令：

```bash
npm run build
```

输出目录：

```text
dist
```

### Cloudflare Pages
优点：
- 静态站部署体验很好
- CDN 快
- 免费额度友好

构建命令同样是：

```bash
npm run build
```

输出目录：

```text
dist
```

---

## 5. 部署前检查清单

正式部署前建议检查：

- `SITE_URL` 是否改成真实域名
- 首页/About 是否替换占位内容
- 社交链接是否替换
- RSS 是否可访问
- sitemap 是否生成
- 搜索页是否能加载 `pagefind-ui.js`

---

## 6. 当前与部署相关的重要文件

### `package.json`
决定：
- 构建命令
- 开发命令
- Pagefind 是否执行

### `astro.config.mjs`
决定：
- 站点 URL
- sitemap 集成
- markdown/rehype 配置

### `src/consts.ts`
决定：
- 站点标题
- 站点描述
- 作者名
- 关键词
- 站点 URL

---

## 7. 部署后如何验证

部署完成后建议检查这些页面：

- `/`
- `/blog`
- `/search`
- `/tags`
- `/series`
- `/archives`
- `/rss.xml`
- `/sitemap-index.xml`

并确认：
- 页面返回正常
- 搜索可以加载
- RSS 内容正常
- 系列和标签页没有空页面

---

## 8. 后续可加的部署增强

后续可以继续接入：

- 自定义域名
- 评论系统（如 Giscus）
- 统计分析
- 图片优化策略
- 自动化 CI/CD 部署
