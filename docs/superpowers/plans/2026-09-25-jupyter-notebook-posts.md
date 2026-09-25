# Jupyter Notebook 文章支持 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `src/content/posts/**/*.ipynb` 与手写 markdown 文章完全等价地渲染为博客文章。

**Architecture:** 生成器把 `.ipynb` 转成同目录同名的 `.md`（Git 忽略、内容 hash 幂等），再交给 Astro 官方 `glob()` loader —— 从而自动继承项目既有的 unified 管线（KaTeX / expressive-code / mermaid / callouts）与全部现有样式，且不触碰 Astro 内部实现。

**Tech Stack:** Astro 7.0.2 content layer、`astro/loaders` 的 `glob()`、TypeScript、tsx、Node 内置模块（不使用任何新增运行时依赖）。

**Spec:** `docs/superpowers/specs/2026-09-25-jupyter-notebook-posts-design.md`

## Global Constraints

- 包管理器 **pnpm@9.14.4**（本机 `pnpm` 是 11.7.0，版本不符，一律用 `npx pnpm@9.14.4` 或 `npx tsx` 调用）。
- 代码风格由 Biome 强制：**tab 缩进、双引号**、`useDefaultParameterLast`、`noParameterAssign`、`noUselessElse`、`useSingleVarDeclarator`、`useConst`、organizeImports。CI 执行 `biome ci ./src`。
- tsconfig 关键项：`verbatimModuleSyntax: true`（类型导入**必须**写 `import type`）、`isolatedModules: true`、`strictNullChecks: true`、`allowJs: false`。`pnpm type-check` 额外带 `--isolatedDeclarations`，**所有导出函数必须写显式返回类型**。
- `strict` 与 `noUncheckedIndexedAccess` **未开启**，数组下标访问不会带 `undefined`。
- **不引入新的运行时/开发依赖**。`js-yaml` 与 `yaml` 在 pnpm 严格依赖下不可解析，frontmatter 必须手写序列化。
- **不引入测试框架**。仓库无测试基础设施，且 CI（`.github/workflows/build.yml`）只跑 `astro check` 与 `astro build`，不跑测试。验证手段：`scripts/check-notebooks.ts` 断言脚本 + dev server / build 实跑证据。
- 新增的 `src/utils/notebook/**` 模块内部**只用相对导入**（不用 `@/` 别名），因为 `scripts/*.ts` 由 tsx 直接执行、不经 Vite 别名解析。
- 生成物路径规则：`src/content/posts/foo.ipynb` → `src/content/posts/foo.md`；图片 → `src/assets/notebooks/<slug>/<hash>.<ext>`。
- 错误处理总原则：单个 notebook 出错只跳过该文件并打印原因，**不中断构建**。
- 生成物文件名**不得以下划线开头**（content 目录下 `_` 前缀是特殊语义）。

---

## File Structure

**新增**

| 文件 | 职责 |
|---|---|
| `src/utils/notebook/types.ts` | .ipynb 结构类型 |
| `src/utils/notebook/text.ts` | `source`/`text` 字段归一化（字符串或行数组 → 字符串） |
| `src/utils/notebook/parseNotebook.ts` | 读文件 + JSON 解析 + 结构校验 |
| `src/utils/notebook/resolveMetadata.ts` | 合并 `metadata.blog` 与首单元格 YAML、校验字段类型、产出 frontmatter |
| `src/utils/notebook/ansi.ts` | `hasAnsi()` / SGR → `<span style>` |
| `src/utils/notebook/extractAssets.ts` | base64 → 内容 hash 命名的文件（幂等） |
| `src/utils/notebook/outputToMarkdown.ts` | 单个输出 → markdown/HTML 片段 |
| `src/utils/notebook/cellToMarkdown.ts` | 单元格 → markdown |
| `src/utils/notebook/convertNotebook.ts` | 编排：ipynb → `{ frontmatter, markdown }` |
| `src/utils/notebook/generate.ts` | 批量生成、清理孤儿、维护 `.gitignore`、幂等写入 |
| `src/loaders/notebook-posts.ts` | 包装 `glob()`，接入内容集合 + dev 监听 |
| `scripts/generate-notebooks.ts` | 命令行入口（构建步骤） |
| `scripts/check-notebooks.ts` | 断言脚本（永久回归检查，`pnpm check:notebooks`） |
| `src/content/posts/.gitignore` | 由生成器维护，不在本计划中手写 |

> `text.ts` 是对 spec §5 模块表的补充：`toText()` 被 `parseNotebook`、`resolveMetadata`、
> `outputToMarkdown`、`cellToMarkdown` 共同需要，放在 `types.ts` 会让类型文件承担逻辑。

**修改**

| 文件 | 改动 |
|---|---|
| `src/content.config.ts` | posts 的 loader 换成 `notebookPosts()` |
| `astro.config.mjs` | `vite.server.watch.ignored` 增加 `.gitignore` |
| `package.json` | 新增 `notebooks` / `check:notebooks` 脚本；`build` 插入生成步骤 |
| `.gitignore` | 增加 `/src/assets/notebooks/` |
| `src/styles/markdown.css` | 补 `.nb-output` / `.nb-output-html` / `.nb-stderr` 样式 |
| `docs/writing-guide.md` | 补 notebook 写法章节 |

---

### Task 1: ANSI 转 HTML

**Files:**
- Create: `src/utils/notebook/ansi.ts`
- Create: `scripts/check-notebooks.ts`
- Modify: `package.json`（新增 `check:notebooks`）

**Interfaces:**
- Consumes: 无
- Produces:
  - `hasAnsi(text: string): boolean`
  - `ansiToHtml(text: string): string`

- [ ] **Step 1: 先写断言（此刻必然失败，因为模块不存在）**

创建 `scripts/check-notebooks.ts`：

```ts
import assert from "node:assert/strict";
import { ansiToHtml, hasAnsi } from "../src/utils/notebook/ansi";

let failures = 0;

function check(name: string, run: () => void): void {
	try {
		run();
		console.log(`  ok  ${name}`);
	} catch (error) {
		failures += 1;
		console.error(`FAIL  ${name}\n      ${(error as Error).message}`);
	}
}

function section(title: string): void {
	console.log(`\n${title}`);
}

section("ansi");
check("普通文本不带 ANSI", () => {
	assert.equal(hasAnsi("hello"), false);
	assert.equal(ansiToHtml("hello"), "hello");
});
check("识别 ANSI 序列", () => {
	assert.equal(hasAnsi("\u001b[31mred\u001b[0m"), true);
});
check("前景色转 span", () => {
	assert.equal(
		ansiToHtml("\u001b[31mred\u001b[0m"),
		'<span style="color:#cd0000">red</span>',
	);
});
check("重置后回到无样式", () => {
	assert.equal(ansiToHtml("\u001b[1mb\u001b[0mplain"), '<span style="font-weight:600">b</span>plain');
});
check("HTML 特殊字符被转义", () => {
	assert.equal(ansiToHtml("<a> &"), "&lt;a&gt; &amp;");
});
check("256 色与 truecolor", () => {
	assert.equal(ansiToHtml("\u001b[38;5;196mX\u001b[0m"), '<span style="color:#ff0000">X</span>');
	assert.equal(ansiToHtml("\u001b[38;2;18;52;86mX\u001b[0m"), '<span style="color:#123456">X</span>');
});
check("未知序列被丢弃且不影响文本", () => {
	assert.equal(ansiToHtml("\u001b[99mX"), "X");
});

console.log(failures === 0 ? "\n全部通过" : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
```

`package.json` 的 `scripts` 增加：

```json
"check:notebooks": "npx tsx scripts/check-notebooks.ts",
```

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 报错 `Cannot find module '../src/utils/notebook/ansi'`

- [ ] **Step 3: 实现 `src/utils/notebook/ansi.ts`**

```ts
/** xterm 标准 16 色 */
const BASIC_COLORS: string[] = [
	"#000000",
	"#cd0000",
	"#00cd00",
	"#cdcd00",
	"#0000ee",
	"#cd00cd",
	"#00cdcd",
	"#e5e5e5",
	"#7f7f7f",
	"#ff0000",
	"#00ff00",
	"#ffff00",
	"#5c5cff",
	"#ff00ff",
	"#00ffff",
	"#ffffff",
];

const SGR_PATTERN = /\u001b\[([0-9;]*)m/g;

interface SgrState {
	fg: string | null;
	bg: string | null;
	bold: boolean;
	dim: boolean;
	italic: boolean;
	underline: boolean;
}

export function hasAnsi(text: string): boolean {
	return /\u001b\[[0-9;]*m/.test(text);
}

function emptyState(): SgrState {
	return { fg: null, bg: null, bold: false, dim: false, italic: false, underline: false };
}

function escapeHtml(text: string): string {
	return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function toHex(red: number, green: number, blue: number): string {
	const part = (value: number): string => value.toString(16).padStart(2, "0");
	return `#${part(red)}${part(green)}${part(blue)}`;
}

function xterm256(index: number): string {
	if (index < 16) return BASIC_COLORS[index] ?? "#000000";
	if (index >= 232) {
		const grey = 8 + (index - 232) * 10;
		return toHex(grey, grey, grey);
	}
	const offset = index - 16;
	const level = (value: number): number => (value === 0 ? 0 : 55 + value * 40);
	return toHex(
		level(Math.floor(offset / 36)),
		level(Math.floor((offset % 36) / 6)),
		level(offset % 6),
	);
}

function applySgr(state: SgrState, params: string): SgrState {
	const codes = params.split(";").map((part) => (part === "" ? 0 : Number.parseInt(part, 10)));
	const next: SgrState = { ...state };
	for (let index = 0; index < codes.length; index += 1) {
		const code = codes[index];
		if (code === 0) {
			Object.assign(next, emptyState());
		} else if (code === 1) {
			next.bold = true;
		} else if (code === 2) {
			next.dim = true;
		} else if (code === 3) {
			next.italic = true;
		} else if (code === 4) {
			next.underline = true;
		} else if (code === 22) {
			next.bold = false;
			next.dim = false;
		} else if (code === 23) {
			next.italic = false;
		} else if (code === 24) {
			next.underline = false;
		} else if (code >= 30 && code <= 37) {
			next.fg = BASIC_COLORS[code - 30];
		} else if (code === 39) {
			next.fg = null;
		} else if (code >= 40 && code <= 47) {
			next.bg = BASIC_COLORS[code - 40];
		} else if (code === 49) {
			next.bg = null;
		} else if (code >= 90 && code <= 97) {
			next.fg = BASIC_COLORS[code - 90 + 8];
		} else if (code >= 100 && code <= 107) {
			next.bg = BASIC_COLORS[code - 100 + 8];
		} else if (code === 38 && codes[index + 1] === 5) {
			next.fg = xterm256(codes[index + 2]);
			index += 2;
		} else if (code === 48 && codes[index + 1] === 5) {
			next.bg = xterm256(codes[index + 2]);
			index += 2;
		} else if (code === 38 && codes[index + 1] === 2) {
			next.fg = toHex(codes[index + 2], codes[index + 3], codes[index + 4]);
			index += 4;
		} else if (code === 48 && codes[index + 1] === 2) {
			next.bg = toHex(codes[index + 2], codes[index + 3], codes[index + 4]);
			index += 4;
		}
	}
	return next;
}

function styleToCss(state: SgrState): string {
	const parts: string[] = [];
	if (state.fg) parts.push(`color:${state.fg}`);
	if (state.bg) parts.push(`background-color:${state.bg}`);
	if (state.bold) parts.push("font-weight:600");
	if (state.dim) parts.push("opacity:0.7");
	if (state.italic) parts.push("font-style:italic");
	if (state.underline) parts.push("text-decoration:underline");
	return parts.join(";");
}

function pushChunk(out: string[], chunk: string, state: SgrState): void {
	if (chunk === "") return;
	const css = styleToCss(state);
	out.push(css === "" ? escapeHtml(chunk) : `<span style="${css}">${escapeHtml(chunk)}</span>`);
}

export function ansiToHtml(text: string): string {
	const out: string[] = [];
	let state = emptyState();
	let cursor = 0;
	SGR_PATTERN.lastIndex = 0;
	let match = SGR_PATTERN.exec(text);
	while (match !== null) {
		pushChunk(out, text.slice(cursor, match.index), state);
		state = applySgr(state, match[1]);
		cursor = match.index + match[0].length;
		match = SGR_PATTERN.exec(text);
	}
	pushChunk(out, text.slice(cursor), state);
	return out.join("");
}
```

- [ ] **Step 4: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 7 个 `ok`，末尾 `全部通过`，退出码 0

- [ ] **Step 5: lint 与类型检查**

Run: `npx pnpm@9.14.4 exec biome check ./src/utils/notebook ./scripts/check-notebooks.ts`
Expected: 无错误（若有格式问题，加 `--write` 后再跑一次）

- [ ] **Step 6: 提交**

```bash
git add src/utils/notebook/ansi.ts scripts/check-notebooks.ts package.json
git commit -m "feat(notebook): convert ANSI SGR sequences to styled HTML spans"
```

---

### Task 2: 类型、文本归一化与 notebook 解析

**Files:**
- Create: `src/utils/notebook/types.ts`
- Create: `src/utils/notebook/text.ts`
- Create: `src/utils/notebook/parseNotebook.ts`
- Modify: `scripts/check-notebooks.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  - `type NotebookSource = string | string[]`
  - `interface NotebookOutput { output_type: string; name?: string; text?: NotebookSource; data?: Record<string, unknown>; ename?: string; evalue?: string; traceback?: string[] }`
  - `interface NotebookCell { cell_type: string; source?: NotebookSource; outputs?: NotebookOutput[]; attachments?: Record<string, Record<string, string>> }`
  - `interface NotebookDocument { nbformat: number; metadata: Record<string, unknown>; cells: NotebookCell[] }`
  - `class NotebookError extends Error`
  - `toText(source: NotebookSource | null | undefined): string`
  - `parseNotebook(filePath: string): Promise<NotebookDocument>`（失败抛 `NotebookError`）

- [ ] **Step 1: 加断言**

在 `scripts/check-notebooks.ts` 的 `section("ansi")` 之前插入 import：

```ts
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { NotebookError, parseNotebook } from "../src/utils/notebook/parseNotebook";
import { toText } from "../src/utils/notebook/text";
```

在 `console.log(failures === 0 ...)` 之前插入（注意：这段是异步的，需要用下面的 `main()` 包装 —— 见 Step 1b）：

```ts
section("text");
check("行数组拼接为字符串", () => {
	assert.equal(toText(["a\n", "b"]), "a\nb");
	assert.equal(toText("plain"), "plain");
	assert.equal(toText(undefined), "");
});

section("parseNotebook");
await checkAsync("合法 notebook 解析成功", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "ok.ipynb");
	await fs.writeFile(
		file,
		JSON.stringify({ nbformat: 4, metadata: {}, cells: [{ cell_type: "markdown", source: ["hi"] }] }),
	);
	const doc = await parseNotebook(file);
	assert.equal(doc.cells.length, 1);
});
await checkAsync("JSON 损坏报 NotebookError", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "broken.ipynb");
	await fs.writeFile(file, "{ not json");
	await assert.rejects(() => parseNotebook(file), NotebookError);
});
await checkAsync("缺 nbformat 报 NotebookError", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "old.ipynb");
	await fs.writeFile(file, JSON.stringify({ cells: [] }));
	await assert.rejects(() => parseNotebook(file), NotebookError);
});
await checkAsync("cells 非数组报 NotebookError", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "bad.ipynb");
	await fs.writeFile(file, JSON.stringify({ nbformat: 4, cells: {} }));
	await assert.rejects(() => parseNotebook(file), NotebookError);
});
```

- [ ] **Step 1b: 把脚本改成支持异步断言的形态**

把 `scripts/check-notebooks.ts` 顶部的 `check` 改为同时提供异步版本，并把全部断言包进 `main()`（顶层 await 在本仓库 tsconfig 下可用，但包一层便于汇总退出码）：

```ts
async function checkAsync(name: string, run: () => Promise<void>): Promise<void> {
	try {
		await run();
		console.log(`  ok  ${name}`);
	} catch (error) {
		failures += 1;
		console.error(`FAIL  ${name}\n      ${(error as Error).message}`);
	}
}
```

脚本末尾改成：

```ts
console.log(failures === 0 ? "\n全部通过" : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
```

（`process.exit` 必须在全部 `await checkAsync(...)` 之后，因此所有异步断言要写在文件末尾、`process.exit` 之前。）

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 模块找不到而报错

- [ ] **Step 3: 实现三个模块**

`src/utils/notebook/types.ts`：

```ts
/** notebook 的 source / text 字段既可能是整段字符串，也可能是行数组 */
export type NotebookSource = string | string[];

export interface NotebookOutput {
	output_type: string;
	name?: string;
	text?: NotebookSource;
	data?: Record<string, unknown>;
	ename?: string;
	evalue?: string;
	traceback?: string[];
}

export interface NotebookCell {
	cell_type: string;
	source?: NotebookSource;
	outputs?: NotebookOutput[];
	attachments?: Record<string, Record<string, string>>;
}

export interface NotebookDocument {
	nbformat: number;
	metadata: Record<string, unknown>;
	cells: NotebookCell[];
}
```

`src/utils/notebook/text.ts`：

```ts
import type { NotebookSource } from "./types";

/** 把 notebook 的 source / text 字段归一化为字符串 */
export function toText(source: NotebookSource | null | undefined): string {
	if (source === null || source === undefined) return "";
	return Array.isArray(source) ? source.join("") : source;
}
```

`src/utils/notebook/parseNotebook.ts`：

```ts
import fs from "node:fs/promises";
import type { NotebookCell, NotebookDocument } from "./types";

export class NotebookError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "NotebookError";
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export async function parseNotebook(filePath: string): Promise<NotebookDocument> {
	let raw: string;
	try {
		raw = await fs.readFile(filePath, "utf-8");
	} catch (error) {
		throw new NotebookError(`读取失败：${(error as Error).message}`);
	}

	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch (error) {
		throw new NotebookError(`JSON 解析失败：${(error as Error).message}`);
	}

	if (!isRecord(parsed)) {
		throw new NotebookError("notebook 根节点必须是对象");
	}
	if (typeof parsed.nbformat !== "number") {
		throw new NotebookError("缺少 nbformat 字段，不是有效的 .ipynb");
	}
	if (!Array.isArray(parsed.cells)) {
		throw new NotebookError("cells 字段缺失或不是数组");
	}
	const cells = parsed.cells.filter((cell): cell is NotebookCell => isRecord(cell));

	return {
		nbformat: parsed.nbformat,
		metadata: isRecord(parsed.metadata) ? parsed.metadata : {},
		cells,
	};
}
```

- [ ] **Step 4: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 12 个 `ok`（7 个 ansi + 1 个 text + 4 个 parseNotebook），末尾 `全部通过`

- [ ] **Step 5: 提交**

```bash
git add src/utils/notebook/types.ts src/utils/notebook/text.ts src/utils/notebook/parseNotebook.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): parse and validate .ipynb documents"
```

---

### Task 3: 元数据解析

**Files:**
- Create: `src/utils/notebook/resolveMetadata.ts`
- Modify: `scripts/check-notebooks.ts`

**Interfaces:**
- Consumes: `NotebookDocument`（Task 2）、`NotebookError`（Task 2）、`toText`（Task 2）
- Produces:
  - `interface MetadataResult { frontmatter: Record<string, unknown>; firstBodyCell: number }`
  - `resolveMetadata(doc: NotebookDocument, fileName: string): MetadataResult`

`fileName` 是 `.ipynb` 的文件名（含扩展名），用于 `title` 兜底。

- [ ] **Step 1: 加断言**

在 `scripts/check-notebooks.ts` 末尾（`process.exit` 之前）插入：

```ts
section("resolveMetadata");

function notebook(metadata: Record<string, unknown>, cells: NotebookCell[]): NotebookDocument {
	return { nbformat: 4, metadata, cells };
}

await checkAsync("首单元格 YAML 覆盖 metadata.blog", async () => {
	const doc = notebook(
		{ blog: { title: "来自 metadata", published: "2026-01-01" } },
		[
			{
				cell_type: "markdown",
				source: ["---\n", 'title: "来自 YAML"\n', "---\n"],
			},
		],
	);
	const result = resolveMetadata(doc, "x.ipynb");
	assert.equal(result.frontmatter.title, "来自 YAML");
	assert.equal(result.frontmatter.published, "2026-01-01");
	assert.equal(result.firstBodyCell, 1);
});

await checkAsync("缺 published 报错", async () => {
	const doc = notebook({ blog: { title: "t" } }, []);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});

await checkAsync("缺 title 时用文件名兜底", async () => {
	const doc = notebook({ blog: { published: "2026-01-01" } }, []);
	const result = resolveMetadata(doc, "my-notebook.ipynb");
	assert.equal(result.frontmatter.title, "my notebook");
});

await checkAsync("tags 类型错误报错而不漏到 schema", async () => {
	const doc = notebook({ blog: { title: "t", published: "2026-01-01", tags: "CUDA" } }, []);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});

await checkAsync("YAML 损坏报错", async () => {
	const doc = notebook({}, [{ cell_type: "markdown", source: ["---\n", "a: [1,\n", "---\n"] }]);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});
```

并在文件顶部补：

```ts
import { resolveMetadata } from "../src/utils/notebook/resolveMetadata";
import type { NotebookCell, NotebookDocument } from "../src/utils/notebook/types";
```

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: `Cannot find module '../src/utils/notebook/resolveMetadata'`

- [ ] **Step 3: 实现 `src/utils/notebook/resolveMetadata.ts`**

```ts
import { parseFrontmatter } from "@astrojs/markdown-remark";
import { NotebookError } from "./parseNotebook";
import { toText } from "./text";
import type { NotebookDocument } from "./types";

export interface MetadataResult {
	frontmatter: Record<string, unknown>;
	/** 正文从第几个单元格开始（首单元格被 YAML 消费时为 1） */
	firstBodyCell: number;
}

const STRING_FIELDS = new Set([
	"title",
	"description",
	"image",
	"category",
	"lang",
	"author",
	"sourceLink",
	"licenseName",
	"licenseUrl",
	"password",
	"passwordHint",
]);

const DATE_FIELDS = new Set(["published", "updated"]);

const BOOLEAN_FIELDS = new Set(["draft", "pinned", "comment"]);

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** YAML 里未加引号的日期会被当成 Date，字符串则原样保留 */
function normalizeDate(field: string, value: unknown): string {
	const text =
		value instanceof Date
			? value.toISOString().slice(0, 10)
			: typeof value === "string"
				? value.trim()
				: "";
	if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) {
		throw new NotebookError(`字段 ${field} 不是合法的 YYYY-MM-DD 日期：${String(value)}`);
	}
	return text;
}

function titleFromFileName(fileName: string): string {
	return fileName
		.replace(/\.ipynb$/i, "")
		.replace(/[-_]+/g, " ")
		.trim();
}

function validate(frontmatter: Record<string, unknown>): Record<string, unknown> {
	const output: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(frontmatter)) {
		if (value === undefined || value === null) continue;
		if (STRING_FIELDS.has(key)) {
			if (typeof value !== "string") {
				throw new NotebookError(`字段 ${key} 必须是字符串，实际是 ${typeof value}`);
			}
			output[key] = value;
		} else if (DATE_FIELDS.has(key)) {
			output[key] = normalizeDate(key, value);
		} else if (BOOLEAN_FIELDS.has(key)) {
			if (typeof value !== "boolean") {
				throw new NotebookError(`字段 ${key} 必须是布尔值，实际是 ${typeof value}`);
			}
			output[key] = value;
		} else if (key === "tags") {
			if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
				throw new NotebookError("字段 tags 必须是字符串数组");
			}
			output[key] = value;
		} else {
			output[key] = value;
		}
	}
	return output;
}

export function resolveMetadata(doc: NotebookDocument, fileName: string): MetadataResult {
	let fromCell: Record<string, unknown> = {};
	let firstBodyCell = 0;

	const first = doc.cells[0];
	if (first && first.cell_type === "markdown") {
		const source = toText(first.source);
		if (source.startsWith("---")) {
			try {
				const { frontmatter } = parseFrontmatter(source);
				if (isRecord(frontmatter) && Object.keys(frontmatter).length > 0) {
					fromCell = frontmatter;
					firstBodyCell = 1;
				}
			} catch (error) {
				throw new NotebookError(`首单元格 YAML 解析失败：${(error as Error).message}`);
			}
		}
	}

	const blog = doc.metadata.blog;
	const fromNotebook = isRecord(blog) ? blog : {};
	const merged = { ...fromNotebook, ...fromCell };

	if (merged.published === undefined) {
		throw new NotebookError(
			"缺少 published 字段：请在首单元格 YAML 或 metadata.blog 中显式声明（YYYY-MM-DD）",
		);
	}

	const validated = validate(merged);
	if (typeof validated.title !== "string" || validated.title.trim() === "") {
		validated.title = titleFromFileName(fileName);
	}

	return { frontmatter: validated, firstBodyCell };
}
```

- [ ] **Step 4: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 全部 `ok`

- [ ] **Step 5: 提交**

```bash
git add src/utils/notebook/resolveMetadata.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): resolve frontmatter from cell YAML and notebook metadata"
```

---

### Task 4: 资源抽取

**Files:**
- Create: `src/utils/notebook/extractAssets.ts`
- Modify: `scripts/check-notebooks.ts`

**Interfaces:**
- Consumes: 无
- Produces:
  - `interface ExtractedAsset { absolutePath: string; fileName: string }`
  - `writeAsset(options: { base64: string; extension: string; directory: string }): Promise<ExtractedAsset>`

- [ ] **Step 1: 加断言**

```ts
section("extractAssets");
await checkAsync("base64 落地为内容 hash 命名文件", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const payload = Buffer.from("hello-image").toString("base64");
	const first = await writeAsset({ base64: payload, extension: "png", directory: dir });
	assert.equal(first.fileName.endsWith(".png"), true);
	assert.equal((await fs.readFile(first.absolutePath, "utf-8")), "hello-image");
});
await checkAsync("同一内容重复写入得到同一文件名", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const payload = Buffer.from("same").toString("base64");
	const a = await writeAsset({ base64: payload, extension: "png", directory: dir });
	const b = await writeAsset({ base64: payload, extension: "png", directory: dir });
	assert.equal(a.fileName, b.fileName);
	assert.deepEqual(await fs.readdir(dir), [a.fileName]);
});
await checkAsync("不同内容得到不同文件名", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const a = await writeAsset({ base64: Buffer.from("a").toString("base64"), extension: "png", directory: dir });
	const b = await writeAsset({ base64: Buffer.from("b").toString("base64"), extension: "png", directory: dir });
	assert.notEqual(a.fileName, b.fileName);
});
```

顶部补 `import { writeAsset } from "../src/utils/notebook/extractAssets";`

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 模块找不到

- [ ] **Step 3: 实现 `src/utils/notebook/extractAssets.ts`**

```ts
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

export interface ExtractedAsset {
	absolutePath: string;
	fileName: string;
}

async function exists(filePath: string): Promise<boolean> {
	try {
		await fs.access(filePath);
		return true;
	} catch {
		return false;
	}
}

/**
 * 把 base64 图片落地到 directory，文件名取内容 sha256 前 16 位。
 * 内容相同 → 文件名相同 → 天然幂等，重复运行不会产生新文件也不会重写。
 */
export async function writeAsset(options: {
	base64: string;
	extension: string;
	directory: string;
}): Promise<ExtractedAsset> {
	const buffer = Buffer.from(options.base64.replace(/\s+/g, ""), "base64");
	const hash = crypto.createHash("sha256").update(buffer).digest("hex").slice(0, 16);
	const fileName = `${hash}.${options.extension}`;
	const absolutePath = path.join(options.directory, fileName);

	await fs.mkdir(options.directory, { recursive: true });
	if (!(await exists(absolutePath))) {
		await fs.writeFile(absolutePath, buffer);
	}

	return { absolutePath, fileName };
}
```

- [ ] **Step 4: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 全部 `ok`

- [ ] **Step 5: 提交**

```bash
git add src/utils/notebook/extractAssets.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): extract base64 outputs into content-addressed assets"
```

---

### Task 5: 输出与单元格转换

**Files:**
- Create: `src/utils/notebook/outputToMarkdown.ts`
- Create: `src/utils/notebook/cellToMarkdown.ts`
- Modify: `scripts/check-notebooks.ts`

**Interfaces:**
- Consumes: `hasAnsi`/`ansiToHtml`（Task 1）、`NotebookOutput`/`NotebookCell`（Task 2）、`toText`（Task 2）、`NotebookError`（Task 2）
- Produces:
  - `interface OutputContext { resolveAsset(base64: string, extension: string): Promise<string>; resolveAttachment(name: string, data: Record<string, string>): Promise<string> }`
  - `outputToMarkdown(output: NotebookOutput, context: OutputContext): Promise<string>`
  - `cellToMarkdown(cell: NotebookCell, language: string, context: OutputContext): Promise<string>`
  - `detectLanguage(doc: NotebookDocument): string`

`resolveAsset` 返回**markdown 中可直接使用的相对路径**。

- [ ] **Step 1: 加断言**

```ts
section("output / cell");

const stubContext: OutputContext = {
	resolveAsset: async (base64, extension) => `/assets/${extension}/${base64.length}.${extension}`,
	resolveAttachment: async (name) => `/attach/${name}`,
};

await checkAsync("stream 无 ANSI 时用围栏代码块", async () => {
	const md = await outputToMarkdown(
		{ output_type: "stream", name: "stdout", text: ["hello\n"] },
		stubContext,
	);
	assert.match(md, /^```text\nhello\n```$/);
});
await checkAsync("stream 含 ANSI 时用 pre + span", async () => {
	const md = await outputToMarkdown(
		{ output_type: "stream", name: "stdout", text: ["\u001b[31mred\u001b[0m\n"] },
		stubContext,
	);
	assert.match(md, /^<pre class="nb-output"><span style="color:#cd0000">red<\/span>\n<\/pre>$/);
});
await checkAsync("stderr 额外包 nb-stderr", async () => {
	const md = await outputToMarkdown(
		{ output_type: "stream", name: "stderr", text: ["boom\n"] },
		stubContext,
	);
	assert.match(md, /^<div class="nb-stderr">\n[\s\S]*\n<\/div>$/);
});
await checkAsync("text/html 优先于 text/plain", async () => {
	const md = await outputToMarkdown(
		{
			output_type: "display_data",
			data: { "text/plain": ["<table>"], "text/html": ["<table>x</table>"] },
		},
		stubContext,
	);
	assert.match(md, /^<div class="nb-output-html">\n<table>x<\/table>\n<\/div>$/);
});
await checkAsync("text/latex 转 $$ 块", async () => {
	const md = await outputToMarkdown(
		{ output_type: "execute_result", data: { "text/latex": ["x^2"] } },
		stubContext,
	);
	assert.equal(md, "$$\nx^2\n$$");
});
await checkAsync("application/json 重新格式化", async () => {
	const md = await outputToMarkdown(
		{ output_type: "execute_result", data: { "application/json": ['{"a":1}'] } },
		stubContext,
	);
	assert.equal(md, '```json\n{\n  "a": 1\n}\n```');
});
await checkAsync("error 合并 ename/evalue/traceback", async () => {
	const md = await outputToMarkdown(
		{
			output_type: "error",
			ename: "ValueError",
			evalue: "bad",
			traceback: ["Traceback", "ValueError: bad"],
		},
		stubContext,
	);
	assert.match(md, /ValueError: bad[\s\S]*Traceback/);
});
await checkAsync("未知 MIME 回退占位说明", async () => {
	const md = await outputToMarkdown(
		{ output_type: "display_data", data: { "application/vnd.plotly.v1+json": { a: 1 } } },
		stubContext,
	);
	assert.match(md, /无法渲染的输出/);
});
await checkAsync("markdown 单元格原样输出", async () => {
	const md = await cellToMarkdown(
		{ cell_type: "markdown", source: ["# 标题\n", "\n", "正文"] },
		"python",
		stubContext,
	);
	assert.equal(md, "# 标题\n\n正文");
});
await checkAsync("code 单元格带语言围栏与输出", async () => {
	const md = await cellToMarkdown(
		{
			cell_type: "code",
			source: ["print(1)"],
			outputs: [{ output_type: "stream", name: "stdout", text: ["1\n"] }],
		},
		"python",
		stubContext,
	);
	assert.match(md, /^```python\nprint\(1\)\n```\n\n```text\n1\n```$/);
});
await checkAsync("raw 单元格被跳过", async () => {
	const md = await cellToMarkdown({ cell_type: "raw", source: ["x"] }, "python", stubContext);
	assert.equal(md, "");
});
await checkAsync("attachments 被替换为抽取路径", async () => {
	const md = await cellToMarkdown(
		{
			cell_type: "markdown",
			source: ["![p](attachment:image.png)"],
			attachments: { "image.png": { "image/png": "AAAA" } },
		},
		"python",
		stubContext,
	);
	assert.equal(md, "![p](/attach/image.png)");
});
```

顶部补：

```ts
import { cellToMarkdown, detectLanguage } from "../src/utils/notebook/cellToMarkdown";
import type { OutputContext } from "../src/utils/notebook/outputToMarkdown";
import { outputToMarkdown } from "../src/utils/notebook/outputToMarkdown";
```

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 模块找不到

- [ ] **Step 3: 实现 `src/utils/notebook/outputToMarkdown.ts`**

```ts
import { ansiToHtml, hasAnsi } from "./ansi";
import { toText } from "./text";
import type { NotebookOutput } from "./types";

export interface OutputContext {
	/** 落地 base64 图片，返回 markdown 中可直接使用的路径 */
	resolveAsset(base64: string, extension: string): Promise<string>;
	/** 落地单元格附件，返回 markdown 中可直接使用的路径 */
	resolveAttachment(name: string, data: Record<string, string>): Promise<string>;
}

/** 一个输出含多种 MIME 时的取舍顺序 */
const MIME_PRIORITY = [
	"text/html",
	"image/png",
	"image/jpeg",
	"image/svg+xml",
	"text/latex",
	"application/json",
	"text/plain",
];

function fenced(text: string, language: string): string {
	const body = text.replace(/\n+$/, "");
	const runs = body.match(/`+/g) ?? [];
	const longest = runs.reduce((max, run) => Math.max(max, run.length), 0);
	const fence = "`".repeat(Math.max(3, longest + 1));
	return `${fence}${language}\n${body}\n${fence}`;
}

/** 含 ANSI 时只能用原生 HTML，围栏代码块会把 span 当字面量转义 */
function textBlock(text: string): string {
	const body = text.replace(/\n+$/, "");
	if (hasAnsi(body)) {
		return `<pre class="nb-output">${ansiToHtml(body)}</pre>`;
	}
	return fenced(body, "text");
}

function pickByPriority(data: Record<string, unknown>): { mime: string; value: unknown } | null {
	for (const mime of MIME_PRIORITY) {
		if (data[mime] !== undefined && data[mime] !== null) {
			return { mime, value: data[mime] };
		}
	}
	return null;
}

function decodeSvg(value: string): string {
	if (value.includes("<svg")) return value;
	try {
		return Buffer.from(value, "base64").toString("utf-8");
	} catch {
		return value;
	}
}

function prettyJson(text: string): string {
	try {
		return JSON.stringify(JSON.parse(text), null, 2);
	} catch {
		return text;
	}
}

async function renderData(
	value: unknown,
	mime: string,
	context: OutputContext,
): Promise<string> {
	if (mime === "image/png" || mime === "image/jpeg") {
		const extension = mime === "image/png" ? "png" : "jpg";
		const reference = await context.resolveAsset(toText(value as string | string[]), extension);
		return `![notebook 输出](${reference})`;
	}
	if (mime === "image/svg+xml") {
		return decodeSvg(toText(value as string | string[]));
	}
	if (mime === "text/html") {
		return `<div class="nb-output-html">\n${toText(value as string | string[])}\n</div>`;
	}
	if (mime === "text/latex") {
		return `$$\n${toText(value as string | string[])}\n$$`;
	}
	if (mime === "application/json") {
		return fenced(prettyJson(toText(value as string | string[])), "json");
	}
	return textBlock(toText(value as string | string[]));
}

export async function outputToMarkdown(
	output: NotebookOutput,
	context: OutputContext,
): Promise<string> {
	if (output.output_type === "stream") {
		const body = textBlock(toText(output.text));
		return output.name === "stderr" ? `<div class="nb-stderr">\n${body}\n</div>` : body;
	}

	if (output.output_type === "error") {
		const header = `${output.ename ?? "Error"}: ${output.evalue ?? ""}`;
		const traceback = (output.traceback ?? []).join("\n");
		return textBlock(traceback === "" ? header : `${header}\n${traceback}`);
	}

	if (output.output_type === "execute_result" || output.output_type === "display_data") {
		const data = output.data ?? {};
		const picked = pickByPriority(data);
		// 未知 MIME：输出占位注释而非抛错，避免单个新奇输出炸掉整篇
		if (picked === null) {
			return `<!-- 无法渲染的输出：${Object.keys(data).join(", ")} -->`;
		}
		return renderData(picked.value, picked.mime, context);
	}

	return `<!-- 无法渲染的输出类型：${output.output_type} -->`;
}
```

> 本模块不需要 `NotebookError`：所有异常路径都降级为占位注释，错误隔离由 Task 7 的生成器负责。

- [ ] **Step 4: 实现 `src/utils/notebook/cellToMarkdown.ts`**

```ts
import { NotebookError } from "./parseNotebook";
import { outputToMarkdown, type OutputContext } from "./outputToMarkdown";
import { toText } from "./text";
import type { NotebookCell, NotebookDocument } from "./types";

const LANGUAGE_ALIASES: Record<string, string> = {
	python3: "python",
	"c++": "cpp",
	"c#": "csharp",
	"f#": "fsharp",
	node: "javascript",
	nodejs: "javascript",
	py: "python",
	sh: "bash",
	shell: "bash",
};

export function detectLanguage(doc: NotebookDocument): string {
	const languageInfo = doc.metadata.language_info;
	const fromLanguageInfo =
		typeof languageInfo === "object" && languageInfo !== null
			? (languageInfo as Record<string, unknown>).name
			: undefined;
	const kernelspec = doc.metadata.kernelspec;
	const fromKernelspec =
		typeof kernelspec === "object" && kernelspec !== null
			? (kernelspec as Record<string, unknown>).language
			: undefined;
	const raw = typeof fromLanguageInfo === "string" ? fromLanguageInfo : fromKernelspec;
	if (typeof raw !== "string" || raw.trim() === "") return "python";
	const normalized = raw.trim().toLowerCase();
	return LANGUAGE_ALIASES[normalized] ?? normalized;
}

function fencedCode(code: string, language: string): string {
	const body = code.replace(/\n+$/, "");
	const runs = body.match(/`+/g) ?? [];
	const longest = runs.reduce((max, run) => Math.max(max, run.length), 0);
	const fence = "`".repeat(Math.max(3, longest + 1));
	return `${fence}${language}\n${body}\n${fence}`;
}

async function markdownCell(cell: NotebookCell, context: OutputContext): Promise<string> {
	let body = toText(cell.source).replace(/\n+$/, "");
	const attachments = cell.attachments ?? {};
	for (const [name, data] of Object.entries(attachments)) {
		const reference = await context.resolveAttachment(name, data);
		body = body.split(`attachment:${name}`).join(reference);
	}
	return body;
}

export async function cellToMarkdown(
	cell: NotebookCell,
	language: string,
	context: OutputContext,
): Promise<string> {
	if (cell.cell_type === "markdown") {
		return markdownCell(cell, context);
	}

	if (cell.cell_type !== "code") {
		return "";
	}

	const parts: string[] = [fencedCode(toText(cell.source), language)];
	for (const output of cell.outputs ?? []) {
		const rendered = await outputToMarkdown(output, context);
		if (rendered.trim() !== "") parts.push(rendered);
	}
	return parts.join("\n\n");
}
```

> `NotebookError` 在 `cellToMarkdown.ts` 里未被使用则删掉该 import（Biome 会报未使用导入）。

- [ ] **Step 5: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 全部 `ok`

- [ ] **Step 6: 提交**

```bash
git add src/utils/notebook/outputToMarkdown.ts src/utils/notebook/cellToMarkdown.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): convert cells and outputs to markdown"
```

---

### Task 6: 转换编排

**Files:**
- Create: `src/utils/notebook/convertNotebook.ts`
- Modify: `scripts/check-notebooks.ts`

**Interfaces:**
- Consumes: Task 2/3/4/5 的全部产物
- Produces:
  - `interface ConvertedNotebook { frontmatter: Record<string, unknown>; markdown: string }`
  - `convertNotebook(options: { ipynbPath: string; targetPath: string; assetsDir: string }): Promise<ConvertedNotebook>`

`targetPath` 是即将生成的 `.md` 的绝对路径，用于计算图片的相对引用；`assetsDir` 是该 notebook 专属的图片目录绝对路径。

- [ ] **Step 1: 加断言**

```ts
section("convertNotebook");
await checkAsync("端到端转换含图片与公式的 notebook", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
	const ipynb = path.join(dir, "demo.ipynb");
	const png = Buffer.from("fake-png").toString("base64");
	await fs.writeFile(
		ipynb,
		JSON.stringify({
			nbformat: 4,
			metadata: { language_info: { name: "python3" } },
			cells: [
				{ cell_type: "markdown", source: ["---\n", "title: \"示例\"\n", "published: 2026-01-02\n", "tags: [\"a\"]\n", "---\n"] },
				{ cell_type: "markdown", source: ["公式 $x^2$\n"] },
				{ cell_type: "code", source: ["1+1"], outputs: [{ output_type: "execute_result", data: { "text/plain": ["2"] } }] },
				{ cell_type: "code", source: ["plot()"], outputs: [{ output_type: "display_data", data: { "image/png": [png] } }] },
			],
		}),
	);
	const result = await convertNotebook({
		ipynbPath: ipynb,
		targetPath: path.join(dir, "demo.md"),
		assetsDir: path.join(dir, "assets"),
	});
	assert.equal(result.frontmatter.title, "示例");
	assert.equal(result.frontmatter.published, "2026-01-02");
	assert.match(result.markdown, /^#?/);
	assert.match(result.markdown, /公式 \$x\^2\$/);
	assert.match(result.markdown, /```python\n1\+1\n```\n\n```text\n2\n```/);
	assert.match(result.markdown, /!\[notebook 输出\]\(assets\/[0-9a-f]{16}\.png\)/);
	assert.equal((await fs.readdir(path.join(dir, "assets"))).length, 1);
});
await checkAsync("frontmatter 中日期不加引号", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
	const ipynb = path.join(dir, "d.ipynb");
	await fs.writeFile(
		ipynb,
		JSON.stringify({
			nbformat: 4,
			metadata: {},
			cells: [{ cell_type: "markdown", source: ["---\n", 'title: "t"\n', "published: 2026-01-02\n", "---\n"] }],
		}),
	);
	const result = await convertNotebook({
		ipynbPath: ipynb,
		targetPath: path.join(dir, "d.md"),
		assetsDir: path.join(dir, "assets"),
	});
	assert.match(result.markdown, /published: 2026-01-02\n/);
	assert.doesNotMatch(result.markdown, /published: "2026-01-02"/);
});
```

顶部补 `import { convertNotebook } from "../src/utils/notebook/convertNotebook";`

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 模块找不到

- [ ] **Step 3: 实现 `src/utils/notebook/convertNotebook.ts`**

```ts
import path from "node:path";
import { cellToMarkdown, detectLanguage } from "./cellToMarkdown";
import { writeAsset } from "./extractAssets";
import type { OutputContext } from "./outputToMarkdown";
import { parseNotebook } from "./parseNotebook";
import { resolveMetadata } from "./resolveMetadata";

export interface ConvertedNotebook {
	frontmatter: Record<string, unknown>;
	/** 含 YAML frontmatter 的完整 markdown 文本 */
	markdown: string;
}

const DATE_FIELDS = new Set(["published", "updated"]);

function quote(text: string): string {
	return `"${text.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n")}"`;
}

function serializeValue(key: string, value: unknown): string {
	if (typeof value === "boolean" || typeof value === "number") return String(value);
	if (value instanceof Date) return value.toISOString().slice(0, 10);
	if (Array.isArray(value)) {
		return `[${value.map((item) => quote(String(item))).join(", ")}]`;
	}
	const text = String(value);
	// 日期必须是不加引号的 YAML 标量，否则 Astro 的 z.date() 会拿到字符串
	if (DATE_FIELDS.has(key) && /^\d{4}-\d{2}-\d{2}$/.test(text)) return text;
	return quote(text);
}

function serializeFrontmatter(frontmatter: Record<string, unknown>): string {
	const lines: string[] = ["---"];
	for (const [key, value] of Object.entries(frontmatter)) {
		if (value === undefined || value === null || value === "") continue;
		lines.push(`${key}: ${serializeValue(key, value)}`);
	}
	lines.push("---");
	return lines.join("\n");
}

function toPosix(value: string): string {
	return value.split(path.sep).join("/");
}

export async function convertNotebook(options: {
	ipynbPath: string;
	targetPath: string;
	assetsDir: string;
}): Promise<ConvertedNotebook> {
	const doc = await parseNotebook(options.ipynbPath);
	const { frontmatter, firstBodyCell } = resolveMetadata(doc, path.basename(options.ipynbPath));
	const language = detectLanguage(doc);
	const targetDir = path.dirname(options.targetPath);

	const context: OutputContext = {
		resolveAsset: async (base64, extension) => {
			const asset = await writeAsset({
				base64,
				extension,
				directory: options.assetsDir,
			});
			return toPosix(path.relative(targetDir, asset.absolutePath));
		},
		resolveAttachment: async (name, data) => {
			const mime = Object.keys(data)[0];
			if (mime === undefined) return `attachment:${name}`;
			const asset = await writeAsset({
				base64: data[mime],
				extension: mime.split("/")[1] ?? "bin",
				directory: options.assetsDir,
			});
			return toPosix(path.relative(targetDir, asset.absolutePath));
		},
	};

	const blocks: string[] = [];
	for (const cell of doc.cells.slice(firstBodyCell)) {
		const rendered = await cellToMarkdown(cell, language, context);
		if (rendered.trim() !== "") blocks.push(rendered.trim());
	}

	const body = blocks.join("\n\n");
	return {
		frontmatter,
		markdown: `${serializeFrontmatter(frontmatter)}\n\n${body}\n`,
	};
}
```

- [ ] **Step 4: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 全部 `ok`

- [ ] **Step 5: 提交**

```bash
git add src/utils/notebook/convertNotebook.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): orchestrate ipynb to markdown conversion"
```

---

### Task 7: 批量生成、清理与幂等

**Files:**
- Create: `src/utils/notebook/generate.ts`
- Modify: `scripts/check-notebooks.ts`

**Interfaces:**
- Consumes: `convertNotebook`（Task 6）、`NotebookError`（Task 2）
- Produces:
  - `interface GenerateReport { generated: string[]; errors: Array<{ file: string; message: string }> }`
  - `generateNotebooks(options: { postsDir: string; assetsDir: string }): Promise<GenerateReport>`

`generated` 与 `errors[].file` 均为**相对 postsDir 的 posix 路径**。

- [ ] **Step 1: 加断言**

```ts
section("generateNotebooks");
const writeNotebook = async (file: string, cells: unknown[]): Promise<void> => {
	await fs.mkdir(path.dirname(file), { recursive: true });
	await fs.writeFile(file, JSON.stringify({ nbformat: 4, metadata: {}, cells }));
};

await checkAsync("生成同名 md 并维护 .gitignore", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "a.ipynb"), [
		{ cell_type: "markdown", source: ["---\n", 'title: "A"\n', "published: 2026-01-01\n", "---\n"] },
	]);
	const report = await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.deepEqual(report.generated, ["a.md"]);
	assert.deepEqual(report.errors, []);
	const ignore = await fs.readFile(path.join(posts, ".gitignore"), "utf-8");
	assert.match(ignore, /^# 自动生成/);
	assert.match(ignore, /a\.md/);
});

await checkAsync("内容未变时不重写文件（mtime 不变）", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "b.ipynb"), [
		{ cell_type: "markdown", source: ["---\n", 'title: "B"\n', "published: 2026-01-01\n", "---\n"] },
	]);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	const target = path.join(posts, "b.md");
	const before = (await fs.stat(target)).mtimeMs;
	await new Promise((resolve) => setTimeout(resolve, 20));
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal((await fs.stat(target)).mtimeMs, before);
});

await checkAsync("notebook 删除后生成的 md 被清理", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	const ipynb = path.join(posts, "c.ipynb");
	await writeNotebook(ipynb, [
		{ cell_type: "markdown", source: ["---\n", 'title: "C"\n', "published: 2026-01-01\n", "---\n"] },
	]);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	await fs.rm(ipynb);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	// 产物为空时 .gitignore 也被删除，目录应当彻底空掉
	assert.deepEqual(await fs.readdir(posts), []);
});

await checkAsync("与手写 md 撞名时报错并跳过", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await fs.writeFile(path.join(posts, "d.md"), "# 手写内容");
	await writeNotebook(path.join(posts, "d.ipynb"), [
		{ cell_type: "markdown", source: ["---\n", 'title: "D"\n', "published: 2026-01-01\n", "---\n"] },
	]);
	const report = await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal(report.errors.length, 1);
	assert.equal(await fs.readFile(path.join(posts, "d.md"), "utf-8"), "# 手写内容");
});

await checkAsync("下划线开头的 notebook 被拒绝", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "_hidden.ipynb"), []);
	const report = await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal(report.errors.length, 1);
	assert.match(report.errors[0].message, /下划线/);
});

await checkAsync("坏 notebook 不影响其他文件", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await fs.writeFile(path.join(posts, "broken.ipynb"), "{ not json");
	await writeNotebook(path.join(posts, "good.ipynb"), [
		{ cell_type: "markdown", source: ["---\n", 'title: "G"\n', "published: 2026-01-01\n", "---\n"] },
	]);
	const report = await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal(report.errors.length, 1);
	assert.deepEqual(report.generated, ["good.md"]);
});
```

顶部补 `import { generateNotebooks } from "../src/utils/notebook/generate";`

- [ ] **Step 2: 运行，确认失败**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 模块找不到

- [ ] **Step 3: 实现 `src/utils/notebook/generate.ts`**

```ts
import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";
import { convertNotebook } from "./convertNotebook";
import { NotebookError } from "./parseNotebook";

export interface GenerateReport {
	generated: string[];
	errors: Array<{ file: string; message: string }>;
}

const MARKER = "<!-- generated-from-notebook:";
const IGNORE_HEADER = [
	"# 自动生成，请勿手改 —— 由 src/utils/notebook/generate.ts 维护",
	"# 下列文件由 notebook 生成，不入库",
].join("\n");

function toPosix(value: string): string {
	return value.split(path.sep).join("/");
}

async function readHead(filePath: string, length = 120): Promise<string> {
	try {
		const handle = await fs.open(filePath, "r");
		try {
			const buffer = Buffer.alloc(length);
			const { bytesRead } = await handle.read(buffer, 0, length, 0);
			return buffer.subarray(0, bytesRead).toString("utf-8");
		} finally {
			await handle.close();
		}
	} catch {
		return "";
	}
}

/** 找出目录下所有由生成器产出的 md（靠正文首行标记识别，不依赖任何清单） */
async function findGeneratedFiles(postsDir: string): Promise<string[]> {
	const files = await glob("**/*.md", { cwd: postsDir, posix: true });
	const generated: string[] = [];
	for (const file of files) {
		const head = await readHead(path.join(postsDir, file));
		if (head.includes(MARKER)) generated.push(file);
	}
	return generated.sort();
}

async function writeManifest(postsDir: string, generated: string[]): Promise<void> {
	const filePath = path.join(postsDir, ".gitignore");
	if (generated.length === 0) {
		await fs.rm(filePath, { force: true });
		return;
	}
	const content = `${IGNORE_HEADER}\n${generated.map((file) => `/${file}`).join("\n")}\n`;
	const existing = await fs.readFile(filePath, "utf-8").catch(() => null);
	if (existing !== content) {
		await fs.writeFile(filePath, content);
	}
}

export async function generateNotebooks(options: {
	postsDir: string;
	assetsDir: string;
}): Promise<GenerateReport> {
	const notebooks = (await glob("**/*.ipynb", { cwd: options.postsDir, posix: true })).sort();
	const report: GenerateReport = { generated: [], errors: [] };

	for (const notebook of notebooks) {
		const target = notebook.replace(/\.ipynb$/i, ".md");
		if (path.posix.basename(notebook).startsWith("_")) {
			report.errors.push({
				file: notebook,
				message: "文件名以下划线开头：content 目录下 `_` 前缀是保留语义，请重命名",
			});
			continue;
		}

		const targetPath = path.join(options.postsDir, target);
		const existingHead = await readHead(targetPath);
		if (existingHead !== "" && !existingHead.includes(MARKER)) {
			report.errors.push({
				file: notebook,
				message: `目标文件 ${target} 已存在且不是生成的，请重命名 notebook 或删除该文件`,
			});
			continue;
		}

		try {
			const converted = await convertNotebook({
				ipynbPath: path.join(options.postsDir, notebook),
				targetPath,
				assetsDir: path.join(options.assetsDir, notebook.replace(/\.ipynb$/i, "")),
			});
			// 标记必须在首行：撞名检测只读文件开头 120 字节
			const marker = `${MARKER}${notebook} -->`;
			const content = `${marker}\n\n${converted.markdown.replace(/\n+$/, "")}\n`;
			const existing = await fs.readFile(targetPath, "utf-8").catch(() => null);
			if (existing !== content) {
				await fs.mkdir(path.dirname(targetPath), { recursive: true });
				await fs.writeFile(targetPath, content);
			}
			report.generated.push(target);
		} catch (error) {
			const message = error instanceof NotebookError ? error.message : (error as Error).message;
			report.errors.push({ file: notebook, message });
		}
	}

	const stale = (await findGeneratedFiles(options.postsDir)).filter(
		(file) => !report.generated.includes(file),
	);
	for (const file of stale) {
		await fs.rm(path.join(options.postsDir, file), { force: true });
	}

	await writeManifest(options.postsDir, report.generated);
	return report;
}
```

> 说明：标记置于首行，`readHead` 只读前 120 字节即可判定文件来源，
> 因此撞名检测与孤儿清理都不依赖任何清单文件，删除 `.gitignore` 也能自愈。

- [ ] **Step 4: 运行，确认全部通过**

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 全部 `ok`

- [ ] **Step 5: 提交**

```bash
git add src/utils/notebook/generate.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): generate markdown, prune orphans and maintain gitignore"
```

---

### Task 8: 接入内容集合

**Files:**
- Create: `src/loaders/notebook-posts.ts`
- Modify: `src/content.config.ts`
- Modify: `scripts/check-notebooks.ts`（只加一条：确认集合能取到生成的文章，见 Step 5）

**Interfaces:**
- Consumes: `generateNotebooks`（Task 7）、`astro/loaders` 的 `glob`
- Produces: `notebookPosts(options?: { base?: string; assetsDir?: string }): Loader`

- [ ] **Step 1: 实现 `src/loaders/notebook-posts.ts`**

```ts
import path from "node:path";
import { fileURLToPath } from "node:url";
import { glob } from "astro/loaders";
import type { Loader } from "astro/loaders";
import { generateNotebooks } from "../utils/notebook/generate";

export interface NotebookPostsOptions {
	/** 相对项目根目录的文章目录 */
	base?: string;
	/** 相对项目根目录的 notebook 图片目录 */
	assetsDir?: string;
}

function resolveDir(root: URL, relative: string): string {
	return path.join(fileURLToPath(root), relative.replace(/^\.?\//, "").replace(/\/$/, ""));
}

export function notebookPosts(options: NotebookPostsOptions = {}): Loader {
	const base = options.base ?? "./src/content/posts";
	const assets = options.assetsDir ?? "./src/assets/notebooks";
	const inner = glob({ pattern: "**/*.{md,mdx}", base });

	return {
		name: "blog:notebook-posts",
		load: async (context) => {
			const postsDir = resolveDir(context.config.root, base);
			const assetsDir = resolveDir(context.config.root, assets);

			const report = await generateNotebooks({ postsDir, assetsDir });
			for (const error of report.errors) {
				context.logger.warn(`${error.file}: ${error.message}`);
			}

			await inner.load(context);
		},
	};
}
```

- [ ] **Step 2: 改 `src/content.config.ts`**

把 posts 集合的 loader 换掉（保留 `spec` 原样）：

```ts
import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import { notebookPosts } from "./loaders/notebook-posts";

const postsCollection = defineCollection({
	loader: notebookPosts({ base: "./src/content/posts" }),
	schema: z.object({
		/* 保持不变 */
	}),
});
```

> `spec` 集合继续用 `glob({ pattern: "**/*.{md,mdx}", base: "./src/content/spec" })`。

- [ ] **Step 3: 造一个真实 notebook 验证集合联通**

```bash
cat > src/content/posts/probe-notebook.ipynb <<'EOF'
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {"language_info": {"name": "python"}},
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["---\n", "title: \"联通性验证\"\n", "published: 2026-09-25\n", "---\n"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["正文一段。\n"]},
  {"cell_type": "code", "metadata": {}, "execution_count": 1, "source": ["print(1+1)"],
   "outputs": [{"output_type": "stream", "name": "stdout", "text": ["2\n"]}]}
 ]
}
EOF
```

- [ ] **Step 4: 启动 dev，确认页面渲染**

Run: `npx pnpm@9.14.4 exec astro dev --port 4399 --background` 然后 `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:4399/posts/probe-notebook/`
Expected: `200`

再确认正文与代码块：

Run: `curl -s http://localhost:4399/posts/probe-notebook/ | grep -c 'expressive-code'`
Expected: 大于 0（代码块走了 expressive-code）

Run: `ls src/content/posts/` 与 `cat src/content/posts/.gitignore`
Expected: 出现 `probe-notebook.md`，`.gitignore` 列出 `/probe-notebook.md`

Run: `git status --porcelain src/content/posts`
Expected: **只有** `?? src/content/posts/.gitignore`（生成物被忽略，不出现在 status）

- [ ] **Step 5: 加集合级断言并收尾**

```ts
section("collection");
await checkAsync("生成的 md 带 frontmatter 且能被 glob 命中", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-col-"));
	const assets = path.join(posts, "..", "assets-out");
	await fs.writeFile(
		path.join(posts, "e.ipynb"),
		JSON.stringify({
			nbformat: 4,
			metadata: {},
			cells: [{ cell_type: "markdown", source: ["---\n", 'title: "E"\n', "published: 2026-01-01\n", "---\n"] }],
		}),
	);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	const generated = await fs.readFile(path.join(posts, "e.md"), "utf-8");
	assert.match(generated, /^<!-- generated-from-notebook: e\.ipynb -->/);
	assert.match(generated, /published: 2026-01-01/);
});
```

Run: `npx tsx scripts/check-notebooks.ts`
Expected: 全部 `ok`

- [ ] **Step 6: 提交**

```bash
git add src/loaders/notebook-posts.ts src/content.config.ts scripts/check-notebooks.ts
git commit -m "feat(notebook): load notebook posts through the posts content collection"
```

---

### Task 9: dev 热更新

**Files:**
- Modify: `src/loaders/notebook-posts.ts`
- Modify: `scripts/check-notebooks.ts`（此处不加断言，dev 行为由实跑验证）

**Interfaces:**
- Consumes: Task 8 的 loader
- Produces: 无新导出（内部注册 Vite watcher 监听）

- [ ] **Step 1: 在 loader 中注册监听**

`src/loaders/notebook-posts.ts` 增加模块级状态与注册函数（`load` 每次同步都会执行，因此必须去重）：

```ts
let registeredWatcher: object | null = null;
let latestContext: LoaderContext | null = null;
let pendingTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleGeneration(postsDir: string, assetsDir: string): void {
	if (pendingTimer !== null) clearTimeout(pendingTimer);
	pendingTimer = setTimeout(() => {
		pendingTimer = null;
		const context = latestContext;
		if (context === null) return;
		void generateNotebooks({ postsDir, assetsDir }).then((report) => {
			for (const error of report.errors) {
				context.logger.warn(`${error.file}: ${error.message}`);
			}
		});
	}, 150);
}

function registerWatcher(
	context: LoaderContext,
	postsDir: string,
	assetsDir: string,
): void {
	const watcher = context.watcher;
	if (!watcher || registeredWatcher === watcher) return;
	registeredWatcher = watcher;
	const onEvent = (changedPath: string): void => {
		if (!changedPath.toLowerCase().endsWith(".ipynb")) return;
		scheduleGeneration(postsDir, assetsDir);
	};
	watcher.on("change", onEvent);
	watcher.on("add", onEvent);
	watcher.on("unlink", onEvent);
}
```

`load` 里改为：

```ts
		load: async (context) => {
			latestContext = context;
			const postsDir = resolveDir(context.config.root, base);
			const assetsDir = resolveDir(context.config.root, assets);

			const report = await generateNotebooks({ postsDir, assetsDir });
			for (const error of report.errors) {
				context.logger.warn(`${error.file}: ${error.message}`);
			}

			await inner.load(context);
			registerWatcher(context, postsDir, assetsDir);
		},
```

顶部补 `import type { Loader, LoaderContext } from "astro/loaders";`

- [ ] **Step 2: 实跑验证「改动 notebook 即时生效」**

dev 已在 Task 8 起在 4399。执行：

```bash
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("src/content/posts/probe-notebook.ipynb")
nb = json.loads(p.read_text())
nb["cells"][1]["source"] = ["改过的正文标记 CHANGE_ME。\n"]
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
EOF
sleep 3
curl -s http://localhost:4399/posts/probe-notebook/ | grep -c CHANGE_ME
```

Expected: 输出大于 0（无需重启 dev）

- [ ] **Step 3: 实跑验证「新增 notebook 即时生效」**

```bash
cp src/content/posts/probe-notebook.ipynb src/content/posts/probe-second.ipynb
sleep 3
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:4399/posts/probe-second/
```

Expected: `200`（新文件被 glob 的 `add` 事件接住，无需重启）

- [ ] **Step 4: 实跑验证「删除 notebook 即时生效」**

```bash
rm src/content/posts/probe-second.ipynb
sleep 3
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:4399/posts/probe-second/
```

Expected: `404`

- [ ] **Step 5: 提交**

```bash
git add src/loaders/notebook-posts.ts
git commit -m "feat(notebook): regenerate notebook posts on file changes in dev"
```

---

### Task 10: 构建接线

**Files:**
- Create: `scripts/generate-notebooks.ts`
- Modify: `package.json`、`.gitignore`、`astro.config.mjs`

**Interfaces:**
- Consumes: `generateNotebooks`（Task 7）
- Produces: 无导出

- [ ] **Step 1: 实现命令行入口 `scripts/generate-notebooks.ts`**

```ts
import path from "node:path";
import process from "node:process";
import { generateNotebooks } from "../src/utils/notebook/generate";

const root = process.cwd();

const report = await generateNotebooks({
	postsDir: path.join(root, "src/content/posts"),
	assetsDir: path.join(root, "src/assets/notebooks"),
});

for (const error of report.errors) {
	console.error(`[notebooks] ${error.file}: ${error.message}`);
}
console.log(
	`[notebooks] 生成 ${report.generated.length} 篇，${report.errors.length} 篇被跳过`,
);

process.exit(report.errors.length === 0 ? 0 : 1);
```

- [ ] **Step 2: 接线 `package.json`**

`scripts` 增加 `notebooks`，并把 `build` 改为（生成必须早于 `generate-lqips.ts`，否则 notebook 配图拿不到 LQIP）：

```json
"notebooks": "npx tsx scripts/generate-notebooks.ts",
"build": "node scripts/generate-icons.js && npx tsx scripts/generate-notebooks.ts && npx tsx scripts/generate-lqips.ts && astro build && npx tsx scripts/subset-fonts.ts && pagefind --site dist",
```

- [ ] **Step 3: `.gitignore` 增加图片目录**

在仓库根 `.gitignore` 末尾追加：

```
# notebook 抽取的图片（由 src/utils/notebook/generate.ts 生成）
/src/assets/notebooks/
```

- [ ] **Step 4: `astro.config.mjs` 忽略频繁重写的 `.gitignore`**

把 `vite.server.watch.ignored` 改为：

```js
		server: {
			watch: {
				ignored: ["**/package/**", "**/Firefly-docs/**", "**/src/content/posts/.gitignore"],
			},
		},
```

- [ ] **Step 5: 验证脚本本身**

Run: `npx tsx scripts/generate-notebooks.ts`
Expected: 打印 `[notebooks] 生成 1 篇，0 篇被跳过`，退出码 0

- [ ] **Step 6: 验证完整构建与 LQIP 覆盖**

```bash
rm -rf src/assets/notebooks src/constants/lqips.json.bak
cp src/constants/lqips.json src/constants/lqips.json.bak
npx pnpm@9.14.4 run build
node -e "const l=require('./src/constants/lqips.json'); const k=Object.keys(l).filter(x=>x.includes('notebooks')); console.log('notebook LQIP 条目:', k.length)"
```

Expected: 构建成功；若 `probe-notebook.ipynb` 含图片输出则条目数 > 0，否则为 0（本步只需确认构建通过、命令不报错）
清理：`rm -f src/constants/lqips.json.bak`

- [ ] **Step 7: 提交**

```bash
git add scripts/generate-notebooks.ts package.json .gitignore astro.config.mjs
git commit -m "build(notebook): wire notebook generation into build pipeline"
```

---

### Task 11: 样式与写作文档

**Files:**
- Modify: `src/styles/markdown.css`
- Modify: `docs/writing-guide.md`

- [ ] **Step 1: 加样式**

在 `src/styles/markdown.css` 的 `.custom-md { ... }` 块内（与 `.expressive-code` 同级处）加入：

```css
  .nb-output {
    @apply my-4 rounded-(--radius-large) overflow-x-auto;
    background: var(--codeblock-bg);
    color: var(--codeblock-content, inherit);
    padding: 0.75rem 1rem;
    font-family: var(--font-jetbrains-mono), ui-monospace, monospace;
    font-size: 0.875rem;
    line-height: 1.5rem;
    white-space: pre;
  }

  .nb-output-html {
    @apply my-4 overflow-x-auto;
  }

  .nb-stderr {
    @apply my-4;
  }
```

> `--codeblock-bg` 定义在 `src/styles/variables.styl`。若 `--codeblock-content` 不存在，
> 去掉那一行而不是新造变量。

- [ ] **Step 2: 视觉验证**

Run: `curl -s http://localhost:4399/posts/probe-notebook/ | grep -o 'class="nb-[a-z-]*"' | sort -u`
Expected: 至少出现 `nb-output` 或 `nb-stderr` 之一（取决于样例输出是否含 ANSI）

随后用浏览器确认深色/浅色两种主题下 stdout 块可读、长行可横向滚动。

- [ ] **Step 3: 补写作文档**

在 `docs/writing-guide.md` 增加一节（按该文件既有风格与语言书写），内容覆盖：

1. 把 `.ipynb` 放进 `src/content/posts/`，文件名即 URL（`softmax.ipynb` → `/posts/softmax/`）
2. 元数据两种写法：首单元格 YAML frontmatter（与普通文章一致），或 Jupyter 的 `Edit → Edit Notebook Metadata` 里的 `metadata.blog`
3. 优先级：首单元格 YAML 覆盖 `metadata.blog`；`published` 必填
4. 约束：文件名不得以 `_` 开头；不要手写与 notebook 同名的 `.md`（会撞名报错）
5. 生成的 `.md` 与抽取的图片不入库（`.gitignore` 自动维护），改 notebook 后 dev 会自动重新生成
6. 支持与不支持：渲染已保存输出（含图片、HTML 表格、报错回溯、ANSI 颜色），**不执行** notebook

- [ ] **Step 4: 提交**

```bash
git add src/styles/markdown.css docs/writing-guide.md
git commit -m "docs(notebook): document notebook posts and style outputs"
```

---

### Task 12: 端到端验收与清理

**Files:**
- Create（临时）: `src/content/posts/verify-nb-a.ipynb`、`verify-nb-b.ipynb`、`verify-nb-c.ipynb`
- 清理: 上述三个文件、`probe-notebook.ipynb`、生成的 `.md`、`.gitignore`

- [ ] **Step 1: 造三个覆盖性样例**

`verify-nb-a.ipynb`：首单元格 YAML + markdown 单元格含 `$E=mc^2$` 与 ```mermaid 围栏 + 一个 `python` 代码单元格。
`verify-nb-b.ipynb`：代码单元格输出 base64 PNG、`stdout`、`HTML` 表格 `<table><tr><td>x</td></tr></table>`、以及 `error` 输出（`ename: ValueError`）。
`verify-nb-c.ipynb`：`stdout` 输出含 `\u001b[31m`/`\u001b[0m`，`error` 的 `traceback` 含 ANSI。

（用 python 脚本生成，确保 JSON 合法。）

- [ ] **Step 2: 逐项核对渲染结果**

```bash
for s in verify-nb-a verify-nb-b verify-nb-c; do printf "%-14s " "$s"; curl -s -o /dev/null -w '%{http_code}\n' "http://localhost:4399/posts/$s/"; done
curl -s http://localhost:4399/posts/verify-nb-a/ | grep -o 'katex' | head -1
curl -s http://localhost:4399/posts/verify-nb-a/ | grep -o 'mermaid' | head -1
curl -s http://localhost:4399/posts/verify-nb-b/ | grep -o 'nb-output-html' | head -1
curl -s http://localhost:4399/posts/verify-nb-c/ | grep -o '<span style="color:' | head -1
curl -s http://localhost:4399/posts/verify-nb-b/ | grep -o 'data:image/png;base64' | head -1
```

Expected：三个路由 200；出现 `katex`；出现 `mermaid`；出现 `nb-output-html`；出现彩色 `<span style="color:`；**最后一条无输出**（图片已抽成文件，不是内联 base64）

- [ ] **Step 3: 确认图片进入优化与 LQIP**

```bash
ls src/assets/notebooks/verify-nb-b/
npx tsx scripts/generate-notebooks.ts && npx tsx scripts/generate-lqips.ts
node -e "const l=require('./src/constants/lqips.json'); console.log(Object.keys(l).filter(k=>k.includes('notebooks')).length)"
git status --porcelain
```

Expected：图片文件存在；LQIP 条目数 > 0；`git status` 中**不出现**任何生成物
（`src/assets/notebooks/` 与生成的 `.md` 均被忽略）；`src/constants/lqips.json` 会显示为已修改
—— 这是预期内的，因为它是**已提交**的生成文件，且本次新增了 notebook 图片条目。
该改动会在 Step 4 连同样例一起回退。

- [ ] **Step 4: 清理临时样例并复跑断言**

```bash
rm -f src/content/posts/verify-nb-*.ipynb src/content/posts/probe-notebook.ipynb
rm -rf src/assets/notebooks
npx tsx scripts/generate-notebooks.ts
git checkout -- src/constants/lqips.json
git status --porcelain src/content/posts src/constants
npx tsx scripts/check-notebooks.ts
```

Expected：`src/content/posts` 与 `src/constants` 下无残留（生成的 `.md` 已随 notebook 删除被清理、
`.gitignore` 因清单为空而自我删除、`lqips.json` 已回退）；断言脚本全部通过

- [ ] **Step 5: 全量检查**

```bash
npx pnpm@9.14.4 exec astro check
npx pnpm@9.14.4 exec biome check ./src
npx pnpm@9.14.4 exec tsc --noEmit --isolatedDeclarations
```

Expected：三者均无错误

- [ ] **Step 6: 停掉验证用 dev server 并提交**

**不要用 `git add -A`**：工作区里还有本次任务之外的改动（早先删除的 23 篇文章、仓库里的遗留文件），
一起提交会把它们卷进来。只提交本任务范围内的路径：

```bash
npx pnpm@9.14.4 exec astro dev stop
git add src/utils/notebook src/loaders src/content.config.ts src/styles/markdown.css \
        scripts/generate-notebooks.ts scripts/check-notebooks.ts \
        docs/writing-guide.md docs/superpowers package.json astro.config.mjs .gitignore
git status --short
git commit -m "feat(notebook): render .ipynb files as blog posts"
```

提交前用 `git status --short` 复核：**不应**出现 `src/content/posts/**` 的删除、
`src/constants/lqips.json` 的修改、以及 `Firefly-master.zip` / `test-write.txt` 等遗留文件。

---

## Self-Review

**Spec 覆盖对照**

| Spec 章节 | 对应任务 |
|---|---|
| §4 数据流 | Task 6（转换）、Task 7（生成）、Task 8（接入） |
| §5 模块设计 | Task 1–7（每个模块一个任务） |
| §6 元数据规范 | Task 3（合并与校验）、Task 6（序列化，含日期不加引号） |
| §7 单元格与输出规则 | Task 5 |
| §8 资源抽取 | Task 4 |
| §9 幂等/监听/清理 | Task 7（幂等与清理）、Task 9（监听）、Task 10 Step 4（watch ignore） |
| §10 错误处理 | Task 2/3（解析与元数据错误）、Task 7（撞名、下划线、单文件隔离） |
| §11 集成点 | Task 8（content.config）、Task 10（package.json/.gitignore/astro.config）、Task 11（样式与文档） |
| §12 验证计划 | Task 12 |
| §13 风险 | Global Constraints 已固化（不引入依赖、不用 `_` 前缀、不 sanitize HTML） |

**占位符扫描**：无 TBD/TODO；每个代码步骤都给出完整可运行代码。

**类型一致性核对**

- `OutputContext.resolveAsset(base64, extension)` / `resolveAttachment(name, data)` 在 Task 5 定义，Task 6 实现 —— 签名一致。
- `NotebookError` 定义于 Task 2 的 `parseNotebook.ts`，Task 3/6/7 均从同一路径导入。
- `generateNotebooks({ postsDir, assetsDir })` 在 Task 7 定义，Task 8/10 调用一致。
- `convertNotebook({ ipynbPath, targetPath, assetsDir })` 在 Task 6 定义，Task 7 调用一致。
- `notebookPosts({ base, assetsDir })` 在 Task 8 定义。

**自检当场修正的问题（已直接写回上方任务，实现时无需再判断）**

1. **Task 3 测试辅助函数类型**：原稿用 `as never` 强转加一个裸 `checkAsync` 占位行，
   改为显式 `notebook(): NotebookDocument` 辅助函数。
2. **Task 5 未知 MIME 行为**：原稿 `throw`，与 spec §10「回退占位说明」及「单个文件失败不中断构建」
   冲突 —— 改为输出 HTML 注释占位，并移除本模块不再需要的 `NotebookError` 导入。
3. **Task 5 `stubContext` 类型**：原稿 `as never`，改为显式 `OutputContext` 并加 `import type`
   （仓库开启 `verbatimModuleSyntax`，类型导入必须用 `import type`）。
4. **Task 7 生成标记位置**：原稿放正文末尾，但撞名检测只读文件前 120 字节，必然读不到 ——
   改为置于首行；这也让 Task 8 Step 5 的首行断言成立。
5. **Task 7 清理断言**：原稿期望删除后目录仍有 `.gitignore`，与「清单为空则删除 `.gitignore`」
   自相矛盾 —— 改为期望空目录。
6. **Task 12 对 `lqips.json` 的污染**：它是**已提交**文件，跑 LQIP 后必然被改动 ——
   已在验收步骤显式 `git checkout` 回退，避免样例残留进入提交。
7. **Task 12 提交范围**：原稿 `git add -A` 会连带把早先删除的 23 篇文章、仓库遗留文件一起提交 ——
   改为显式列出本任务路径，并要求提交前用 `git status --short` 复核。
