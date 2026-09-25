import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { parseFrontmatter } from "@astrojs/markdown-remark";
import type { LoaderContext } from "astro/loaders";
import { notebookPosts } from "../src/loaders/notebook-posts";
import { ansiToHtml, hasAnsi } from "../src/utils/notebook/ansi";
import {
	cellToMarkdown,
	detectLanguage,
} from "../src/utils/notebook/cellToMarkdown";
import { convertNotebook } from "../src/utils/notebook/convertNotebook";
import { writeAsset } from "../src/utils/notebook/extractAssets";
import { fenced } from "../src/utils/notebook/fence";
import { generateNotebooks } from "../src/utils/notebook/generate";
import type { OutputContext } from "../src/utils/notebook/outputToMarkdown";
import { outputToMarkdown } from "../src/utils/notebook/outputToMarkdown";
import {
	NotebookError,
	parseNotebook,
} from "../src/utils/notebook/parseNotebook";
import { resolveMetadata } from "../src/utils/notebook/resolveMetadata";
import { toText } from "../src/utils/notebook/text";
import type {
	NotebookCell,
	NotebookDocument,
} from "../src/utils/notebook/types";

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

async function checkAsync(
	name: string,
	run: () => Promise<void>,
): Promise<void> {
	try {
		await run();
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
	assert.equal(
		ansiToHtml("\u001b[1mb\u001b[0mplain"),
		'<span style="font-weight:600">b</span>plain',
	);
});
check("HTML 特殊字符被转义", () => {
	assert.equal(ansiToHtml("<a> &"), "&lt;a&gt; &amp;");
});
check("256 色与 truecolor", () => {
	assert.equal(
		ansiToHtml("\u001b[38;5;196mX\u001b[0m"),
		'<span style="color:#ff0000">X</span>',
	);
	assert.equal(
		ansiToHtml("\u001b[38;2;18;52;86mX\u001b[0m"),
		'<span style="color:#123456">X</span>',
	);
});
check("未知序列被丢弃且不影响文本", () => {
	assert.equal(ansiToHtml("\u001b[99mX"), "X");
});
check("截断或越界的颜色参数被忽略且不抛错", () => {
	for (const params of [
		"38;2",
		"38;2;1",
		"38;2;1;2",
		"48;2;1;2",
		"38;5",
		"48;5;999",
	]) {
		assert.equal(ansiToHtml(`\u001b[${params}mX`), "X", params);
	}
	assert.equal(ansiToHtml("\u001b[38;2;300;0;0mX"), "X");
	assert.equal(ansiToHtml("\u001b[38;5;256mX"), "X");
	assert.equal(ansiToHtml("\u001b[38;5;999999999999999999999mX"), "X");
	assert.equal(
		ansiToHtml("\u001b[1;38;2mX"),
		'<span style="font-weight:600">X</span>',
	);
	assert.equal(
		ansiToHtml("\u001b[38;5;255mX\u001b[0m"),
		'<span style="color:#eeeeee">X</span>',
	);
	assert.equal(
		ansiToHtml("\u001b[38;2;0;0;0mX\u001b[0m"),
		'<span style="color:#000000">X</span>',
	);
});
check("非 SGR 控制序列被剥离", () => {
	assert.equal(ansiToHtml("\u001b[2KX\r"), "X\n");
	assert.equal(ansiToHtml("\u001b]8;;https://e.com\u001b\\link"), "link");
	assert.equal(ansiToHtml("\u001b]0;title\u0007X"), "X");
	assert.equal(ansiToHtml("\u001b[?25lX"), "X");
	// 被剥离的序列不改变 SGR 状态，因此 B 仍是红色（按转义边界切块，故为两个 span）
	assert.equal(
		ansiToHtml("\u001b[31mA\u001b[2KB"),
		'<span style="color:#cd0000">A</span><span style="color:#cd0000">B</span>',
	);
	// 输出中不留任何 ESC 字节
	assert.equal(
		ansiToHtml("\u001b[2K\u001b]8;;u\u001b\\A").includes("\u001b"),
		false,
	);
});
check("换行被规范化", () => {
	assert.equal(ansiToHtml("a\r\nb\rc"), "a\nb\nc");
});
check("hasAnsi 覆盖非 SGR 序列与裸 CR", () => {
	assert.equal(hasAnsi("\u001b[2Kx"), true);
	assert.equal(hasAnsi("\u001b]0;t\u0007x"), true);
	assert.equal(hasAnsi("a\r\nb"), true);
	assert.equal(hasAnsi("a\nb"), false);
});

section("text");
check("行数组拼接为字符串", () => {
	assert.equal(toText(["a\n", "b"]), "a\nb");
	assert.equal(toText("plain"), "plain");
	assert.equal(toText(undefined), "");
});
check("toText 对任意输入都返回字符串", () => {
	for (const input of [
		{ a: 1 },
		42,
		true,
		["a", "b"],
		"s",
		null,
		undefined,
		[],
	]) {
		assert.equal(typeof toText(input), "string");
	}
	assert.equal(toText({ a: 1 }), "");
	assert.equal(toText(42), "");
	assert.equal(toText(true), "");
	assert.equal(toText(["a", "b"]), "ab");
	assert.equal(toText("s"), "s");
	assert.equal(toText(null), "");
	assert.equal(toText(undefined), "");
	assert.equal(toText([]), "");
});

section("parseNotebook");
await checkAsync("合法 notebook 解析成功", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "ok.ipynb");
	await fs.writeFile(
		file,
		JSON.stringify({
			nbformat: 4,
			metadata: {},
			cells: [{ cell_type: "markdown", source: ["hi"] }],
		}),
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
await checkAsync("读文件失败报 NotebookError", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "missing.ipynb");
	await assert.rejects(() => parseNotebook(file), NotebookError);
});
await checkAsync("根节点非对象报 NotebookError", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-"));
	const file = path.join(dir, "scalar.ipynb");
	await fs.writeFile(file, JSON.stringify("x"));
	await assert.rejects(() => parseNotebook(file), NotebookError);
	await fs.writeFile(file, JSON.stringify([]));
	await assert.rejects(() => parseNotebook(file), NotebookError);
});

section("resolveMetadata");

function notebook(
	metadata: Record<string, unknown>,
	cells: NotebookCell[],
): NotebookDocument {
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

await checkAsync("published 为 null 等同于缺失", async () => {
	const doc = notebook({ blog: { title: "t", published: null } }, []);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});

await checkAsync("缺 title 时用文件名兜底", async () => {
	const doc = notebook({ blog: { published: "2026-01-01" } }, []);
	const result = resolveMetadata(doc, "my-notebook.ipynb");
	assert.equal(result.frontmatter.title, "my notebook");
});

await checkAsync("tags 类型错误报错而不漏到 schema", async () => {
	const doc = notebook(
		{ blog: { title: "t", published: "2026-01-01", tags: "CUDA" } },
		[],
	);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});

await checkAsync("YAML 损坏报错", async () => {
	// metadata.blog 给出合法 published，确保唯一失败源是损坏的 YAML
	const doc = notebook({ blog: { published: "2026-01-01" } }, [
		{ cell_type: "markdown", source: ["---\n", "a: [1,\n", "---\n"] },
	]);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});

await checkAsync(
	"首单元格 YAML 之后的正文被保留为 leadingMarkdown",
	async () => {
		const doc = notebook({ blog: { published: "2026-01-02" } }, [
			{
				cell_type: "markdown",
				source: [
					"---\n",
					'description: "d"\n',
					"---\n",
					"\n",
					"这里是引言，不能被丢掉。\n",
				],
			},
		]);
		const result = resolveMetadata(doc, "x.ipynb");
		assert.equal(result.leadingMarkdown, "这里是引言，不能被丢掉。");
		assert.equal(result.firstBodyCell, 1);
		assert.equal(result.frontmatter.description, "d");
	},
);

await checkAsync("首单元格只有 YAML 时 leadingMarkdown 为空串", async () => {
	const doc = notebook({ blog: { published: "2026-01-02" } }, [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "t"\n', "---\n"],
		},
	]);
	const result = resolveMetadata(doc, "x.ipynb");
	assert.equal(result.leadingMarkdown, "");
});

await checkAsync(
	"没有 frontmatter 单元格时 leadingMarkdown 为空串",
	async () => {
		const filled = notebook({ blog: { published: "2026-01-02" } }, [
			{ cell_type: "markdown", source: ["# 直接是正文\n"] },
		]);
		assert.equal(resolveMetadata(filled, "x.ipynb").leadingMarkdown, "");
		const empty = notebook({ blog: { published: "2026-01-02" } }, []);
		assert.equal(resolveMetadata(empty, "x.ipynb").leadingMarkdown, "");
	},
);

await checkAsync("prevTitle 等内部字段类型错误报错", async () => {
	const doc = notebook(
		{ blog: { published: "2026-01-02", prevTitle: 2024 } },
		[],
	);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
	const ok = notebook(
		{
			blog: { published: "2026-01-02", nextTitle: "下一篇", nextSlug: "next" },
		},
		[],
	);
	const result = resolveMetadata(ok, "x.ipynb");
	assert.equal(result.frontmatter.nextTitle, "下一篇");
	assert.equal(result.frontmatter.nextSlug, "next");
});

await checkAsync("未加引号与加引号的 YAML 日期归一为同一结果", async () => {
	const unquoted = notebook({}, [
		{
			cell_type: "markdown",
			source: ["---\n", "published: 2026-01-02\n", "---\n"],
		},
	]);
	const quoted = notebook({}, [
		{
			cell_type: "markdown",
			source: ["---\n", 'published: "2026-01-02"\n', "---\n"],
		},
	]);
	const a = resolveMetadata(unquoted, "x.ipynb").frontmatter.published;
	const b = resolveMetadata(quoted, "x.ipynb").frontmatter.published;
	assert.equal(a, "2026-01-02");
	assert.equal(a, b);
});

await checkAsync("带时间的 YAML 日期报错而不静默改日", async () => {
	// metadata.blog 给出同日合法值：若 Date 分支放行，本断言将不会抛错
	const doc = notebook({ blog: { published: "2026-01-02" } }, [
		{
			cell_type: "markdown",
			source: ["---\n", "published: 2026-01-01T23:00:00-05:00\n", "---\n"],
		},
	]);
	assert.throws(() => resolveMetadata(doc, "x.ipynb"), NotebookError);
});

await checkAsync("首单元格是代码时不消费，firstBodyCell 为 0", async () => {
	const doc = notebook({ blog: { published: "2026-01-02" } }, [
		{ cell_type: "code", source: ["1"] },
	]);
	const result = resolveMetadata(doc, "x.ipynb");
	assert.equal(result.firstBodyCell, 0);
	assert.equal(result.leadingMarkdown, "");
});

await checkAsync(
	"文件名推导不出标题时报错，不让空 title 漏到 schema",
	async () => {
		// 空 title 序列化时会被整键丢掉，schema 的 title: z.string() 无默认值 → 内容同步中断
		const doc = notebook({ blog: { published: "2026-01-02" } }, []);
		assert.throws(() => resolveMetadata(doc, "--.ipynb"), NotebookError);
	},
);

section("extractAssets");
await checkAsync("base64 落地为内容 hash 命名文件", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const payload = Buffer.from("hello-image").toString("base64");
	const first = await writeAsset({
		base64: payload,
		extension: "png",
		directory: dir,
	});
	assert.equal(first.fileName.endsWith(".png"), true);
	assert.equal(path.isAbsolute(first.absolutePath), true);
	assert.equal(await fs.readFile(first.absolutePath, "utf-8"), "hello-image");
});
await checkAsync("同一内容重复写入得到同一文件名且不重写", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const payload = Buffer.from("same").toString("base64");
	const a = await writeAsset({
		base64: payload,
		extension: "png",
		directory: dir,
	});
	const firstMtime = (await fs.stat(a.absolutePath)).mtimeMs;
	await delay(20);
	const b = await writeAsset({
		base64: payload,
		extension: "png",
		directory: dir,
	});
	assert.equal(a.fileName, b.fileName);
	assert.deepEqual(await fs.readdir(dir), [a.fileName]);
	// 重写会刷新 mtime，故 mtime 不变即证明命中已存在分支、未写盘。
	assert.equal((await fs.stat(a.absolutePath)).mtimeMs, firstMtime);
});
await checkAsync("不同内容得到不同文件名", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const a = await writeAsset({
		base64: Buffer.from("a").toString("base64"),
		extension: "png",
		directory: dir,
	});
	const b = await writeAsset({
		base64: Buffer.from("b").toString("base64"),
		extension: "png",
		directory: dir,
	});
	assert.notEqual(a.fileName, b.fileName);
});
await checkAsync("目录不存在时递归创建且 base64 容忍空白", async () => {
	const root = await fs.mkdtemp(path.join(os.tmpdir(), "nb-assets-"));
	const dir = path.join(root, "a", "b");
	const payload = Buffer.from("wrapped").toString("base64");
	const wrapped = `${payload.slice(0, 4)}\n  ${payload.slice(4)}\n`;
	const asset = await writeAsset({
		base64: wrapped,
		extension: "svg",
		directory: dir,
	});
	assert.equal(asset.fileName.endsWith(".svg"), true);
	assert.equal(await fs.readFile(asset.absolutePath, "utf-8"), "wrapped");
});

section("output / cell");

const stubContext: OutputContext = {
	resolveAsset: async (base64, extension) =>
		`/assets/${extension}/${base64.length}.${extension}`,
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
		{
			output_type: "stream",
			name: "stdout",
			text: ["\u001b[31mred\u001b[0m\n"],
		},
		stubContext,
	);
	assert.match(
		md,
		/^<pre class="nb-output"><span style="color:#cd0000">red<\/span><\/pre>$/,
	);
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
	assert.match(
		md,
		/^<div class="nb-output-html">\n<table>x<\/table>\n<\/div>$/,
	);
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
		{
			output_type: "execute_result",
			data: { "application/json": ['{"a":1}'] },
		},
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
		{
			output_type: "display_data",
			data: { "application/vnd.plotly.v1+json": { a: 1 } },
		},
		stubContext,
	);
	assert.match(md, /无法渲染的输出/);
});
await checkAsync("未知输出类型回退占位说明", async () => {
	const md = await outputToMarkdown(
		{ output_type: "update_display_data", data: { "text/plain": ["x"] } },
		stubContext,
	);
	assert.match(md, /无法渲染的输出类型/);
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
	const md = await cellToMarkdown(
		{ cell_type: "raw", source: ["x"] },
		"python",
		stubContext,
	);
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
await checkAsync("图片输出经 resolveAsset 落地", async () => {
	const md = await outputToMarkdown(
		{ output_type: "display_data", data: { "image/png": ["AAAA"] } },
		stubContext,
	);
	assert.equal(md, "![notebook 输出](/assets/png/4.png)");
});
await checkAsync("image/jpeg 输出用 jpg 扩展名", async () => {
	const md = await outputToMarkdown(
		{ output_type: "display_data", data: { "image/jpeg": ["BBBB"] } },
		stubContext,
	);
	assert.equal(md, "![notebook 输出](/assets/jpg/4.jpg)");
});
await checkAsync("SVG 输出原样透传，base64 则解码", async () => {
	const raw = await outputToMarkdown(
		{ output_type: "display_data", data: { "image/svg+xml": ["<svg/>"] } },
		stubContext,
	);
	assert.equal(raw, "<svg/>");
	const encoded = await outputToMarkdown(
		{
			output_type: "display_data",
			data: {
				"image/svg+xml": [Buffer.from("<svg>x</svg>").toString("base64")],
			},
		},
		stubContext,
	);
	assert.equal(encoded, "<svg>x</svg>");
});
await checkAsync("内容含反引号时围栏加长", async () => {
	const md = await outputToMarkdown(
		{ output_type: "stream", name: "stdout", text: ["a\n```\nb\n"] },
		stubContext,
	);
	assert.equal(md, "````text\na\n```\nb\n````");
});
check("语言别名归一", () => {
	assert.equal(
		detectLanguage({
			nbformat: 4,
			metadata: { language_info: { name: "Python3" } },
			cells: [],
		}),
		"python",
	);
	assert.equal(
		detectLanguage({
			nbformat: 4,
			metadata: { kernelspec: { language: "C++" } },
			cells: [],
		}),
		"cpp",
	);
	assert.equal(
		detectLanguage({
			nbformat: 4,
			metadata: { kernelspec: { language: "node" } },
			cells: [],
		}),
		"javascript",
	);
});
check("language_info 优先于 kernelspec，缺失时默认 python", () => {
	assert.equal(
		detectLanguage({
			nbformat: 4,
			metadata: {
				language_info: { name: "julia" },
				kernelspec: { language: "python" },
			},
			cells: [],
		}),
		"julia",
	);
	assert.equal(
		detectLanguage({ nbformat: 4, metadata: {}, cells: [] }),
		"python",
	);
	assert.equal(
		detectLanguage({
			nbformat: 4,
			metadata: { language_info: { name: "  " } },
			cells: [],
		}),
		"python",
	);
});

check("fenced 普通内容用三反引号并去掉尾部换行", () => {
	assert.equal(fenced("hello\n", "text"), "```text\nhello\n```");
	assert.equal(fenced("a\n\n", "python"), "```python\na\n```");
});
check("fenced 按内容里的最长反引号串加长", () => {
	assert.equal(fenced("a\n```\nb", "text"), "````text\na\n```\nb\n````");
	assert.equal(fenced("a\n````\nb", "text"), "`````text\na\n````\nb\n`````");
});
await checkAsync("cell 与 output 共用同一围栏实现", async () => {
	const snippet = "a\n```\nb\n";
	const fromCell = await cellToMarkdown(
		{ cell_type: "code", source: [snippet], outputs: [] },
		"text",
		stubContext,
	);
	const fromOutput = await outputToMarkdown(
		{ output_type: "stream", name: "stdout", text: [snippet] },
		stubContext,
	);
	assert.equal(fromCell, "````text\na\n```\nb\n````");
	assert.equal(fromOutput, fromCell);
});

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
				{
					cell_type: "markdown",
					source: [
						"---\n",
						'title: "示例"\n',
						"published: 2026-01-02\n",
						'tags: ["a"]\n',
						"---\n",
					],
				},
				{ cell_type: "markdown", source: ["公式 $x^2$\n"] },
				{
					cell_type: "code",
					source: ["1+1"],
					outputs: [
						{
							output_type: "execute_result",
							data: { "text/plain": ["2"] },
						},
					],
				},
				{
					cell_type: "code",
					source: ["plot()"],
					outputs: [
						{
							output_type: "display_data",
							data: { "image/png": [png] },
						},
					],
				},
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
	// 原计划的 /^#?/ 恒真（裁决 R2），改为校验 frontmatter 块真的存在
	assert.match(result.markdown, /^---\n/);
	// 键与值都经 JSON.stringify 转义：键不加引号时 `"a: b"` 这类键会产出坏 YAML
	assert.match(
		result.markdown,
		/^---\n"title": "示例"\n"published": 2026-01-02\n/,
	);
	assert.match(result.markdown, /"tags": \["a"\]\n/);
	assert.match(result.markdown, /公式 \$x\^2\$/);
	assert.match(result.markdown, /```python\n1\+1\n```\n\n```text\n2\n```/);
	assert.match(
		result.markdown,
		/!\[notebook 输出\]\(assets\/[0-9a-f]{16}\.png\)/,
	);
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
			cells: [
				{
					cell_type: "markdown",
					source: [
						"---\n",
						'title: "t"\n',
						"published: 2026-01-02\n",
						"updated: 2026-02-03\n",
						"---\n",
					],
				},
			],
		}),
	);
	const result = await convertNotebook({
		ipynbPath: ipynb,
		targetPath: path.join(dir, "d.md"),
		assetsDir: path.join(dir, "assets"),
	});
	assert.match(result.markdown, /"published": 2026-01-02\n/);
	assert.doesNotMatch(result.markdown, /published: "2026-01-02"/);
	assert.match(result.markdown, /"updated": 2026-02-03\n/);
	assert.doesNotMatch(result.markdown, /updated: "2026-02-03"/);
	// 正文为空时不留多余空行（frontmatter-only 也是合法文档）
	assert.equal(
		result.markdown,
		'---\n"title": "t"\n"published": 2026-01-02\n"updated": 2026-02-03\n---\n',
	);
});
await checkAsync(
	"首单元格 YAML 之后的正文排在 frontmatter 与正文块之间",
	async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
		const ipynb = path.join(dir, "lead.ipynb");
		await fs.writeFile(
			ipynb,
			JSON.stringify({
				nbformat: 4,
				metadata: {},
				cells: [
					{
						cell_type: "markdown",
						source: [
							"---\n",
							'title: "L"\n',
							"published: 2026-01-01\n",
							"---\n",
							"\n",
							"这里是引言，不能被丢掉。\n",
						],
					},
					{ cell_type: "code", source: ["2+2"], outputs: [] },
				],
			}),
		);
		const result = await convertNotebook({
			ipynbPath: ipynb,
			targetPath: path.join(dir, "lead.md"),
			assetsDir: path.join(dir, "assets"),
		});
		assert.match(
			result.markdown,
			/---\n\n这里是引言，不能被丢掉。\n\n```python\n2\+2\n```\n$/,
		);
	},
);
await checkAsync("空串与 null 的键不写进 frontmatter", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
	const ipynb = path.join(dir, "e.ipynb");
	await fs.writeFile(
		ipynb,
		JSON.stringify({
			nbformat: 4,
			metadata: {},
			cells: [
				{
					cell_type: "markdown",
					source: [
						"---\n",
						'title: "E"\n',
						"published: 2026-01-01\n",
						'description: ""\n',
						"image:\n",
						"category: null\n",
						"---\n",
					],
				},
			],
		}),
	);
	const result = await convertNotebook({
		ipynbPath: ipynb,
		targetPath: path.join(dir, "e.md"),
		assetsDir: path.join(dir, "assets"),
	});
	assert.equal(result.frontmatter.description, "");
	assert.equal(Object.hasOwn(result.frontmatter, "image"), false);
	assert.equal(Object.hasOwn(result.frontmatter, "category"), false);
	assert.equal(
		result.markdown,
		'---\n"title": "E"\n"published": 2026-01-01\n---\n',
	);
});
await checkAsync("需要引号的键被转义，产出仍是合法 YAML", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
	const ipynb = path.join(dir, "keys.ipynb");
	await fs.writeFile(
		ipynb,
		JSON.stringify({
			nbformat: 4,
			metadata: {
				blog: {
					title: "K",
					published: "2026-01-01",
					"a: b": 1,
					"需要 引号": true,
				},
			},
			cells: [],
		}),
	);
	const result = await convertNotebook({
		ipynbPath: ipynb,
		targetPath: path.join(dir, "keys.md"),
		assetsDir: path.join(dir, "assets"),
	});
	// 键不加引号时产出 `a: b: 1`，Astro 侧会以 bad indentation of a mapping
	// entry 中断整次内容同步；这里必须仍能解析且键原样保留
	const { frontmatter } = parseFrontmatter(result.markdown);
	assert.equal(frontmatter["a: b"], 1);
	assert.equal(frontmatter["需要 引号"], true);
	assert.equal(frontmatter.title, "K");
});

await checkAsync("子目录中的 notebook 用相对路径引用图片与附件", async () => {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
	const targetDir = path.join(dir, "p", "d");
	await fs.mkdir(targetDir, { recursive: true });
	const png = Buffer.from("attachment-png").toString("base64");
	const ipynb = path.join(targetDir, "sub.ipynb");
	await fs.writeFile(
		ipynb,
		JSON.stringify({
			nbformat: 4,
			metadata: {},
			cells: [
				{
					cell_type: "markdown",
					source: ["---\n", 'title: "S"\n', "published: 2026-01-01\n", "---\n"],
				},
				{
					cell_type: "markdown",
					source: ["![示意图](attachment:fig.png)\n"],
					attachments: { "fig.png": { "image/png": png } },
				},
				{
					cell_type: "code",
					source: ["plot()"],
					outputs: [
						{
							output_type: "display_data",
							data: { "image/png": [png] },
						},
					],
				},
			],
		}),
	);
	const assetsDir = path.join(dir, "as", "sub");
	const result = await convertNotebook({
		ipynbPath: ipynb,
		targetPath: path.join(targetDir, "sub.md"),
		assetsDir,
	});
	assert.ok(result.markdown.includes("![示意图]("));
	assert.doesNotMatch(result.markdown, /attachment:fig\.png/);
	const references = [...result.markdown.matchAll(/\]\(([^)]+)\)/g)].map(
		(match) => match[1],
	);
	assert.equal(references.length, 2);
	const [attachmentRef, outputRef] = references;
	// 目标在 p/d 下、图片在 as 下 → 必须上溯两层，而不是写死 `..`
	assert.match(attachmentRef, /^\.\.\/\.\.\/as\/sub\/[0-9a-f]{16}\.png$/);
	// 附件与输出内容相同 → 命中同一个内容哈希文件
	assert.equal(attachmentRef, outputRef);
	for (const reference of references) {
		await fs.access(path.resolve(targetDir, reference));
	}
	assert.equal((await fs.readdir(assetsDir)).length, 1);
});

await checkAsync(
	"没有 YAML 单元格时用 metadata.blog 做 frontmatter",
	async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "nb-convert-"));
		const ipynb = path.join(dir, "b.ipynb");
		await fs.writeFile(
			ipynb,
			JSON.stringify({
				nbformat: 4,
				metadata: { blog: { title: "B", published: "2026-05-06" } },
				cells: [{ cell_type: "markdown", source: ["# 正文首格\n"] }],
			}),
		);
		const result = await convertNotebook({
			ipynbPath: ipynb,
			targetPath: path.join(dir, "b.md"),
			assetsDir: path.join(dir, "assets"),
		});
		assert.equal(result.frontmatter.title, "B");
		// firstBodyCell 为 0：首格必须照常渲染，且不因空的 leadingMarkdown 多出空块
		assert.equal(
			result.markdown,
			'---\n"title": "B"\n"published": 2026-05-06\n---\n\n# 正文首格\n',
		);
	},
);

section("generateNotebooks");
const writeNotebook = async (file: string, cells: unknown[]): Promise<void> => {
	await fs.mkdir(path.dirname(file), { recursive: true });
	await fs.writeFile(
		file,
		JSON.stringify({ nbformat: 4, metadata: {}, cells }),
	);
};

await checkAsync("生成同名 md 并维护 .gitignore", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "a.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "A"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
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
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "B"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	const target = path.join(posts, "b.md");
	const before = (await fs.stat(target)).mtimeMs;
	await delay(20);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal((await fs.stat(target)).mtimeMs, before);
});

await checkAsync("notebook 删除后生成的 md 被清理", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	const ipynb = path.join(posts, "c.ipynb");
	await writeNotebook(ipynb, [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "C"\n', "published: 2026-01-01\n", "---\n"],
		},
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
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "D"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.equal(report.errors.length, 1);
	assert.equal(
		await fs.readFile(path.join(posts, "d.md"), "utf-8"),
		"# 手写内容",
	);
});

await checkAsync("下划线开头的 notebook 被拒绝", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "_hidden.ipynb"), []);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.equal(report.errors.length, 1);
	assert.match(report.errors[0].message, /下划线/);
});

await checkAsync("坏 notebook 不影响其他文件", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await fs.writeFile(path.join(posts, "broken.ipynb"), "{ not json");
	await writeNotebook(path.join(posts, "good.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "G"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.equal(report.errors.length, 1);
	assert.deepEqual(report.generated, ["good.md"]);
});

await checkAsync(
	"嵌套 notebook：首行标记写相对路径，.gitignore 逐行锚定",
	async () => {
		const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
		const assets = path.join(posts, "..", "assets-out");
		await writeNotebook(path.join(posts, "sub", "deep", "n.ipynb"), [
			{
				cell_type: "markdown",
				source: ["---\n", 'title: "N"\n', "published: 2026-01-01\n", "---\n"],
			},
		]);
		const report = await generateNotebooks({
			postsDir: posts,
			assetsDir: assets,
		});
		assert.deepEqual(report.generated, ["sub/deep/n.md"]);
		const md = await fs.readFile(
			path.join(posts, "sub", "deep", "n.md"),
			"utf-8",
		);
		// 标记是唯一的来源凭据：必须在 frontmatter 块首行、用相对 postsDir 的 posix 路径，
		// 且整体落在 120 字节短读窗口内（撞名检测与孤儿清理都只读开头这一段）
		const lines = md.split("\n");
		assert.equal(lines[0], "---");
		assert.equal(lines[1], "# generated-from-notebook: sub/deep/n.ipynb");
		assert.ok(
			md.slice(0, 120).includes("# generated-from-notebook: sub/deep/n.ipynb"),
		);
		const ignoreLines = (
			await fs.readFile(path.join(posts, ".gitignore"), "utf-8")
		).split("\n");
		assert.deepEqual(
			ignoreLines.filter((line) => line.startsWith("/")),
			["/sub/deep/n.md"],
		);
		assert.ok(
			ignoreLines.every(
				(line) => line === "" || line.startsWith("#") || line.startsWith("/"),
			),
		);
	},
);

await checkAsync("手写 md 不会被当成孤儿清理", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await fs.writeFile(path.join(posts, "manual.md"), "# 我手写的");
	await writeNotebook(path.join(posts, "m.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "M"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(report.generated, ["m.md"]);
	assert.equal(
		await fs.readFile(path.join(posts, "manual.md"), "utf-8"),
		"# 我手写的",
	);
});

await checkAsync("正文提到标记的手写 md 不被删也不被覆写", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	// ① 首行不是 `---`，正文里提到标记 —— 旧判据（前 120 字节含该子串）会把它当孤儿删掉
	const manual =
		"手写标题\n\n本文的产物由 `# generated-from-notebook: note.ipynb` 识别。\n";
	await fs.writeFile(path.join(posts, "intro.md"), manual);
	// ② 首行是 `---`，但第二行不是标记行（frontmatter 里写了别的字段）
	const fenced =
		"---\ntitle: 手写\n---\n\n# generated-from-notebook: ghost.ipynb\n";
	await fs.writeFile(path.join(posts, "later.md"), fenced);
	await writeNotebook(path.join(posts, "n.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "N"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(report.generated, ["n.md"]);
	assert.equal(
		await fs.readFile(path.join(posts, "intro.md"), "utf-8"),
		manual,
	);
	assert.equal(
		await fs.readFile(path.join(posts, "later.md"), "utf-8"),
		fenced,
	);
});

await checkAsync("正文提到标记的手写 md 撞名时不报成功的覆写", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	const manual =
		"手写文章\n\n# generated-from-notebook: x.ipynb 就是本文要介绍的机制。\n";
	await fs.writeFile(path.join(posts, "x.md"), manual);
	await writeNotebook(path.join(posts, "x.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "X"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.equal(report.errors.length, 1);
	assert.match(report.errors[0].message, /已存在且不是生成的/);
	assert.deepEqual(report.generated, []);
	assert.equal(await fs.readFile(path.join(posts, "x.md"), "utf-8"), manual);
});

await checkAsync(
	"复刻两行头部的手写 md 会被当成产物（刻意接受的取舍）",
	async () => {
		const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
		const assets = path.join(posts, "..", "assets-out");
		// 判据锚定行位置，只能区分「复刻了 `---` + 标记行」与「没复刻」两种文件；
		// 与真实产物逐字节同形的文件无法区分，本就不该由用户手写
		await fs.writeFile(
			path.join(posts, "clone.md"),
			"---\n# generated-from-notebook: ghost.ipynb\ntitle: 手写\n---\n",
		);
		const report = await generateNotebooks({
			postsDir: posts,
			assetsDir: assets,
		});
		assert.deepEqual(report.generated, []);
		assert.deepEqual(await fs.readdir(posts), []);
	},
);

await checkAsync(
	"大小写不同名（落回同一 inode）的产物不被当成孤儿删除",
	async () => {
		const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
		const assets = path.join(posts, "..", "assets-out");
		await writeNotebook(path.join(posts, "a.ipynb"), [
			{
				cell_type: "markdown",
				source: ["---\n", 'title: "A"\n', "published: 2026-01-01\n", "---\n"],
			},
		]);
		await generateNotebooks({ postsDir: posts, assetsDir: assets });
		await fs.rename(path.join(posts, "a.ipynb"), path.join(posts, "A.ipynb"));
		const report = await generateNotebooks({
			postsDir: posts,
			assetsDir: assets,
		});
		assert.deepEqual(report.errors, []);
		assert.deepEqual(report.generated, ["A.md"]);
		// 大小写不敏感的卷（APFS）上产物仍叫 a.md，文件名与 report 对不上但来源一致
		// —— 两个平台上都只该留下一个 md，且标记指向改名后的 notebook
		const produced = (await fs.readdir(posts)).filter((file) =>
			file.endsWith(".md"),
		);
		assert.equal(produced.length, 1, `产物被误删：${produced.join(", ")}`);
		assert.match(
			await fs.readFile(path.join(posts, produced[0]), "utf-8"),
			/^---\n# generated-from-notebook: A\.ipynb\n/,
		);
	},
);

await checkAsync("用户自己的 .gitignore 不被删除也不被覆写", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	const mine = "*.tmp\n!keep.txt\n";
	await fs.writeFile(path.join(posts, ".gitignore"), mine);
	await writeNotebook(path.join(posts, "k.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "K"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(report.generated, ["k.md"]);
	assert.deepEqual(report.errors, []);
	assert.equal(report.warnings.length, 1);
	assert.match(report.warnings[0], /\.gitignore 不是本工具生成的/);
	assert.equal(
		await fs.readFile(path.join(posts, ".gitignore"), "utf-8"),
		mine,
	);
	// 产物为空（来源 notebook 被删）时同样不能顺手删掉用户文件
	await fs.rm(path.join(posts, "k.ipynb"));
	const second = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(second.generated, []);
	assert.equal(
		await fs.readFile(path.join(posts, ".gitignore"), "utf-8"),
		mine,
	);
	assert.match(second.warnings[0], /\.gitignore 不是本工具生成的/);
});

await checkAsync("文件名推导不出标题时报错并跳过，不产出坏文件", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "--.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.equal(report.errors.length, 1);
	assert.match(report.errors[0].message, /推导标题/);
	assert.deepEqual(report.generated, []);
	assert.deepEqual(await fs.readdir(posts), ["--.ipynb"]);
});

await checkAsync("键含冒号的 frontmatter 产出能被 Astro 解析", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "y.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'published: "2026-01-01"\n', '"a: b": 1\n', "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(report.errors, []);
	const generated = await fs.readFile(path.join(posts, "y.md"), "utf-8");
	const { frontmatter } = parseFrontmatter(generated);
	assert.equal(frontmatter["a: b"], 1);
	// 标记插在 frontmatter 内部第一行，含冒号的键照样解析得出来
	assert.equal(frontmatter.published instanceof Date, true);
});

// 145 字节的长非 ASCII 路径：标记行远超旧 120 字节窗口留给路径的 89 字节余量，
// 窗口一旦切在多字节字符中间，toString("utf-8") 就会在末尾补 U+FFFD
const LONG_NOTEBOOK =
	"数据分析/非常长的中文目录名称第一层/非常长的中文目录名称第二层/非常长的中文目录名称第三层/report.ipynb";

await checkAsync(
	"长非 ASCII 路径的产物不被当成孤儿删除（窗口必须装下整行标记）",
	async () => {
		const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
		const assets = path.join(posts, "..", "assets-out");
		await writeNotebook(path.join(posts, LONG_NOTEBOOK), [
			{
				cell_type: "markdown",
				source: [
					"---\n",
					'title: "长路径"\n',
					"published: 2026-01-01\n",
					"---\n",
				],
			},
		]);
		const report = await generateNotebooks({
			postsDir: posts,
			assetsDir: assets,
		});
		assert.deepEqual(report.errors, []);
		assert.deepEqual(report.generated, [
			LONG_NOTEBOOK.replace(/\.ipynb$/, ".md"),
		]);
		// report.generated 与磁盘必须一致：旧实现在这里声称生成成功、文件却已被删
		assert.deepEqual(
			(await fs.readdir(posts, { recursive: true })).filter((file) =>
				file.endsWith(".md"),
			),
			report.generated,
		);
		assert.match(
			await fs.readFile(path.join(posts, report.generated[0]), "utf-8"),
			/^---\n# generated-from-notebook: 数据分析\//,
		);
	},
);

await checkAsync("长路径 notebook 本轮失败时旧产物照旧被清掉", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, LONG_NOTEBOOK), [
		{
			cell_type: "markdown",
			source: [
				"---\n",
				'title: "长路径"\n',
				"published: 2026-01-01\n",
				"---\n",
			],
		},
	]);
	const first = await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal(first.generated.length, 1);
	// 同一路径的 notebook 变成坏 JSON：本轮报错，旧产物按既有语义清掉
	// （靠 failed.has(记录里的 notebook) —— 长路径下标记必须被完整读出来才命中）
	await fs.writeFile(path.join(posts, LONG_NOTEBOOK), "{ not json");
	const second = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.equal(second.errors.length, 1);
	assert.equal(second.errors[0].file, LONG_NOTEBOOK);
	assert.deepEqual(second.generated, []);
	assert.deepEqual(
		(await fs.readdir(posts, { recursive: true })).filter((file) =>
			file.endsWith(".md"),
		),
		[],
	);
});

await checkAsync("标记行读不完整时保守保留并告警，绝不删除", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	// 标记行比读取窗口还长：来源不明 → 一律保留（宁可留下陌生文件也不误删）
	const overlong = `---\n# generated-from-notebook: ${"a".repeat(5000)}.ipynb\ntitle: 手写\n---\n`;
	await fs.writeFile(path.join(posts, "overlong.md"), overlong);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(report.generated, []);
	assert.equal(
		await fs.readFile(path.join(posts, "overlong.md"), "utf-8"),
		overlong,
	);
	assert.equal(report.warnings.length, 1);
	assert.match(report.warnings[0], /超出 \d+ 字节读取窗口/);
});

await checkAsync("标记行读不完整的同名目标不被覆写", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	// 来源不明的目标对撞名保护也是「不确定」：按用户手写的处理，绝不覆写
	const overlong = `---\n# generated-from-notebook: ${"a".repeat(5000)}.ipynb\ntitle: 手写\n---\n`;
	await fs.writeFile(path.join(posts, "w.md"), overlong);
	await writeNotebook(path.join(posts, "w.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "W"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	const report = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(report.generated, []);
	assert.equal(report.errors.length, 1);
	assert.match(report.errors[0].message, /已存在且不是生成的/);
	assert.equal(await fs.readFile(path.join(posts, "w.md"), "utf-8"), overlong);
});

await checkAsync(".gitignore 内容不变时不重写", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "g.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "G"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	const ignorePath = path.join(posts, ".gitignore");
	const before = (await fs.stat(ignorePath)).mtimeMs;
	await delay(20);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	assert.equal((await fs.stat(ignorePath)).mtimeMs, before);
});

await checkAsync(
	"标记不破坏 frontmatter：Astro parseFrontmatter 能取到字段",
	async () => {
		const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
		const assets = path.join(posts, "..", "assets-out");
		await writeNotebook(path.join(posts, "pf.ipynb"), [
			{
				cell_type: "markdown",
				source: ["---\n", 'title: "PF"\n', "published: 2026-02-03\n", "---\n"],
			},
		]);
		await generateNotebooks({ postsDir: posts, assetsDir: assets });
		const md = await fs.readFile(path.join(posts, "pf.md"), "utf-8");
		const { frontmatter, content } = parseFrontmatter(md);
		assert.equal(frontmatter.title, "PF");
		assert.ok(frontmatter.published instanceof Date);
		assert.equal(
			(frontmatter.published as Date).toISOString().slice(0, 10),
			"2026-02-03",
		);
		// frontmatter 与其中的标记注释在渲染前被整体摘掉，不会漏进正文
		assert.doesNotMatch(content, /generated-from-notebook/);
	},
);

await checkAsync("生成的旧文件被认作同源，重跑不报撞名", async () => {
	const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-gen-"));
	const assets = path.join(posts, "..", "assets-out");
	await writeNotebook(path.join(posts, "r.ipynb"), [
		{
			cell_type: "markdown",
			source: ["---\n", 'title: "R"\n', "published: 2026-01-01\n", "---\n"],
		},
	]);
	await generateNotebooks({ postsDir: posts, assetsDir: assets });
	const second = await generateNotebooks({
		postsDir: posts,
		assetsDir: assets,
	});
	assert.deepEqual(second.errors, []);
	assert.deepEqual(second.generated, ["r.md"]);
});

section("collection");
await checkAsync(
	"posts 集合的 loader 接上生成器，生成物满足 schema 要求",
	async () => {
		const posts = await fs.mkdtemp(path.join(os.tmpdir(), "nb-col-"));
		const assets = path.join(posts, "..", "assets-out");
		await writeNotebook(path.join(posts, "e.ipynb"), [
			{
				cell_type: "markdown",
				source: ["---\n", 'title: "E"\n', "published: 2026-01-01\n", "---\n"],
			},
			{ cell_type: "markdown", source: ["正文一段。\n"] },
		]);

		// 绝对路径同样能解析（内部按 URL 相对根目录解析，glob 接受同一种写法）
		const loader = notebookPosts({ base: posts, assetsDir: assets });
		assert.equal(typeof loader.load, "function");

		await generateNotebooks({ postsDir: posts, assetsDir: assets });
		const generated = await fs.readFile(path.join(posts, "e.md"), "utf-8");
		// 标记必须是 frontmatter 块内部第一行的 YAML 注释：`---` 得在文件第 0
		// 字节，否则 Astro 的 frontmatter 正则匹配不到，posts schema 会校验失败
		assert.match(generated, /^---\n# generated-from-notebook: e\.ipynb\n/);
		// 集合要拿到 title/published，frontmatter 必须能被 Astro 解析成对象
		const { frontmatter } = parseFrontmatter(generated);
		assert.equal(frontmatter.title, "E");
		assert.ok(frontmatter.published instanceof Date);
	},
);

section("dev watcher");
await checkAsync(
	"每次 load 都重挂 watcher（全量 sync 会清掉监听），只理会 .ipynb，连发的变动合并成一轮生成",
	async () => {
		const root = await fs.mkdtemp(path.join(os.tmpdir(), "nb-watch-"));
		const posts = path.join(root, "posts");
		await fs.mkdir(posts, { recursive: true });
		// 最小假 watcher：只关心 loader 怎么挂监听、怎么反应，
		// 并像 Astro 的 listener-tracking Proxy 一样支持 removeAllTrackedListeners()
		const handlers: Record<string, Array<(file: string) => void>> = {};
		const watcher = {
			add: (): void => {},
			on: (event: string, handler: (file: string) => void): void => {
				handlers[event] = [...(handlers[event] ?? []), handler];
			},
			removeAllTrackedListeners: (): void => {
				for (const event of Object.keys(handlers)) handlers[event] = [];
			},
		};
		const emit = (event: string, file: string): void => {
			for (const handler of handlers[event] ?? []) handler(file);
		};
		const warnings: string[] = [];
		const context = {
			collection: "posts",
			store: { keys: (): string[] => [] },
			meta: {},
			logger: {
				warn: (message: string): void => void warnings.push(message),
				error: (): void => {},
			},
			config: { root: pathToFileURL(`${root}/`) },
			parseData: async (props: unknown): Promise<unknown> => props,
			generateDigest: (): string => "digest",
			watcher,
		} as unknown as LoaderContext;
		const loader = notebookPosts({
			base: posts,
			assetsDir: path.join(root, "assets"),
		});
		// 生成的轮数用「坏 notebook 每轮报一次错」来数：告警条数 == 生成轮数
		const runs = (notebook: string): number =>
			warnings.filter((message) => message.startsWith(`${notebook}:`)).length;
		const badCells = [
			{ cell_type: "markdown", source: ["没有 frontmatter 的正文\n"] },
		];
		await writeNotebook(path.join(posts, "seed.ipynb"), badCells);

		// load 每轮内容同步都会跑，每次都重挂（一次也不挂或只挂一次都会在下面露馅）
		await loader.load(context);
		await loader.load(context);
		await loader.load(context);
		assert.equal(handlers.change?.length, 3);
		assert.equal(handlers.add?.length, 3);
		assert.equal(handlers.unlink?.length, 3);
		assert.equal(runs("seed.ipynb"), 3, "每轮 load 应各生成一次");
		assert.deepEqual(await fs.readdir(posts), ["seed.ipynb"]);

		// 非 .ipynb 的变动不该触发重新生成
		await fs.writeFile(path.join(posts, "notes.txt"), "x");
		emit("change", path.join(posts, "notes.txt"));
		await delay(300);
		assert.equal(runs("seed.ipynb"), 3, "非 .ipynb 的变动不该触发生成");
		assert.deepEqual((await fs.readdir(posts)).sort(), [
			"notes.txt",
			"seed.ipynb",
		]);

		// 连发三次变动应合并成一轮生成（与大小写无关地识别扩展名）
		const bad = path.join(posts, "bad.ipynb");
		await writeNotebook(bad, badCells);
		emit("add", bad);
		await delay(20);
		emit("change", path.join(posts, "bad.IPYNB"));
		await delay(20);
		emit("change", bad);
		await delay(600);
		assert.equal(
			runs("bad.ipynb"),
			1,
			`连发三次变动应合并成一轮生成，实际 ${runs("bad.ipynb")} 轮：${warnings.join(" | ")}`,
		);

		// 全量 sync（改 content.config.ts）会 removeAllTrackedListeners() 把监听全 off，
		// 但 watcher 实例不变：这时只有「每次 load 都重挂」能让 .ipynb 监听活过来
		watcher.removeAllTrackedListeners();
		assert.equal(handlers.add?.length, 0);
		await loader.load(context);
		assert.equal(handlers.add?.length, 1, "全量 sync 后必须重挂监听");
		const after = path.join(posts, "after.ipynb");
		await writeNotebook(after, badCells);
		emit("add", after);
		await delay(300);
		assert.equal(
			runs("after.ipynb"),
			1,
			"全量 sync 后的 notebook 变动必须照旧生效",
		);
		await fs.rm(root, { recursive: true, force: true });
	},
);

console.log(failures === 0 ? "\n全部通过" : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
