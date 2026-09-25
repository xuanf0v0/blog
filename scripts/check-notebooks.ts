import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { ansiToHtml, hasAnsi } from "../src/utils/notebook/ansi";
import {
	NotebookError,
	parseNotebook,
} from "../src/utils/notebook/parseNotebook";
import { toText } from "../src/utils/notebook/text";

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

console.log(failures === 0 ? "\n全部通过" : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
