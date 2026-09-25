import path from "node:path";
import { parseFrontmatter } from "@astrojs/markdown-remark";
import { cellToMarkdown, detectLanguage } from "./cellToMarkdown";
import { writeAsset } from "./extractAssets";
import type { OutputContext } from "./outputToMarkdown";
import { isRecord, NotebookError, parseNotebook } from "./parseNotebook";
import { resolveMetadata } from "./resolveMetadata";

export interface ConvertedNotebook {
	frontmatter: Record<string, unknown>;
	/** 含 YAML frontmatter 的完整 markdown 文本 */
	markdown: string;
}

export interface ConvertNotebookOptions {
	/** .ipynb 绝对路径 */
	ipynbPath: string;
	/** 即将生成的 .md 绝对路径，用于计算图片的相对引用 */
	targetPath: string;
	/** 该 notebook 专属的图片目录绝对路径 */
	assetsDir: string;
}

/** 只有这两个字段会被序列化成不加引号的 YAML 标量 */
const DATE_FIELDS: Record<string, true> = {
	published: true,
	updated: true,
};

function serializeValue(key: string, value: unknown): string {
	if (typeof value === "boolean" || typeof value === "number") {
		return String(value);
	}
	if (value instanceof Date) return value.toISOString().slice(0, 10);
	if (Array.isArray(value)) {
		return `[${value.map((item) => JSON.stringify(String(item))).join(", ")}]`;
	}
	const text = String(value);
	// 日期必须写成不加引号的标量：带引号时 Astro 的 z.date() 会拿到字符串而校验失败。
	// 其余字符串统一用 JSON.stringify 转义（与 YAML 双引号风格兼容：\、\"、\n）。
	if (Object.hasOwn(DATE_FIELDS, key) && /^\d{4}-\d{2}-\d{2}$/.test(text)) {
		return text;
	}
	return JSON.stringify(text);
}

/** 空值写出来只会污染 YAML，Astro 的 schema 本身也会给默认值 */
function frontmatterEntries(
	frontmatter: Record<string, unknown>,
): Array<[string, unknown]> {
	return Object.entries(frontmatter).filter(
		([, value]) => value !== undefined && value !== null && value !== "",
	);
}

function serializeFrontmatter(frontmatter: Record<string, unknown>): string {
	const lines = ["---"];
	for (const [key, value] of frontmatterEntries(frontmatter)) {
		// 键同样要转义：metadata.blog 里出现 `"a: b"` 这类键时不加引号会产出
		// `a: b: 1`，YAML 直接报 bad indentation，单篇坏产出就把内容同步打断
		lines.push(`${JSON.stringify(key)}: ${serializeValue(key, value)}`);
	}
	lines.push("---");
	return lines.join("\n");
}

/**
 * 写入前的往返校验：坏 frontmatter 会越过隔离带把整次内容同步（astro sync /
 * check）打断，必须在生成阶段就抛错 —— 单个 notebook 失败只该跳过该文件。
 * 字段既不丢也不多，Astro 后续解析拿到的就是这里序列化的那批键。
 */
function assertRoundTrips(
	frontmatter: Record<string, unknown>,
	markdown: string,
): void {
	let parsed: unknown;
	try {
		parsed = parseFrontmatter(markdown).frontmatter;
	} catch (error) {
		throw new NotebookError(
			`生成的 frontmatter 无法解析：${(error as Error).message}`,
		);
	}
	if (!isRecord(parsed)) {
		throw new NotebookError("生成的 frontmatter 不是对象");
	}
	const expected = frontmatterEntries(frontmatter)
		.map(([key]) => key)
		.sort();
	const actual = Object.keys(parsed).sort();
	if (expected.join("\n") !== actual.join("\n")) {
		throw new NotebookError(
			`frontmatter 往返校验失败：期望字段 ${expected.join("、")}，实际 ${actual.join("、")}`,
		);
	}
}

export async function convertNotebook(
	options: ConvertNotebookOptions,
): Promise<ConvertedNotebook> {
	const doc = await parseNotebook(options.ipynbPath);
	const { frontmatter, firstBodyCell, leadingMarkdown } = resolveMetadata(
		doc,
		path.basename(options.ipynbPath),
	);
	const language = detectLanguage(doc);
	const targetDir = path.dirname(options.targetPath);

	// 图片与生成的 md 不在同一目录，统一用 posix 分隔符写相对路径。
	// 图片与附件两个回调共用这段「落地 + 取相对路径」，不重复维护同一条路径规则。
	const writeAndResolve = async (
		base64: string,
		extension: string,
	): Promise<string> => {
		const asset = await writeAsset({
			base64,
			extension,
			directory: options.assetsDir,
		});
		return path
			.relative(targetDir, asset.absolutePath)
			.split(path.sep)
			.join("/");
	};

	const context: OutputContext = {
		resolveAsset: writeAndResolve,
		resolveAttachment: async (name, data) => {
			const entry = Object.entries(data)[0];
			if (entry === undefined) return `attachment:${name}`;
			const [mime, base64] = entry;
			const subtype = mime.split("/")[1] ?? "bin";
			return writeAndResolve(base64, subtype.split("+")[0] ?? "bin");
		},
	};

	const blocks: string[] = [];
	// 被 YAML 消费的首单元格里，frontmatter 之后的正文不能丢，排在所有单元格之前
	if (leadingMarkdown.trim() !== "") blocks.push(leadingMarkdown.trim());
	for (const cell of doc.cells.slice(firstBodyCell)) {
		const rendered = await cellToMarkdown(cell, language, context);
		if (rendered.trim() !== "") blocks.push(rendered.trim());
	}

	const header = serializeFrontmatter(frontmatter);
	const body = blocks.join("\n\n");
	const markdown = body === "" ? `${header}\n` : `${header}\n\n${body}\n`;
	assertRoundTrips(frontmatter, markdown);
	return { frontmatter, markdown };
}
