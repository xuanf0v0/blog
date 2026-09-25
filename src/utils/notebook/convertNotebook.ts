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

function serializeFrontmatter(frontmatter: Record<string, unknown>): string {
	const lines = ["---"];
	for (const [key, value] of Object.entries(frontmatter)) {
		// 空值写出来只会污染 YAML，Astro 的 schema 本身也会给默认值
		if (value === undefined || value === null || value === "") continue;
		lines.push(`${key}: ${serializeValue(key, value)}`);
	}
	lines.push("---");
	return lines.join("\n");
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

	const context: OutputContext = {
		resolveAsset: async (base64, extension) => {
			const asset = await writeAsset({
				base64,
				extension,
				directory: options.assetsDir,
			});
			// 图片与生成的 md 不在同一目录，统一用 posix 分隔符写相对路径
			return path
				.relative(targetDir, asset.absolutePath)
				.split(path.sep)
				.join("/");
		},
		resolveAttachment: async (name, data) => {
			const entry = Object.entries(data)[0];
			if (entry === undefined) return `attachment:${name}`;
			const [mime, base64] = entry;
			const subtype = mime.split("/")[1] ?? "bin";
			const asset = await writeAsset({
				base64,
				extension: subtype.split("+")[0] ?? "bin",
				directory: options.assetsDir,
			});
			return path
				.relative(targetDir, asset.absolutePath)
				.split(path.sep)
				.join("/");
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
	return {
		frontmatter,
		markdown: body === "" ? `${header}\n` : `${header}\n\n${body}\n`,
	};
}
