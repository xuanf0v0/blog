import { parseFrontmatter } from "@astrojs/markdown-remark";
import { isRecord, NotebookError } from "./parseNotebook";
import { toText } from "./text";
import type { NotebookDocument } from "./types";

export interface MetadataResult {
	frontmatter: Record<string, unknown>;
	/** 正文从第几个单元格开始（首单元格被 YAML 消费时为 1） */
	firstBodyCell: number;
	/** 被消费的首单元格里 frontmatter 块之后的剩余正文；无 frontmatter 单元格时为 "" */
	leadingMarkdown: string;
}

const STRING_FIELDS: Record<string, true> = {
	title: true,
	description: true,
	image: true,
	category: true,
	lang: true,
	author: true,
	sourceLink: true,
	licenseName: true,
	licenseUrl: true,
	password: true,
	passwordHint: true,
};

const DATE_FIELDS: Record<string, true> = {
	published: true,
	updated: true,
};

const BOOLEAN_FIELDS: Record<string, true> = {
	draft: true,
	pinned: true,
	comment: true,
};

/** YAML 里未加引号的日期会被当成 Date，字符串则原样保留 */
function normalizeDate(field: string, value: unknown): string {
	const text =
		value instanceof Date
			? value.toISOString().slice(0, 10)
			: typeof value === "string"
				? value.trim()
				: "";
	if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) {
		throw new NotebookError(
			`字段 ${field} 不是合法的 YYYY-MM-DD 日期：${String(value)}`,
		);
	}
	return text;
}

function titleFromFileName(fileName: string): string {
	return fileName
		.replace(/\.ipynb$/i, "")
		.replace(/[-_]+/g, " ")
		.trim();
}

function validate(
	frontmatter: Record<string, unknown>,
): Record<string, unknown> {
	const output: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(frontmatter)) {
		if (value === undefined || value === null) continue;
		if (Object.hasOwn(STRING_FIELDS, key)) {
			if (typeof value !== "string") {
				throw new NotebookError(
					`字段 ${key} 必须是字符串，实际是 ${typeof value}`,
				);
			}
			output[key] = value;
		} else if (Object.hasOwn(DATE_FIELDS, key)) {
			output[key] = normalizeDate(key, value);
		} else if (Object.hasOwn(BOOLEAN_FIELDS, key)) {
			if (typeof value !== "boolean") {
				throw new NotebookError(
					`字段 ${key} 必须是布尔值，实际是 ${typeof value}`,
				);
			}
			output[key] = value;
		} else if (key === "tags") {
			if (
				!Array.isArray(value) ||
				value.some((item) => typeof item !== "string")
			) {
				throw new NotebookError("字段 tags 必须是字符串数组");
			}
			output[key] = value;
		} else {
			output[key] = value;
		}
	}
	return output;
}

export function resolveMetadata(
	doc: NotebookDocument,
	fileName: string,
): MetadataResult {
	let fromCell: Record<string, unknown> = {};
	let firstBodyCell = 0;
	let leadingMarkdown = "";

	const first = doc.cells[0];
	if (first && first.cell_type === "markdown") {
		const source = toText(first.source);
		if (source.startsWith("---")) {
			try {
				const { frontmatter, content } = parseFrontmatter(source);
				if (isRecord(frontmatter) && Object.keys(frontmatter).length > 0) {
					fromCell = frontmatter;
					firstBodyCell = 1;
					leadingMarkdown = content.trim();
				}
			} catch (error) {
				throw new NotebookError(
					`首单元格 YAML 解析失败：${(error as Error).message}`,
				);
			}
		}
	}

	const blog = doc.metadata.blog;
	const fromNotebook = isRecord(blog) ? blog : {};
	const merged = { ...fromNotebook, ...fromCell };

	if (merged.published === undefined || merged.published === null) {
		throw new NotebookError(
			"缺少 published 字段：请在首单元格 YAML 或 metadata.blog 中显式声明（YYYY-MM-DD）",
		);
	}

	const validated = validate(merged);
	if (typeof validated.title !== "string" || validated.title.trim() === "") {
		validated.title = titleFromFileName(fileName);
	}

	return { frontmatter: validated, firstBodyCell, leadingMarkdown };
}
