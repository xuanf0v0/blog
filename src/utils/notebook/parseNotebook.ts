import fs from "node:fs/promises";
import type { NotebookCell, NotebookDocument } from "./types";

export class NotebookError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "NotebookError";
	}
}

export function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export async function parseNotebook(
	filePath: string,
): Promise<NotebookDocument> {
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
	const cells = parsed.cells.filter((cell): cell is NotebookCell =>
		isRecord(cell),
	);

	return {
		nbformat: parsed.nbformat,
		metadata: isRecord(parsed.metadata) ? parsed.metadata : {},
		cells,
	};
}
