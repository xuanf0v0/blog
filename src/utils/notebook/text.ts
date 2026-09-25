import type { NotebookSource } from "./types";

/** 把 notebook 的 source / text 字段归一化为字符串 */
export function toText(source: NotebookSource | null | undefined): string {
	if (source === null || source === undefined) return "";
	return Array.isArray(source) ? source.join("") : source;
}
