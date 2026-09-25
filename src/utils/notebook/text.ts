/** 把 notebook 的 source / text 字段归一化为字符串；非字符串、非字符串数组的输入一律归一为 "" */
export function toText(source: unknown): string {
	if (typeof source === "string") return source;
	if (Array.isArray(source)) return source.join("");
	return "";
}
