import { ansiToHtml, hasAnsi } from "./ansi";
import { fenced } from "./fence";
import { toText } from "./text";
import type { NotebookOutput } from "./types";

export interface OutputContext {
	/** 落地 base64 图片，返回 markdown 中可直接使用的路径 */
	resolveAsset(base64: string, extension: string): Promise<string>;
	/** 落地单元格附件，返回 markdown 中可直接使用的路径 */
	resolveAttachment(
		name: string,
		data: Record<string, string>,
	): Promise<string>;
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

/** 含 ANSI 时只能用原生 HTML，围栏代码块会把 span 当字面量转义 */
function textBlock(text: string): string {
	const body = text.replace(/\n+$/, "");
	if (hasAnsi(body)) {
		return `<pre class="nb-output">${ansiToHtml(body)}</pre>`;
	}
	return fenced(body, "text");
}

function pickByPriority(
	data: Record<string, unknown>,
): { mime: string; value: unknown } | null {
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
		const reference = await context.resolveAsset(
			toText(value as string | string[]),
			extension,
		);
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
		return output.name === "stderr"
			? `<div class="nb-stderr">\n${body}\n</div>`
			: body;
	}

	if (output.output_type === "error") {
		const header = `${output.ename ?? "Error"}: ${output.evalue ?? ""}`;
		const traceback = (output.traceback ?? []).join("\n");
		return textBlock(traceback === "" ? header : `${header}\n${traceback}`);
	}

	if (
		output.output_type === "execute_result" ||
		output.output_type === "display_data"
	) {
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
