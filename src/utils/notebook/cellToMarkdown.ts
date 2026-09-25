import { fenced } from "./fence";
import { type OutputContext, outputToMarkdown } from "./outputToMarkdown";
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
	const raw =
		typeof fromLanguageInfo === "string" ? fromLanguageInfo : fromKernelspec;
	if (typeof raw !== "string" || raw.trim() === "") return "python";
	const normalized = raw.trim().toLowerCase();
	return LANGUAGE_ALIASES[normalized] ?? normalized;
}

async function markdownCell(
	cell: NotebookCell,
	context: OutputContext,
): Promise<string> {
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

	const parts: string[] = [fenced(toText(cell.source), language)];
	for (const output of cell.outputs ?? []) {
		const rendered = await outputToMarkdown(output, context);
		if (rendered.trim() !== "") parts.push(rendered);
	}
	return parts.join("\n\n");
}
