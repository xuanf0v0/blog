/** xterm 标准 16 色 */
const BASIC_COLORS: string[] = [
	"#000000",
	"#cd0000",
	"#00cd00",
	"#cdcd00",
	"#0000ee",
	"#cd00cd",
	"#00cdcd",
	"#e5e5e5",
	"#7f7f7f",
	"#ff0000",
	"#00ff00",
	"#ffff00",
	"#5c5cff",
	"#ff00ff",
	"#00ffff",
	"#ffffff",
];

// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC (0x1b) introduces ANSI SGR sequences
const SGR_PATTERN = /\u001b\[([0-9;]*)m/g;

interface SgrState {
	fg: string | null;
	bg: string | null;
	bold: boolean;
	dim: boolean;
	italic: boolean;
	underline: boolean;
}

export function hasAnsi(text: string): boolean {
	// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC (0x1b) introduces ANSI SGR sequences
	return /\u001b\[[0-9;]*m/.test(text);
}

function emptyState(): SgrState {
	return {
		fg: null,
		bg: null,
		bold: false,
		dim: false,
		italic: false,
		underline: false,
	};
}

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;");
}

function toHex(red: number, green: number, blue: number): string {
	const part = (value: number): string => value.toString(16).padStart(2, "0");
	return `#${part(red)}${part(green)}${part(blue)}`;
}

function xterm256(index: number): string {
	if (index < 16) return BASIC_COLORS[index] ?? "#000000";
	if (index >= 232) {
		const grey = 8 + (index - 232) * 10;
		return toHex(grey, grey, grey);
	}
	const offset = index - 16;
	const level = (value: number): number => (value === 0 ? 0 : 55 + value * 40);
	return toHex(
		level(Math.floor(offset / 36)),
		level(Math.floor((offset % 36) / 6)),
		level(offset % 6),
	);
}

function applySgr(state: SgrState, params: string): SgrState {
	const codes = params
		.split(";")
		.map((part) => (part === "" ? 0 : Number.parseInt(part, 10)));
	const next: SgrState = { ...state };
	for (let index = 0; index < codes.length; index += 1) {
		const code = codes[index];
		if (code === 0) {
			Object.assign(next, emptyState());
		} else if (code === 1) {
			next.bold = true;
		} else if (code === 2) {
			next.dim = true;
		} else if (code === 3) {
			next.italic = true;
		} else if (code === 4) {
			next.underline = true;
		} else if (code === 22) {
			next.bold = false;
			next.dim = false;
		} else if (code === 23) {
			next.italic = false;
		} else if (code === 24) {
			next.underline = false;
		} else if (code >= 30 && code <= 37) {
			next.fg = BASIC_COLORS[code - 30];
		} else if (code === 39) {
			next.fg = null;
		} else if (code >= 40 && code <= 47) {
			next.bg = BASIC_COLORS[code - 40];
		} else if (code === 49) {
			next.bg = null;
		} else if (code >= 90 && code <= 97) {
			next.fg = BASIC_COLORS[code - 90 + 8];
		} else if (code >= 100 && code <= 107) {
			next.bg = BASIC_COLORS[code - 100 + 8];
		} else if (code === 38 && codes[index + 1] === 5) {
			next.fg = xterm256(codes[index + 2]);
			index += 2;
		} else if (code === 48 && codes[index + 1] === 5) {
			next.bg = xterm256(codes[index + 2]);
			index += 2;
		} else if (code === 38 && codes[index + 1] === 2) {
			next.fg = toHex(codes[index + 2], codes[index + 3], codes[index + 4]);
			index += 4;
		} else if (code === 48 && codes[index + 1] === 2) {
			next.bg = toHex(codes[index + 2], codes[index + 3], codes[index + 4]);
			index += 4;
		}
	}
	return next;
}

function styleToCss(state: SgrState): string {
	const parts: string[] = [];
	if (state.fg) parts.push(`color:${state.fg}`);
	if (state.bg) parts.push(`background-color:${state.bg}`);
	if (state.bold) parts.push("font-weight:600");
	if (state.dim) parts.push("opacity:0.7");
	if (state.italic) parts.push("font-style:italic");
	if (state.underline) parts.push("text-decoration:underline");
	return parts.join(";");
}

function pushChunk(out: string[], chunk: string, state: SgrState): void {
	if (chunk === "") return;
	const css = styleToCss(state);
	out.push(
		css === ""
			? escapeHtml(chunk)
			: `<span style="${css}">${escapeHtml(chunk)}</span>`,
	);
}

export function ansiToHtml(text: string): string {
	const out: string[] = [];
	let state = emptyState();
	let cursor = 0;
	SGR_PATTERN.lastIndex = 0;
	let match = SGR_PATTERN.exec(text);
	while (match !== null) {
		pushChunk(out, text.slice(cursor, match.index), state);
		state = applySgr(state, match[1]);
		cursor = match.index + match[0].length;
		match = SGR_PATTERN.exec(text);
	}
	pushChunk(out, text.slice(cursor), state);
	return out.join("");
}
