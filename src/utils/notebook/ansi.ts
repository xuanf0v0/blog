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

/**
 * 终端转义序列。第一个分支捕获 SGR 参数（走颜色逻辑），其余分支整体丢弃：
 * CSI（`ESC [ 参数 终止字节`）、OSC（`ESC ] … BEL 或 ESC \`）、其它 ESC 序列、尾部孤立的 ESC。
 */
const ESCAPE_PATTERN =
	// biome-ignore lint/suspicious/noControlCharactersInRegex: ESC (0x1b) introduces terminal escape sequences
	/\u001b\[([0-9;]*)m|\u001b\[[0-?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)?|\u001b[ -/]*[0-~]|\u001b/g;

interface SgrState {
	fg: string | null;
	bg: string | null;
	bold: boolean;
	dim: boolean;
	italic: boolean;
	underline: boolean;
}

/**
 * 文本是否需要 ANSI 清洗：含任何 ESC（终端控制序列，含非 SGR）或裸回车时返回 true。
 * 消费方据此决定交给 {@link ansiToHtml} 还是直接放进代码块。
 */
export function hasAnsi(text: string): boolean {
	return text.includes("\u001b") || text.includes("\r");
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

/** 合法的颜色通道值：存在且落在 0..255 */
function isByte(value: number | undefined): value is number {
	return value !== undefined && value >= 0 && value <= 255;
}

interface ColorSpec {
	/** 校验通过的颜色；参数不合法时为 null，即该颜色请求被忽略 */
	color: string | null;
	/** 该颜色规格消耗的参数个数（含 38/48 自身之后的 1 或 3 个） */
	length: number;
}

/** 读取 `38`/`48` 之后的颜色规格；参数缺失或越界时返回 `color: null`，但照常消耗参数以免错位 */
function readColor(codes: number[], index: number): ColorSpec | null {
	const mode = codes[index + 1];
	if (mode === 5) {
		const value = codes[index + 2];
		return { color: isByte(value) ? xterm256(value) : null, length: 2 };
	}
	if (mode === 2) {
		const red = codes[index + 2];
		const green = codes[index + 3];
		const blue = codes[index + 4];
		return {
			color:
				isByte(red) && isByte(green) && isByte(blue)
					? toHex(red, green, blue)
					: null,
			length: 4,
		};
	}
	return null;
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
		} else if (code === 38 || code === 48) {
			const spec = readColor(codes, index);
			if (spec !== null) {
				if (spec.color !== null) {
					if (code === 38) {
						next.fg = spec.color;
					} else {
						next.bg = spec.color;
					}
				}
				index += spec.length;
			}
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
	const source = text.replace(/\r\n?/g, "\n");
	ESCAPE_PATTERN.lastIndex = 0;
	let match = ESCAPE_PATTERN.exec(source);
	while (match !== null) {
		pushChunk(out, source.slice(cursor, match.index), state);
		if (match[1] !== undefined) state = applySgr(state, match[1]);
		cursor = match.index + match[0].length;
		match = ESCAPE_PATTERN.exec(source);
	}
	pushChunk(out, source.slice(cursor), state);
	return out.join("");
}
