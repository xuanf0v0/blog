import { visit } from "unist-util-visit";

/**
 * 支持 LaTeX 原生定界符：行内 \(...\)、块级 \[...\]。
 * 美元语法（$...$ / $$...$$ / ```math）仍由 remark-math 处理，二者互不干扰。
 *
 * \( 与 \[ 在 CommonMark 里是合法字符转义：解析阶段反斜杠就被吃掉了，
 * 文本节点里只剩括号——写公式时会静默降级成普通文字，公式内的 \, 还会被
 * 还原成逗号直接改坏内容。因此这里借 position 回到原始 markdown 上匹配，
 * 命中片段替换成与 remark-math 产物完全一致的 inlineMath / math 节点
 * （hName / hProperties / hChildren 走 mdast-util-to-hast 的通用通道），
 * 由 rehype-katex 按同一路径渲染。
 *
 * 只在「raw 切片按 CommonMark 转义规则还原后与文本节点 value 完全一致」
 * 的节点上动手；拿不准的节点原样跳过，避免误伤。
 */

/** CommonMark 可转义标点：`\X` 在文本值里只剩 X，用它把 raw 片段还原成文本值 */
const ESCAPED_PUNCTUATION = /\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g;

/** 行内 \(...\)；两端的 \( 均需自身未被转义 */
const INLINE_MATH = /(?<!\\)\\\(([\s\S]*?)(?<!\\)\\\)/g;

/** 块级 \[...\] */
const DISPLAY_MATH = /(?<!\\)\\\[([\s\S]*?)(?<!\\)\\\]/g;

function unescapePunctuation(raw) {
	return raw.replace(ESCAPED_PUNCTUATION, "$1");
}

/** 与 remark-math 的行内节点一致 */
function inlineMathNode(value) {
	return {
		type: "inlineMath",
		value,
		data: {
			hName: "code",
			hProperties: { className: ["language-math", "math-inline"] },
			hChildren: [{ type: "text", value }],
		},
	};
}

/** 与 remark-math 的块级节点一致（mathFlow 路径，rehype-katex 会拿走 <pre> 做 display） */
function displayMathNode(value) {
	return {
		type: "math",
		meta: null,
		value,
		data: {
			hName: "pre",
			hChildren: [
				{
					type: "element",
					tagName: "code",
					properties: { className: ["language-math", "math-display"] },
					children: [{ type: "text", value }],
				},
			],
		},
	};
}

/** 在 raw 上收集所有匹配；交叠时保留先出现的一段 */
function collectMatches(raw) {
	const matches = [];
	for (const [pattern, kind] of [
		[INLINE_MATH, "inline"],
		[DISPLAY_MATH, "display"],
	]) {
		pattern.lastIndex = 0;
		let match = pattern.exec(raw);
		while (match !== null) {
			// 空内容不算公式，保留原文
			if (match[1].trim()) {
				matches.push({
					start: match.index,
					end: match.index + match[0].length,
					value: match[1],
					kind,
				});
			}
			match = pattern.exec(raw);
		}
	}
	matches.sort((a, b) => a.start - b.start);

	const kept = [];
	let end = 0;
	for (const match of matches) {
		if (match.start >= end) {
			kept.push(match);
			end = match.end;
		}
	}
	return kept;
}

/** 把文本节点按匹配切成 文本 / inlineMath / math 节点序列；无匹配返回 null */
function splitTextNode(raw) {
	const matches = collectMatches(raw);
	if (matches.length === 0) return null;

	const nodes = [];
	let cursor = 0;
	for (const match of matches) {
		if (match.start > cursor) {
			nodes.push({
				type: "text",
				value: unescapePunctuation(raw.slice(cursor, match.start)),
			});
		}
		if (match.kind === "inline") {
			nodes.push(inlineMathNode(match.value));
		} else {
			// 与 mathFlow 一致：去掉首尾换行
			nodes.push(
				displayMathNode(match.value.replace(/^(\r?\n|\r)|(\r?\n|\r)$/g, "")),
			);
		}
		cursor = match.end;
	}
	if (cursor < raw.length) {
		nodes.push({ type: "text", value: unescapePunctuation(raw.slice(cursor)) });
	}
	return nodes;
}

export function remarkLatexDelimiters() {
	return (tree, file) => {
		const source = typeof file?.value === "string" ? file.value : "";
		if (!source) return;

		visit(tree, "text", (node, index, parent) => {
			if (!parent || typeof index !== "number") return;
			const start = node.position?.start?.offset;
			const end = node.position?.end?.offset;
			if (typeof start !== "number" || typeof end !== "number") return;

			const raw = source.slice(start, end);
			// raw 切片必须能完整还原成文本值，否则说明存在实体等未覆盖的还原规则，跳过
			if (unescapePunctuation(raw) !== node.value) return;

			const nodes = splitTextNode(raw);
			if (nodes) parent.children.splice(index, 1, ...nodes);
		});

		// 独占段落的块级公式提升到段落外，产物与 $$ 的顶层 katex-display 对齐
		visit(tree, "paragraph", (node, index, parent) => {
			if (!parent || typeof index !== "number") return;
			if (node.children.length === 1 && node.children[0].type === "math") {
				parent.children.splice(index, 1, node.children[0]);
			}
		});
	};
}
