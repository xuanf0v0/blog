/** 围栏长度取「内容里最长反引号串 + 1」并与 3 取大，防止内容里的 ``` 提前闭合代码块 */
export function fenced(text: string, language: string): string {
	const body = text.replace(/\n+$/, "");
	let longest = 0;
	// 不能对 `match(...) ?? []` 的联合直接 reduce：回调参数会被推断为 never
	for (const run of body.match(/`+/g) ?? []) {
		longest = Math.max(longest, run.length);
	}
	const fence = "`".repeat(Math.max(3, longest + 1));
	return `${fence}${language}\n${body}\n${fence}`;
}
