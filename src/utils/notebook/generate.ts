import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";
import { convertNotebook } from "./convertNotebook";
import { NotebookError } from "./parseNotebook";

export interface GenerateReport {
	/** 本轮成功生成的 md，相对 postsDir 的 posix 路径 */
	generated: string[];
	errors: Array<{ file: string; message: string }>;
}

/**
 * 生成标记：YAML 注释，插在 frontmatter 块内部第一行。
 * 不用 HTML 注释配 `---` 在前的写法 —— Astro 的 frontmatter 正则
 * `/(?:^\uFEFF?|^\s*\n)(?:---|\+\+\+)/` 只容许一个纯空白行作前缀，
 * 首行是 HTML 注释时 frontmatter 会被解析成 `{}`，导致 schema 校验失败。
 * 放在 frontmatter 内部则 `---` 仍在文件第 0 字节，且标记仍落在开头 60 字节内，
 * 供下面的 120 字节短读识别。
 */
const MARKER = "# generated-from-notebook:";
const YAML_FENCE = "---\n";
const HEAD_LENGTH = 120;
const IGNORE_HEADER = [
	"# 自动生成，请勿手改 —— 由 src/utils/notebook/generate.ts 维护",
	"# 下列文件由 notebook 生成，不入库",
].join("\n");

/** 文件不存在返回 null（不是「读到了空内容」），其余读取错误照常抛出 */
async function readHead(filePath: string): Promise<string | null> {
	try {
		const handle = await fs.open(filePath, "r");
		try {
			const buffer = Buffer.alloc(HEAD_LENGTH);
			const { bytesRead } = await handle.read(buffer, 0, HEAD_LENGTH, 0);
			return buffer.subarray(0, bytesRead).toString("utf-8");
		} finally {
			await handle.close();
		}
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
		throw error;
	}
}

/** 找出目录下所有由生成器产出的 md（靠首行标记识别，不依赖任何清单） */
async function findGeneratedFiles(postsDir: string): Promise<string[]> {
	const files = await glob("**/*.md", { cwd: postsDir, posix: true });
	const generated: string[] = [];
	for (const file of files) {
		const head = await readHead(path.join(postsDir, file));
		if (head?.includes(MARKER)) generated.push(file);
	}
	return generated.sort();
}

/** 维护 postsDir/.gitignore：只登记生成的 md，产物为空时连文件一起删掉 */
async function writeGitignore(
	postsDir: string,
	generated: string[],
): Promise<void> {
	const filePath = path.join(postsDir, ".gitignore");
	if (generated.length === 0) {
		await fs.rm(filePath, { force: true });
		return;
	}
	const content = `${IGNORE_HEADER}\n${generated
		.map((file) => `/${file}`)
		.join("\n")}\n`;
	const existing = await fs.readFile(filePath, "utf-8").catch(() => null);
	if (existing !== content) {
		await fs.writeFile(filePath, content);
	}
}

/**
 * 把 postsDir 下所有 notebook 转成同目录同名的 md。
 * 单个 notebook 失败只记录到 errors，不影响其余文件。
 */
export async function generateNotebooks(options: {
	postsDir: string;
	assetsDir: string;
}): Promise<GenerateReport> {
	const notebooks = (
		await glob("**/*.ipynb", { cwd: options.postsDir, posix: true })
	).sort();
	const report: GenerateReport = { generated: [], errors: [] };

	for (const notebook of notebooks) {
		const target = notebook.replace(/\.ipynb$/i, ".md");
		if (path.posix.basename(notebook).startsWith("_")) {
			report.errors.push({
				file: notebook,
				message:
					"文件名以下划线开头：content 目录下 `_` 前缀是保留语义，请重命名",
			});
			continue;
		}

		const targetPath = path.join(options.postsDir, target);
		try {
			// 撞名保护：目标已存在且不带标记 → 是用户手写的，绝不覆写
			const targetHead = await readHead(targetPath);
			if (targetHead !== null && !targetHead.includes(MARKER)) {
				report.errors.push({
					file: notebook,
					message: `目标文件 ${target} 已存在且不是生成的，请重命名 notebook 或删除该文件`,
				});
				continue;
			}

			const converted = await convertNotebook({
				ipynbPath: path.join(options.postsDir, notebook),
				targetPath,
				assetsDir: path.join(
					options.assetsDir,
					notebook.replace(/\.ipynb$/i, ""),
				),
			});
			// 标记插进 frontmatter 块内部第一行（`---` 之后），既保证 `---`
			// 仍在文件第 0 字节，又让标记落在开头若干字节内供短读识别。
			// 缺 frontmatter 是不该出现的状态，明确报错而不是静默产出坏文件。
			if (!converted.markdown.startsWith(YAML_FENCE)) {
				throw new NotebookError(
					"转换结果缺少 YAML frontmatter，无法插入生成标记",
				);
			}
			const marked = `${YAML_FENCE}${MARKER} ${notebook}\n${converted.markdown.slice(
				YAML_FENCE.length,
			)}`;
			const content = `${marked.replace(/\n+$/, "")}\n`;
			// 逐字节比较后再写：内容没变就不动 mtime，避免 Vite 热重载循环
			const existing = await fs.readFile(targetPath, "utf-8").catch(() => null);
			if (existing !== content) {
				await fs.mkdir(path.dirname(targetPath), { recursive: true });
				await fs.writeFile(targetPath, content);
			}
			report.generated.push(target);
		} catch (error) {
			report.errors.push({
				file: notebook,
				message:
					error instanceof NotebookError
						? error.message
						: (error as Error).message,
			});
		}
	}

	// 孤儿清理：带标记但本轮没生成 → 来源 notebook 已被删除或改名
	const stale = (await findGeneratedFiles(options.postsDir)).filter(
		(file) => !report.generated.includes(file),
	);
	for (const file of stale) {
		await fs.rm(path.join(options.postsDir, file), { force: true });
	}

	await writeGitignore(options.postsDir, report.generated);
	return report;
}
