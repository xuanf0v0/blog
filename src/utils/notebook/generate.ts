import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";
import { convertNotebook } from "./convertNotebook";
import { NotebookError } from "./parseNotebook";

export interface GenerateReport {
	/** 本轮成功生成的 md，相对 postsDir 的 posix 路径 */
	generated: string[];
	/** 单个 notebook 的失败：只跳过该文件，不中断其余生成 */
	errors: Array<{ file: string; message: string }>;
	/** 不影响生成的提示，由调用方打印 */
	warnings: string[];
}

/**
 * 生成标记：YAML 注释，插在 frontmatter 块内部第一行，记下产物的来源 notebook。
 *
 * 不用 HTML 注释配 `---` 在前的写法 —— Astro 的 frontmatter 正则
 * `/(?:^\uFEFF?|^\s*\n)(?:---|\+\+\+)/` 只容许一个纯空白行作前缀，
 * 首行是 HTML 注释时 frontmatter 会被解析成 `{}`，导致 schema 校验失败。
 * 放在 frontmatter 内部则 `---` 仍在文件第 0 字节，且标记整体落在开头 120
 * 字节内，供短读识别。
 *
 * 判据锚定行位置（首行 `---`、第二行恰为标记行），不是「开头若干字节里出现
 * 该字符串」：后者会让正文里介绍这个机制的手写文章被当成产物删除或覆写。
 */
const MARKER = "# generated-from-notebook:";
/** 标记行前缀：写入与识别共用，保证两侧判据永远一致 */
const MARKER_LINE = `${MARKER} `;
const YAML_FENCE = "---\n";
/** 短读窗口：覆盖 `---`、标记行与常见长度的 notebook 路径 */
const HEAD_LENGTH = 120;
const GITIGNORE_FILE = ".gitignore";
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

/** 产物标记里记录的所有权信息 */
interface Owner {
	/** 标记记录的来源 notebook，相对 postsDir 的 posix 路径 */
	notebook: string;
	/** 标记行没读到换行：路径很可能被短读窗口截断，只能按前缀比对 */
	truncated: boolean;
}

interface GeneratedFile extends Owner {
	/** 相对 postsDir 的 posix 路径 */
	file: string;
}

/**
 * 只承认「首行 `---`、第二行是标记行」的文件为生成物，返回标记记录的来源
 * notebook；不是生成物返回 null。判据锚定行位置，不做子串搜索：手写文章只要
 * 正文里提到该字样就会被误判（删除或覆写），锚定行位置后只有真的复刻了这两行
 * 头部的手写文件才会被当成产物 —— 这是刻意接受、且注释里写明了的取舍。
 */
function readOwner(head: string | null): Owner | null {
	if (head === null || !head.startsWith(`${YAML_FENCE}${MARKER_LINE}`)) {
		return null;
	}
	const rest = head.slice(YAML_FENCE.length + MARKER_LINE.length);
	const end = rest.indexOf("\n");
	return end === -1
		? { notebook: rest, truncated: true }
		: { notebook: rest.slice(0, end), truncated: false };
}

/** 找出目录下所有由生成器产出的 md（靠 frontmatter 首行标记识别，不依赖任何清单） */
async function findGeneratedFiles(postsDir: string): Promise<GeneratedFile[]> {
	const files = await glob("**/*.md", { cwd: postsDir, posix: true });
	const generated: GeneratedFile[] = [];
	for (const file of files) {
		const owner = readOwner(await readHead(path.join(postsDir, file)));
		if (owner !== null) generated.push({ file, ...owner });
	}
	return generated.sort((a, b) => a.file.localeCompare(b.file));
}

/**
 * 维护 postsDir/.gitignore：只登记生成的 md，产物为空时连文件一起删掉。
 * 只认带自己头部标记的文件 —— 用户自己写的 .gitignore 可能承载他自己的忽略
 * 规则，宁可不忽略产物，也不能把它删掉或整份覆写。
 */
async function writeGitignore(
	postsDir: string,
	generated: string[],
	warnings: string[],
): Promise<void> {
	const filePath = path.join(postsDir, GITIGNORE_FILE);
	const existing = await fs.readFile(filePath, "utf-8").catch(() => null);
	if (existing !== null && !existing.startsWith(IGNORE_HEADER)) {
		warnings.push(
			`${GITIGNORE_FILE} 不是本工具生成的，已原样保留：生成的 md 不会被忽略`,
		);
		return;
	}
	if (generated.length === 0) {
		await fs.rm(filePath, { force: true });
		return;
	}
	const content = `${IGNORE_HEADER}\n${generated
		.map((file) => `/${file}`)
		.join("\n")}\n`;
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
	const report: GenerateReport = { generated: [], errors: [], warnings: [] };

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
			// 撞名保护：目标已存在且不带生成标记 → 是用户手写的，绝不覆写
			const targetHead = await readHead(targetPath);
			if (targetHead !== null && readOwner(targetHead) === null) {
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
			const marked = `${YAML_FENCE}${MARKER_LINE}${notebook}\n${converted.markdown.slice(
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

	// 孤儿清理：产物标记里记的来源 notebook 已不在本轮集合里（被删除、改名或移走）。
	// 判据取标记内容而不是「目标名与磁盘文件名精确比较」：APFS 等大小写不敏感的卷上
	// `a.ipynb` 改名成 `A.ipynb` 后写入会落到原 `a.md` 的 inode（磁盘名仍是 a.md），
	// 按文件名比较会把仍属本轮的产物误判成孤儿删掉，还造成 macOS 与 Linux 行为分歧。
	const failed = new Set(report.errors.map((error) => error.file));
	const stale = (await findGeneratedFiles(options.postsDir)).filter((entry) => {
		// 标记行被短读窗口截断时只剩前缀，退化成前缀匹配
		const owned = entry.truncated
			? notebooks.some((notebook) => notebook.startsWith(entry.notebook))
			: notebooks.includes(entry.notebook);
		// 本轮失败的 notebook 留下的旧产物照旧清掉（既有语义，不在本次改动范围）
		return !owned || failed.has(entry.notebook);
	});
	for (const { file } of stale) {
		await fs.rm(path.join(options.postsDir, file), { force: true });
	}

	await writeGitignore(options.postsDir, report.generated, report.warnings);
	return report;
}
