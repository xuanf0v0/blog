import { fileURLToPath } from "node:url";
import type { Loader, LoaderContext } from "astro/loaders";
import { glob } from "astro/loaders";
import { generateNotebooks } from "../utils/notebook/generate";

export interface NotebookPostsOptions {
	/** 相对项目根目录的文章目录 */
	base?: string;
	/** 相对项目根目录的 notebook 图片目录 */
	assetsDir?: string;
}

/** 一次保存常触发多个事件（同名 md 的 change/add、编辑器的临时文件），合并成一轮 */
const REGENERATE_DEBOUNCE_MS = 150;

/** 已经挂过监听的 watcher：load 每轮同步都会被调用，靠引用比对只挂一次 */
let watched: NonNullable<LoaderContext["watcher"]> | null = null;
/** 最近一次同步的 context：watcher 回调里没有参数，只能借它打日志 */
let latestContext: LoaderContext | null = null;
/** 防抖定时器 */
let pendingTimer: NodeJS.Timeout | undefined;

/**
 * 把一批 .ipynb 事件合并成一次重新生成：等 150ms 没有再来的事件才真正跑。
 * 日志借最近一次同步的 context —— watcher 回调本身不携带它。
 */
function scheduleRegeneration(postsDir: string, assetsDir: string): void {
	clearTimeout(pendingTimer);
	pendingTimer = setTimeout(() => {
		pendingTimer = undefined;
		const context = latestContext;
		if (context === null) return;
		void generateNotebooks({ postsDir, assetsDir }).then(
			(report) => {
				for (const error of report.errors) {
					context.logger.warn(`${error.file}: ${error.message}`);
				}
			},
			(error: Error) => {
				context.logger.error(`notebook 重新生成失败：${error.message}`);
			},
		);
	}, REGENERATE_DEBOUNCE_MS);
}

/**
 * dev 下监听 notebook 变动并重新生成 md。
 * 不碰 store：新 md 由 glob() 自带的 watcher（change/add/unlink）接住。
 */
function registerWatcher(
	context: LoaderContext,
	postsDir: string,
	assetsDir: string,
): void {
	const watcher = context.watcher;
	// 构建时没有 watcher；dev 下 load 会反复执行，只认第一次那个实例
	if (watcher === undefined || watched === watcher) return;
	watched = watcher;

	const onNotebookEvent = (changedPath: string): void => {
		if (!changedPath.toLowerCase().endsWith(".ipynb")) return;
		scheduleRegeneration(postsDir, assetsDir);
	};
	watcher.on("change", onNotebookEvent);
	watcher.on("add", onNotebookEvent);
	watcher.on("unlink", onNotebookEvent);
}

/**
 * 把 posts 目录里的 .ipynb 先转成同目录同名的 .md，再交给官方 glob() 入库。
 *
 * 生成物与手写文章走完全相同的加载路径：id/slug 规则一致（路由靠
 * removeEntryId 去扩展名生成 URL），也不会因为「事后追加 entry」被 glob
 * 结尾的 stale 清理当成孤儿删掉。
 */
export function notebookPosts(options: NotebookPostsOptions = {}): Loader {
	const base = options.base ?? "./src/content/posts";
	const assets = options.assetsDir ?? "./src/assets/notebooks";
	const inner = glob({ pattern: "**/*.{md,mdx}", base });

	return {
		name: "blog:notebook-posts",
		load: async (context): Promise<void> => {
			// watcher 回调只能借最近一次同步的 context 打日志
			latestContext = context;
			// 与 glob() 内部同款解析：base/assetsDir 都相对项目根目录，
			// 绝对路径与 file:// URL 也能被 new URL 正确处理
			const postsDir = fileURLToPath(new URL(base, context.config.root));
			const assetsDir = fileURLToPath(new URL(assets, context.config.root));

			const report = await generateNotebooks({ postsDir, assetsDir });
			for (const error of report.errors) {
				// 单个 notebook 转换失败不该拖垮整站：告警后退化成「少一篇文章」
				context.logger.warn(`${error.file}: ${error.message}`);
			}

			await inner.load(context);
			registerWatcher(context, postsDir, assetsDir);
		},
	};
}
