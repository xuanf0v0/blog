import { fileURLToPath } from "node:url";
import type { Loader } from "astro/loaders";
import { glob } from "astro/loaders";
import { generateNotebooks } from "../utils/notebook/generate";

export interface NotebookPostsOptions {
	/** 相对项目根目录的文章目录 */
	base?: string;
	/** 相对项目根目录的 notebook 图片目录 */
	assetsDir?: string;
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
		},
	};
}
