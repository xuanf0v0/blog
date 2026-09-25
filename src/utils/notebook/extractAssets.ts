import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

export interface ExtractedAsset {
	absolutePath: string;
	fileName: string;
}

/**
 * 把 base64 图片落地到 directory，文件名取内容 sha256 前 16 位。
 * 内容相同 → 文件名相同 → 天然幂等，重复运行不会产生新文件也不会重写。
 */
export async function writeAsset(options: {
	base64: string;
	extension: string;
	directory: string;
}): Promise<ExtractedAsset> {
	const buffer = Buffer.from(options.base64.replace(/\s+/g, ""), "base64");
	const hash = crypto
		.createHash("sha256")
		.update(buffer)
		.digest("hex")
		.slice(0, 16);
	const fileName = `${hash}.${options.extension}`;
	const absolutePath = path.join(options.directory, fileName);

	await fs.mkdir(options.directory, { recursive: true });
	try {
		// 已存在即内容相同（文件名即内容摘要），跳过写入以免触发 Vite 热重载。
		await fs.access(absolutePath);
	} catch {
		await fs.writeFile(absolutePath, buffer);
	}

	return { absolutePath, fileName };
}
