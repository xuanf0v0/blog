import path from "node:path";
import process from "node:process";
import { generateNotebooks } from "../src/utils/notebook/generate";

const root = process.cwd();

const report = await generateNotebooks({
	postsDir: path.join(root, "src/content/posts"),
	assetsDir: path.join(root, "src/assets/notebooks"),
});

for (const error of report.errors) {
	console.error(`[notebooks] ${error.file}: ${error.message}`);
}
console.log(
	`[notebooks] 生成 ${report.generated.length} 篇，${report.errors.length} 篇被跳过`,
);

process.exit(report.errors.length === 0 ? 0 : 1);
