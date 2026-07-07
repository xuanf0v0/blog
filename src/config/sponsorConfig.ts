import type { SponsorConfig } from "../types/sponsorConfig";

export const sponsorConfig: SponsorConfig = {
	// 页面标题，如果留空则使用 i18n 中的翻译
	title: "",

	// 页面描述文本，如果留空则使用 i18n 中的翻译
	description: "",

	// 打赏用途说明
	usage: "您的支持是我创作的动力。",

	// 是否显示打赏者列表
	showSponsorsList: true,

	// 是否显示评论区，需要先在commentConfig.ts启用评论系统
	showComment: true,

	// 是否在文章详情页底部显示打赏按钮
	showButtonInPost: true,

	// 打赏方式列表
	methods: [
		// 在这里添加你的打赏方式，示例：
		// {
		// 	name: "支付宝",
		// 	icon: "fa7-brands:alipay",
		// 	qrCode: "/assets/images/sponsor/alipay.png",
		// 	link: "",
		// 	description: "使用 支付宝 扫码打赏",
		// 	enabled: true,
		// },
	],

	// 打赏者列表（可选）
	sponsors: [
		// 在这里添加打赏者，示例：
		// {
		// 	name: "用户昵称",
		// 	avatar: "https://example.com/avatar.jpg",
		// 	amount: "¥50",
		// 	date: "2025-10-01",
		// },
	],
};
