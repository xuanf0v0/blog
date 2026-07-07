import type { FriendLink, FriendsPageConfig } from "../types/friendsConfig";

// 友链配置
// 可以在 src/content/spec/friends.md 中编写友链页面下方的自定义内容。
// 保留页面和组件，但清空原作者示例数据；之后你把自己的友链加到 friendsConfig 数组即可。

export const friendsPageConfig: FriendsPageConfig = {
	title: "友链",
	description: "这里会展示我的朋友和常用站点。",
	showCustomContent: true,
	showComment: true,
	randomizeSort: false,
};

export const friendsConfig: FriendLink[] = [
	// 示例占位，启用后会显示在友链页：
	// {
	// 	title: "站点名称",
	// 	imgurl: "https://example.com/avatar.png",
	// 	desc: "一句话介绍这个站点。",
	// 	siteurl: "https://example.com",
	// 	tags: ["博客", "朋友"],
	// 	weight: 10,
	// 	enabled: true,
	// },
];

export const getEnabledFriends = (): FriendLink[] => {
	const friends = friendsConfig.filter((friend) => friend.enabled);
	if (friendsPageConfig.randomizeSort) {
		return [...friends].sort(() => Math.random() - 0.5);
	}
	return [...friends].sort((a, b) => b.weight - a.weight);
};
