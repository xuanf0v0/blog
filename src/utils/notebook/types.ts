/** notebook 的 source / text 字段既可能是整段字符串，也可能是行数组 */
export type NotebookSource = string | string[];

export interface NotebookOutput {
	output_type: string;
	name?: string;
	text?: NotebookSource;
	data?: Record<string, unknown>;
	ename?: string;
	evalue?: string;
	traceback?: string[];
}

export interface NotebookCell {
	cell_type: string;
	source?: NotebookSource;
	outputs?: NotebookOutput[];
	attachments?: Record<string, Record<string, string>>;
}

export interface NotebookDocument {
	nbformat: number;
	metadata: Record<string, unknown>;
	cells: NotebookCell[];
}
