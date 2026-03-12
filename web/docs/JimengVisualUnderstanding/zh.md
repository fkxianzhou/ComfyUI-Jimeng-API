### Jimeng Visual Understanding
使用 Seed 2.0 模型进行视觉理解对话，支持图片/视频输入和多轮对话。

*   **Node ID**: `JimengVisualUnderstanding`
*   **Python Class**: `JimengVisualUnderstanding`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 是 | - | 连接 `Jimeng API Client` 的输出。 |
| `model` | `COMBO` | 是 | - | 选择要使用的视觉理解模型。 |
| `system_prompt` | `STRING` | 是 | "You are a helpful assistant." | 系统级别的指令，用于定义助手的行为、角色和回答风格。System 角色指令优先级通常高于 User 角色。 |
| `user_prompt` | `STRING` | 是 | "Describe this content." | 用户的输入文本，描述您想了解的内容或向模型提出的问题 (User 角色)。 |
| `detail` | `COMBO` | 是 | "high" | 图片理解的精细度。'low': 低精度模式，消耗 Token 较少；'high': 高精度模式，提供更详细的分析。 |
| `fps` | `FLOAT` | 是 | 1.0 | 上传视频时采样的帧率。模型会根据此帧率从视频中抽取画面进行理解。 |
| `reasoning_mode` | `COMBO` | 是 | "auto" | 控制模型是否开启深度思考模式。'auto': 自动判断是否需要思考；'enabled': 强制开启思考；'disabled': 关闭思考，直接回答。 |
| `reasoning_effort` | `COMBO` | 是 | "medium" | 限制深度思考的工作量。'medium': 均衡模式(兼顾速度与深度)；'minimal': 关闭思考(直接回答)；'low': 轻量思考(侧重快速响应)；'high': 深度分析(处理复杂问题)。 |
| `turns` | `INT` | 是 | 1 | 多轮对话控制。设置为 1 表示开始新对话；大于 1 时，节点会尝试使用上一轮的 ID 维持上下文，实现连续对话。 |
| `stream` | `BOOLEAN` | 是 | False | 是否启用流式返回。开启后，将在 ComfyUI 控制台实时打印生成的文本和推理内容，并在生成完成后一次性输出结果。 |
| `seed` | `INT` | 是 | 0 | 随机种子，用于控制生成结果的确定性。 |
| `visual_input_1` | `IMAGE/VIDEO` | 否 | - | (可选) 输入模型的图片或视频文件。支持多模态理解。 |
| `visual_input_2` | `IMAGE/VIDEO` | 否 | - | (可选) 输入模型的图片或视频文件。支持多模态理解。 |
| `visual_input_3` | `IMAGE/VIDEO` | 否 | - | (可选) 输入模型的图片或视频文件。支持多模态理解。 |

#### 输出 (Outputs)

| 输出名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `content` | `STRING` | 模型生成的文本回答。 |
| `raw_json` | `STRING` | 包含完整元数据的 JSON 响应。 |
