### Jimeng Seedream 5
基于最新的 Seedream 5.0 Lite 模型进行高质量图像生成，支持联网搜索。

*   **Node ID**: `JimengSeedream5`
*   **Python Class**: `JimengSeedream5`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 是 | - | 连接 `Jimeng API Client` 的输出。 |
| `model_version` | `COMBO` | 是 | - | 选择模型版本 (目前支持 "doubao-seedream-5.0-lite")。 |
| `prompt` | `STRING` | 是 | "" | 图像生成的提示词。 |
| `enable_group_generation`| `BOOLEAN`| 是 | False | 是否启用组图生成 (多张图内容一致性)。 |
| `enable_web_search` | `BOOLEAN` | 是 | False | 是否启用联网搜索，提升时效性。 |
| `max_images` | `INT` | 是 | 1 | 组图生成时的最大图像数量 (1-15)。 |
| `size` | `COMBO` | 是 | - | 预设尺寸 (含 2K/3K) 或 "Custom"。 |
| `width` | `INT` | 是 | 2048 | 自定义宽度，范围 1-8192。 |
| `height` | `INT` | 是 | 2048 | 自定义高度，范围 1-8192。 |
| `seed` | `INT` | 是 | 0 | 随机种子。 |
| `generation_count` | `INT` | 是 | 1 | 批量生成数量。 |
| `watermark` | `BOOLEAN` | 是 | False | 是否添加水印。 |
| `images` | `IMAGE` | 否 | - | (可选) 输入图像 Batch，用于参考生成。 |

#### 输出 (Outputs)

| 输出名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `images` | `IMAGE` | 生成的图像 (Batch)。 |
| `response` | `STRING` | API 响应的 JSON 数据。 |
