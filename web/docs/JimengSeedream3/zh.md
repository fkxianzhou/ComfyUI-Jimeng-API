### Jimeng Seedream 3
基于 Seedream 3 模型进行图像生成 (Text-to-Image / Image-to-Image)。

*   **Node ID**: `JimengSeedream3`
*   **Python Class**: `JimengSeedream3`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 是 | - | 连接 `Jimeng API Client` 的输出。 |
| `prompt` | `STRING` | 是 | "" | 图像生成的提示词 (支持多行)。 |
| `size` | `COMBO` | 是 | - | 预设的图像尺寸比例 (如 "1024x1024 (1:1)")。若选择 "Custom"，则使用 `width` 和 `height`。 |
| `width` | `INT` | 是 | 1024 | 自定义宽度 (仅当 `size` 为 "Custom" 时生效)，范围 1-8192。 |
| `height` | `INT` | 是 | 1024 | 自定义高度 (仅当 `size` 为 "Custom" 时生效)，范围 1-8192。 |
| `seed` | `INT` | 是 | 0 | 随机种子。-1 表示随机。 |
| `guidance_scale` | `FLOAT` | 是 | 5.0 | 提示词跟随度 (CFG Scale)，范围 1.0-10.0。 |
| `generation_count` | `INT` | 是 | 1 | 批量生成的数量 (1-2048)。 |
| `watermark` | `BOOLEAN` | 是 | False | 是否添加水印。 |
| `image` | `IMAGE` | 否 | - | (可选) 输入图像，用于图生图模式。 |

#### 输出 (Outputs)

| 输出名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `image` | `IMAGE` | 生成的图像 (Batch)。 |
| `response` | `STRING` | API 响应的原始 JSON 数据。 |