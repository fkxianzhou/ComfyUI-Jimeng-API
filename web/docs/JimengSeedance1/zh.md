### Jimeng Seedance 1.0
基础视频生成节点，支持文生视频和图生视频。

*   **Node ID**: `JimengSeedance1`
*   **Python Class**: `JimengSeedance1`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 是 | - | API 客户端。 |
| `model_version` | `COMBO` | 是 | - | 视频模型版本。 |
| `prompt` | `STRING` | 是 | "" | 视频提示词。 |
| `duration` | `FLOAT` | 是 | 5.0 | 视频时长 (秒)，范围 1.2-12.0。 |
| `resolution` | `COMBO` | 是 | "720p"| 分辨率 (480p, 720p, 1080p)。 |
| `aspect_ratio` | `COMBO` | 是 | "adaptive" | 宽高比 (16:9, 9:16 等)。 |
| `camerafixed` | `BOOLEAN` | 是 | True | 是否固定相机视角。 |
| `enable_random_seed` | `BOOLEAN` | 是 | True | 是否启用随机种子 (覆盖 `seed` 参数)。 |
| `seed` | `INT` | 是 | 0 | 种子值。 |
| `generation_count` | `INT` | 是 | 1 | 批量生成数量。 |
| `filename_prefix` | `STRING` | 是 | "Jimeng/Video" | 保存文件的前缀。 |
| `save_last_frame_batch`| `BOOLEAN` | 是 | False | 是否单独保存最后一帧。 |
| `timeout_seconds` | `INT` | 是 | 172800 | 任务超时时间 (秒)。 |
| `enable_offline_inference`| `BOOLEAN`| 是 | False | 是否启用离线推理模式 (Flex Tier)。 |
| `non_blocking` | `BOOLEAN` | 是 | False | 是否使用非阻塞异步模式 (立即返回，后台轮询)。 |
| `image` | `IMAGE` | 否 | - | (可选) 首帧图像。 |
| `last_frame_image` | `IMAGE` | 否 | - | (可选) 尾帧图像 (需同时提供首帧)。 |

#### 输出 (Outputs)

| 输出名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `video` | `VIDEO` | 生成的视频文件路径对象。 |
| `last_frame` | `IMAGE` | 视频的最后一帧图像。 |
| `response` | `STRING` | 任务响应 JSON。 |