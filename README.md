# ComfyUI 即梦 API 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型 API。用户可以通过这些节点在 ComfyUI 中使用文生图、图生图以及多种视频生成功能。

## 📦 安装

1.  **克隆仓库**:
    打开您的终端，`cd` 到 ComfyUI 的 `custom_nodes` 目录，然后运行：

    ```bash
    git clone [https://github.com/Jimeng-AI/ComfyUI-Jimeng-API.git](https://github.com/Jimeng-AI/ComfyUI-Jimeng-API.git)
    ```

2.  **安装依赖**:
    `cd` 到新克隆的 `ComfyUI-Jimeng-API` 目录，然后安装所需的 Python 包：

    ```bash
    cd ComfyUI-Jimeng-API
    pip install -r requirements.txt
    ```

3.  **重启 ComfyUI**。

## ⚙️ 设置：配置 API 密钥

1.  在插件根目录 (`ComfyUI/custom_nodes/ComfyUI-Jimeng-API/`) 中，找到 `api_keys.json.example` 文件。
2.  将其**重命名**为 `api_keys.json`。
3.  打开 `api_keys.json` 并填入您的密钥信息（[从此获取 Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）。

## ✨ 项目特点

- **智能统一节点**:
  - **图像生成**:
    - `即梦 Seedream 3` 节点会根据是否连接图像输入，**自动在后端智能切换**文生图 (`doubao-seedream-3-0-t2i-250415`) 和图生图编辑 (`doubao-seededit-3-0-i2i-250628`) 模型，无需用户手动配置。
    - `即梦 Seedream 4` 节点支持更高级的图生图（最多 10 张输入图像）和组图生成模式。
  - **视频生成**: `即梦视频生成` 节点统一了**文生视频**、**图生视频（首帧）** 和 **首尾帧生视频** 三种核心模式，极大简化了工作流。
- **优秀的用户体验**:
  - **流式进度条**: 支持视频生成等耗时任务的**流式进度条**，可在 ComfyUI 界面实时反馈任务状态。
  - **非阻塞模式**: 视频节点支持 `non_blocking` 模式。启用后，任务将提交到后台执行，工作流暂停。再次运行时会自动检查并获取已完成的结果。
  - **批量生成与保存**: 视频节点支持 `generation_count` 参数，可一次性并发提交多个任务。当数量大于 1 时，所有生成的视频和（可选的）尾帧将按 Seed 排序，并自动保存到 `output` 目录下的指定路径。
  - **多 Key 管理**: `即梦API客户端` 节点会自动从 `api_keys.json` 配置文件中读取并管理你的多个 API Key，方便在 UI 上通过下拉菜单快速切换。

## 📖 功能节点列表

(所有节点均位于 `JimengAPI` 菜单下)

- **基础设置**:
  - `即梦API客户端`: **(必须)** 用于加载和选择 `api_keys.json` 中的 API 密钥。
- **图像生成**:
  - `即梦 Seedream 3`: **统一的基础图像生成节点**。支持**文生图**（不连接图像输入时）和**图生图编辑**（连接图像输入时）。
  - `即梦 Seedream 4`: 高级图像生成节点，支持**组图生成**和**多图输入**。
- **视频生成**:
  - `即梦视频生成`: **统一的核心视频生成节点**。支持**文生视频**、**图生视频**（首帧）和**首尾帧生视频**三种模式。同时支持批量生成。
  - `即梦参考图转视频`: 根据一张或多张**参考图像**生成视频。同时支持批量生成。
  - `即梦视频生成任务列表查询`: 用于查询和管理在 API 上运行的视频任务。

## 📑 节点详解

### `即梦API客户端 (Jimeng API Client)`

加载 `api_keys.json` 文件中的密钥配置。**这是所有工作流的起点。**

- **输入**:
  - `key_name (STRING)`: 在 `api_keys.json` 中配置的密钥名称（例如 "default"）。
- **输出**:
  - `client (JIMENG_CLIENT)`: 即梦 API 客户端实例。

### `即梦 Seedream 3 (Jimeng Seedream 3)`

统一的文生图 (T2I) 和图生图 (I2I) 节点。

- **模式**:
  1.  **文生图**: 不连接 `image` 输入。
  2.  **图生图**: 连接 `image` 输入。
- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `prompt (STRING)`: 提示词。
  - `size (STRING)`: 尺寸。
  - `width (INT)`: 宽度 (仅当尺寸为 'Custom' 时生效)。
  - `height (INT)`: 高度 (仅当尺寸为 'Custom' 时生效)。
  - `seed (INT)`: 种子。
  - `guidance_scale (FLOAT)`: 引导系数。
  - `watermark (BOOLEAN)`: 水印。
  - `image (IMAGE)`: (可选, 用于图生图)。
- **输出**:
  - `image (IMAGE)`: 生成的图像。
  - `response (STRING)`: API 的 JSON 响应。

### `即梦 Seedream 4 (Jimeng Seedream 4)`

高级图像生成节点，支持组图生成和多图输入。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `prompt (STRING)`: 提示词。
  - `generation_mode (STRING)`: 生成模式 (单图或组图)。
  - `max_images (INT)`: 最大图片数 (用于组图模式)。
  - `size (STRING)`: 尺寸。
  - `width (INT)`: 宽度。
  - `height (INT)`: 高度。
  - `seed (INT)`: 种子。
  - `watermark (BOOLEAN)`: 水印。
  - `images (IMAGE)`: (可选, 用于图生图, 最多 10 张)。
- **输出**:
  - `images (IMAGE)`: 生成的图像（批处理）。
  - `response (STRING)`: API 的 JSON 响应。

### `即梦视频生成 (Jimeng Video Generation)`

统一的 T2V, I2V, 和 首尾帧视频生成节点。

- **模式**:
  1.  **文生视频**: 不连接任何图像输入。
  2.  **图生视频**: 只连接 `image` (首帧图像) 输入。
  3.  **首尾帧生视频**: 同时连接 `image` (首帧图像) 和 `last_frame_image` (尾帧图像) 输入。
- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `model_choice (STRING)`: 模型选择。
  - `prompt (STRING)`: 提示词。
  - `duration (FLOAT)`: 时长 (秒)。
  - `resolution (STRING)`: 分辨率。
  - `aspect_ratio (STRING)`: 宽高比。
  - `camerafixed (BOOLEAN)`: 固定镜头。
  - `seed (INT)`: 种子。
  - `generation_count (INT)`: (可选) 一次性并发生成的任务数量。如果大于 1，将激活批量保存功能。
  - `batch_save_path (STRING)`: (可选) 当生成数量大于 1 时，所有文件在 'output' 目录下的保存路径。
  - `save_last_frame_batch (BOOLEAN)`: (可选) 当批量生成时，是否同时保存所有视频的最后一帧。
  - `non_blocking (BOOLEAN)`: 非阻塞模式。
  - `image (IMAGE)`: (可选, 首帧图像)。
  - `last_frame_image (IMAGE)`: (可选, 尾帧图像)。
- **输出**:
  - `video (VIDEO)`: 生成的视频（批量生成时为第一个）。
  - `last_frame (IMAGE)`: 视频的最后一帧图像（批量生成时为第一个）。
  - `response (STRING)`: 任务完成后的 API 原始响应（批量生成时为列表）。

### `即梦参考图转视频 (Jimeng Reference to Video)`

根据一张或多张参考图生成视频。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `prompt (STRING)`: 提示词。
  - `duration (FLOAT)`: 时长 (秒)。
  - `resolution (STRING)`: 分辨率 (480p, 720p)。
  - `aspect_ratio (STRING)`: 宽高比。
  - `seed (INT)`: 种子。
  - `generation_count (INT)`: (可选) 一次性并发生成的任务数量。如果大于 1，将激活批量保存功能。
  - `batch_save_path (STRING)`: (可选) 当生成数量大于 1 时，所有文件在 'output' 目录下的保存路径。
  - `save_last_frame_batch (BOOLEAN)`: (可选) 当批量生成时，是否同时保存所有视频的最后一帧。
  - `non_blocking (BOOLEAN)`: 非阻塞模式。
  - `ref_image_1 (IMAGE)`: (必须) 参考图 1。
  - `ref_image_2 (IMAGE)`: (可选) 参考图 2。
  - `ref_image_3 (IMAGE)`: (可选) 参考图 3。
  - `ref_image_4 (IMAGE)`: (可选) 参考图 4。
- **输出**:
  - `video (VIDEO)`: 生成的视频（批量生成时为第一个）。
  - `last_frame (IMAGE)`: 视频的最后一帧图像（批量生成时为第一个）。
  - `response (STRING)`: 任务完成后的 API 原始响应（批量生成时为列表）。

### `即梦视频生成任务列表查询 (Jimeng Query Tasks)`

查询符合条件的即梦 API 视频生成任务列表。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `page_num (INT)`: 页码。
  - `page_size (INT)`: 每页数量。
  - `status (STRING)`: 任务状态 (all, succeeded, failed, running, etc.)。
  - `task_ids (STRING)`: (可选) 要查询的特定任务 ID（每行一个）。
  - `model_choice (STRING)`: (可选) 按特定模型过滤。
  - `custom_model_id (STRING)`: (可选) 按自定义模型 ID 过滤。
- **输出**:
  - `task_list_json (STRING)`: 查询到的任务列表（JSON 字符串）。
  - `total_tasks (INT)`: 符合条件的总任务数量。

## workflows 示例工作流

您可以在 `example_workflows` 目录中找到所有节点的示例工作流。