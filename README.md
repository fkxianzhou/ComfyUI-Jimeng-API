# ComfyUI 即梦 API 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型 API。用户可以通过这些节点在 ComfyUI 中使用文生图、图生图以及多种视频生成功能。

## 项目特点

- **智能统一节点**:
  - **图像生成**: `即梦 Seedream 3` 节点会根据是否连接图像输入，**自动在后端智能切换**文生图 (`doubao-seedream-3.0`) 和图生图编辑 (`doubao-seededit-3.0`) 模型，无需用户手动配置。
  - **视频生成**: `即梦视频生成` 节点统一了**文生视频**、**图生视频（首帧）** 和 **首尾帧生视频** 三种核心模式，极大简化了工作流。
- **优秀的用户体验**:
  - **流式进度条**: 支持视频生成等耗时任务的**流式进度条**，可在 ComfyUI 界面实时反馈任务状态。
  - **多 Key 管理**: `即梦API客户端` 节点会自动从 `api_keys.json` 配置文件中读取并管理你的多个 API Key，方便在 UI 上通过下拉菜单快速切换。

## 功能节点列表

- **图像生成**:
  - `即梦 Seedream 3`: **统一的基础图像生成节点**。支持**文生图**（不连接图像输入时）和**图生图编辑**（连接图像输入时），智能切换 `doubao-seedream-3.0` 和 `doubao-seededit-3.0` 模型。
  - `即梦 Seedream 4`: 使用最新的 `doubao-seedream-4.0` 模型，支持高达 4K 分辨率的文生图、多图参考以及强大的**组图生成**功能。
- **视频生成**:
  - `即梦视频生成`: **统一的核心视频生成节点**。通过不同的输入组合，支持**文生视频**、**图生视频（首帧）以及首尾帧生视频**三种模式。
  - `即梦参考图生视频 (Ref-I2V)`: 使用 1 至 4 张参考图像影响视频的生成结果（最高支持 `720p`）。
- **工具节点**:
  - `即梦API客户端 (Jimeng Client)`: **（必须）** 从配置文件 `api_keys.json` 读取密钥（[从此获取 Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）。
  - `即梦任务状态检查器 (Jimeng Task Status Checker)`: 手动查询视频任务的状态和结果。

## 安装与更新

1.  **安装**:
    - 确保你已安装 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)。
    - 在 ComfyUI-Manager 中搜索 `Jimeng` 并安装此节点包。
    - **或者**：手动 `git clone` 本仓库到 `ComfyUI/custom_nodes/` 目录下。
2.  **安装依赖**:
    - 安装完成后，在 `ComfyUI/custom_nodes/ComfyUI-Jimeng-API/` 目录下，运行 `pip install -r requirements.txt` 来安装所需的 Python 依赖库。
3.  **配置密钥**:
    - 在 `ComfyUI/custom_nodes/ComfyUI-Jimeng-API/` 目录下，将 `api_keys.json.example` 文件**重命名**为 `api_keys.json`。
    - 打开 `api_keys.json` 文件，填入你的火山方舟 API Key（`apiKey`）并为其指定一个自定义名称（`customName`）。你可以配置多个 Key。
    <!-- end list -->
    ```json
    [
      {
        "customName": "我的即梦Key-1",
        "apiKey": "volc_key_goes_here"
      },
      {
        "customName": "备用Key",
        "apiKey": "another_volc_key_goes_here"
      }
    ]
    ```
4.  **重启 ComfyUI**。

## 节点使用说明

所有节点均位于 `"JimengAI"` 分类下。

### `即梦API客户端 (Jimeng API Client)`

此节点是所有操作的起点，负责管理你的 API 密钥。

- **输入**:
  - `key_name`: 节点上的下拉菜单，它会自动读取你在 `api_keys.json` 中配置的所有 `customName`。
- **输出**:
  - `client`: **将此统一的客户端输出连接到所有其他的即梦节点** (`Seedream 3`, `Seedream 4`, `视频生成` 等)。

### `即梦 Seedream 3 (Jimeng Seedream 3)`

使用 Seedream 3 / Seededit 3 进行文生图和图生图。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `prompt (STRING)`: 提示词。
  - `size (STRING)`: 尺寸。
  - `width (INT)`, `height (INT)`: (当尺寸为 "Custom" 时使用)。
  - `seed (INT)`: 种子。
  - `guidance_scale (FLOAT)`: 引导系数。
  - `watermark (BOOLEAN)`: 水印。
  - `image (IMAGE)`: (可选, 用于图生图)。
- **输出**:
  - `image (IMAGE)`: 生成的图像。
  - `response (STRING)`: 包含 URL 和种子等信息的 API 原始响应。

### `即梦 Seedream 4 (Jimeng Seedream 4)`

使用 Seedream 4 进行文生图和图生图。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `prompt (STRING)`: 提示词。
  - `generation_mode (STRING)`: 生成模式。
  - `max_images (INT)`: 最大图片数。
  - `size (STRING)`: 尺寸。
  - `width (INT)`, `height (INT)`
  - `seed (INT)`: 种子。
  - `watermark (BOOLEAN)`: 水印。
  - `images (IMAGE)`: (可选, 用于图生图)。
- **输出**:
  - `images (IMAGE)`: 生成的图像批次。
  - `response (STRING)`: 包含 URL 和种子等信息的 API 原始响应。

### `即梦视频生成 (Jimeng Video Generation)`

统一的视频生成入口，功能强大且智能。

- **核心模式**:
  1.  **文生视频**: 不连接 `首帧图像` 和 `尾帧图像` 输入。
  2.  **图生视频**: 只连接 `首帧图像` 输入，它将作为视频的起始画面。
  3.  **首尾帧生视频**: 同时连接 `首帧图像` 和 `尾帧图像` 输入。
- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `model_choice (STRING)`: 模型选择。
  - `prompt (STRING)`: 提示词。
  - `duration (FLOAT)`: 时长 (秒)。支持输入整数（如 5.0）或小数（如 2.4）。输入小数时，节点会自动计算并使用最接近的有效帧数。
  - `resolution (STRING)`: 分辨率。
  - `aspect_ratio (STRING)`: 宽高比。
  - `camerafixed (BOOLEAN)`: 固定镜头。
  - `seed (INT)`: 种子。
  - `image (IMAGE)`: (可选, 首帧图像)。
  - `last_frame_image (IMAGE)`: (可选, 尾帧图像)。
- **输出**:
  - `video (VIDEO)`: 生成的视频。
  - `last_frame (IMAGE)`: 视频的最后一帧图像。
  - `response (STRING)`: 任务完成后的 API 原始响应。

### `即梦参考图转视频 (Jimeng Reference to Video)`

根据多张参考图生成视频。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `prompt (STRING)`: 提示词。
  - `duration (INT)`: 时长 (秒)。支持输入整数（如 5.0）或小数（如 2.4）。输入小数时，节点会自动计算并使用最接近的有效帧数。
  - `resolution (STRING)`: 分辨率。
  - `aspect_ratio (STRING)`: 宽高比。
  - `seed (INT)`: 种子。
  - `ref_image_1` 到 `ref_image_4 (IMAGE)` (可选)。
- **输出**:
  - `video (VIDEO)`: 生成的视频。
  - `last_frame (IMAGE)`: 视频的最后一帧图像。
  - `response (STRING)`: 任务完成后的 API 原始响应。

### `即梦任务状态检查器 (Jimeng Task Status Checker)`

手动检查视频生成任务的状态。

- **输入**:
  - `client (JIMENG_CLIENT)`: 客户端。
  - `task_id (STRING)`: 任务 ID。
- **输出**:
  - `video_url (STRING)`: 视频链接。
  - `last_frame_url (STRING)`: 尾帧链接。
  - `status (STRING)`: 状态。
  - `error_message (STRING)`: 错误信息。
  - `model (STRING)`: 模型。
  - `created_at (STRING)`: 创建时间。
  - `updated_at (STRING)`: 更新时间。
