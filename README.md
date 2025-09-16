# ComfyUI Jimeng API Nodes

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型API。用户可以通过这些节点在 ComfyUI 中使用文生图、图生图以及多种视频生成功能。

本项目使用了火山方舟官方的 `volcengine-python-sdk` 以确保API调用的稳定性。

## 功能节点列表

* **图像生成**:
    * `JimengText2Image (Seedream 3)`: 使用 `doubao-seedream-3.0` 模型进行文生图。
    * `JimengImageEdit (Seededit 3)`: 使用 `doubao-seededit-3.0` 模型进行图生图编辑。
    * `JimengSeedream4`: 使用最新的 `doubao-seedream-4.0` 模型，支持高达4K分辨率的文生图和多图参考(暂时不支持组图生成）。
* **视频生成**:
    * `JimengImage2Video (I2V)`: 使用单个首帧图像引导生成视频。
    * `JimengFirstLastFrame2Video (F&L-I2V)`: 使用首、尾两张图像控制视频的开始和结束。
    * `JimengReferenceImage2Video (Ref-I2V)`: 使用1至4张参考图像影响视频的生成结果。
* **工具节点**:
    * `JimengAPIClient`: 用于配置火山方舟的 API Key。
    * `JimengTaskStatusChecker`: 查询视频任务的状态，可获取视频URL、尾帧URL、错误信息及各项生成参数。此节点可用于恢复因网络超时等原因未能正常返回结果的任务。
    * `PreviewImageFromUrl` / `PreviewVideoFromUrl`: 用于在UI中直接预览并保存从API获取的图像和视频结果。

## 安装

1.  在终端中，导航至 `ComfyUI/custom_nodes/` 目录。
2.  克隆本仓库：
    ```bash
    git clone [https://github.com/fkxianzhou/ComfyUI-Jimeng-API.git](https://github.com/fkxianzhou/ComfyUI-Jimeng-API.git)
    ```
3.  安装依赖：
    ```bash
    pip install -r ComfyUI-Jimeng-API/requirements.txt
    ```
4.  重启 ComfyUI。

## 节点使用说明

所有节点均位于 `"JimengAI"` 分类下。

### `JimengAPIClient`
* **输入**: `api_key` (火山方舟 API Key)。
* **输出**: `client` (用于连接到其他 Jimeng 节点)。

### `JimengText2Image (Seedream 3)`
* **功能**: 基础文生图。
* **输出**: `image`, `seed`。

### `JimengImageEdit (Seededit 3)`
* **功能**: 根据输入图像和提示词进行编辑。
* **输入**: `client`, `image`, `prompt`, `seed` 等。
* **输出**: `image`, `seed`。

### `JimengSeedream4`
* **功能**: 高级文生图/图生图，支持组图模式。
* **输入**: `client`, `prompt`, `size`, `seed`, `images` (可选参考图), `sequential_image_generation` (组图开关), `max_images` (组图数量)。
* **输出**: `images` (单张或多张图片), `seed`。

### `JimengImage2Video (I2V)`
* **功能**: 首帧图生视频。
* **输出**: `url` (视频链接), `task_id`, `seed`。

### `JimengFirstLastFrame2Video (F&L-I2V)`
* **功能**: 首尾帧图生视频。
* **输出**: `url`, `task_id`, `seed`, `last_frame_url`。

### `JimengReferenceImage2Video (Ref-I2V)`
* **功能**: 参考图生视频。
* **输出**: `url`, `task_id`, `seed`。

### `JimengTaskStatusChecker`
* **功能**: 查询视频任务详情。
* **输出**: 视频/尾帧URL、任务状态、错误信息、模型、时间戳、详细视频参数和Token用量。

### 关于种子 (Seed) 的说明

根据官方文档，即便是相同的参数和种子，每次生成的视频结果也可能存在细微差异。因此，固定种子的主要用途是在保持整体风格和动态一致的前提下，通过修改提示词来进行迭代创作，而不是为了精确复现某一结果。

---

## 来源与致谢

本项目 Fork 自 [xuhongming251/ComfyUI-Jimeng](https://github.com/xuhongming251/ComfyUI-Jimeng)，并在其基础上进行了功能开发和修改。感谢原作者 `xuhongming251` 的工作。
