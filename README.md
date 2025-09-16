# ComfyUI Jimeng API Nodes

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型API。用户可以通过这些节点在 ComfyUI 中使用文生图和多种视频生成功能。

本项目使用了火山方舟官方的 `volcengine-python-sdk` 以确保API调用的稳定性。

## 功能节点列表

* **文生图**:
    * `JimengText2Image`: 基础的文本到图片生成节点。
* **视频生成**:
    * `JimengImage2Video (I2V)`: 使用单个首帧图像引导生成视频。
    * `JimengFirstLastFrame2Video (F&L-I2V)`: 使用首、尾两张图像控制视频的开始和结束。
    * `JimengReferenceImage2Video (Ref-I2V)`: 使用1至4张参考图像影响视频的生成结果。
* **工具节点**:
    * `JimengAPIClient`: 用于配置火山方舟的 API Key。
    * `JimengTaskStatusChecker`: 查询指定 `task_id` 的任务状态，可以获取视频URL、尾帧URL、错误信息及各项生成参数。此节点可用于恢复因网络超时等原因未能正常返回结果的任务。
    * `PreviewImageFromUrl` / `PreviewVideoFromUrl`: 用于在UI中直接预览并保存从API获取的图像和视频结果。

## 安装

1.  在终端中，导航至 `ComfyUI/custom_nodes/` 目录。
2.  克隆本仓库：
    ```bash
    git clone <你的项目仓库HTTPS链接>
    ```
3.  安装依赖：
    ```bash
    pip install -r <你的项目文件夹名>/requirements.txt
    ```
4.  重启 ComfyUI。

## 节点使用说明

所有节点均位于 `"JimengAI"` 分类下。

### `JimengAPIClient`
* **输入**: `api_key` (火山方舟 API Key)。
* **输出**: `client` (用于连接到其他 Jimeng 节点)。

### `JimengText2Image (T2I)`
* **输入**: `client`, `prompt`, `size`, `seed` 等。
* **输出**: `image_url`, `seed`。

### `JimengImage2Video (I2V)`
* **输入**: `client`, `image` (首帧), `prompt`, `duration` (3-12秒), `resolution`, `seed` 等。
* **输出**: `url` (视频链接), `task_id`, `seed`。

### `JimengFirstLastFrame2Video (F&L-I2V)`
* **输入**: `client`, `first_frame_image`, `last_frame_image`, `return_last_frame` (布尔开关)等。
* **输出**: `url`, `task_id`, `seed`, `last_frame_url`。

### `JimengReferenceImage2Video (Ref-I2V)`
* **输入**: `client`, `prompt`, `ref_image_1` 至 `ref_image_4` (至少提供一张)。
* **输出**: `url`, `task_id`, `seed`。

### `JimengTaskStatusChecker`
* **输入**: `client`, `task_id`。
* **输出**: 视频/尾帧URL、任务状态、错误信息、模型、时间戳、详细视频参数和Token用量。

### 关于种子 (Seed) 的说明

根据官方文档，即便是相同的参数和种子，每次生成的视频结果也可能存在细微差异。因此，固定种子的主要用途是在保持整体风格和动态一致的前提下，通过修改提示词来进行迭代创作，而不是为了精确复现某一结果。

---

## 来源与致谢

本项目 Fork 自 [xuhongming251/ComfyUI-Jimeng](https://github.com/xuhongming251/ComfyUI-Jimeng)，并在其基础上进行了功能开发和修改。感谢原作者 `xuhongming251` 的工作。
