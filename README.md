# ComfyUI 即梦 API 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型API。用户可以通过这些节点在 ComfyUI 中使用文生图、图生图以及多种视频生成功能。

为了提升代码的简洁性和标准化，本项目采用**混合模式**调用API：

  * **图片生成**功能采用 `openai` 库，利用火山方舟兼容OpenAI的API格式，代码更简洁、稳定。
  * **视频生成**功能则继续使用火山方舟官方的 `volcengine-python-sdk`，以确保对特殊任务接口的完全兼容。

## 功能节点列表

  * **图像生成**:
      * `即梦文生图 (Seedream 3)`: 使用 `doubao-seedream-3.0` 模型进行文生图。
      * `即梦图像编辑 (Seededit 3)`: 使用 `doubao-seededit-3.0` 模型进行图生图编辑。
      * `即梦高级图像 (Seedream 4)`: 使用最新的 `doubao-seedream-4.0` 模型，支持高达4K分辨率的文生图、多图参考以及强大的**组图生成**功能。
  * **视频生成**:
      * `即梦视频生成`: **统一的视频生成节点**，支持**文生视频**和**图生视频**两种模式。
      * `即梦首尾帧生视频 (F&L-I2V)`: 使用首、尾两张图像控制视频的开始和结束。
      * `即梦参考图生视频 (Ref-I2V)`: 使用1至4张参考图像影响视频的生成结果。
  * **工具节点**:
      * `即梦API客户端 (Jimeng Client)`: 用于配置火山方舟的 API Key，会为不同类型的任务提供相应的客户端。
      * `即梦任务状态检查器`: 手动查询视频任务的状态，可用于恢复因意外中断的任务。

## 安装

1.  在终端中，导航至 `ComfyUI/custom_nodes/` 目录。
2.  如果尚未克隆，请克隆本仓库：
    ```bash
    git clone https://github.com/fkxianzhou/ComfyUI-Jimeng-API.git
    ```
3.  安装依赖（需要分别安装两个库）：
    ```bash
    # 进入ComfyUI的根目录，使用其内置的Python环境
    # 安装火山引擎SDK (用于视频)
    .\python_embeded\python.exe -m pip install --upgrade "volcengine-python-sdk[ark]"

    # 安装OpenAI库 (用于图片)
    .\python_embeded\python.exe -m pip install --upgrade openai
    ```
4.  重启 ComfyUI。

## 节点使用说明

所有节点均位于 `"JimengAI"` 分类下。

### `即梦API客户端 (Jimeng Client)`

此节点是所有操作的起点。

  * **输入**: `api_key` (你的火山方舟 API Key)。
  * **输出**:
      * `openai_client`: **请将此输出连接到所有“图片”生成节点** (`文生图`, `图像编辑`, `高级图像`)。
      * `ark_client`: **请将此输出连接到所有“视频”生成节点** (`视频生成`, `首尾帧`, `参考图`) 及 `任务状态检查器`。

### `即梦高级图像 (Seedream 4)`

功能最强大的图像节点。

  * **功能**: 高级文生图/图生图，支持单图和组图模式。
  * **重要输入**:
      * `generation_mode`: 切换 `生成单图 (disabled)` 或 `生成组图 (auto)` 模式。
      * `max_images`: 在“生成组图”模式下，设置期望生成的最大图片数量。
      * `images` (可选): 连接单张或多张（批次）图片作为参考。
  * **输出**: `images` (单张或一组图片), `seed`。

### `即梦视频生成`

统一的视频生成入口。

  * **功能**: 根据输入自动判断执行文生视频或图生视频。
      * **文生视频**: 不连接 `image` 输入。
      * **图生视频**: 连接 `image` 输入作为视频的首帧。
  * **输出**: **直接在节点上预览生成的视频**。该节点没有数据输出端口。

### `即梦首尾帧生视频 (F&L-I2V)` & `即梦参考图生视频 (Ref-I2V)`

  * **功能**: 与原版功能一致，分别用于首尾帧和参考图的视频生成。
  * **输出**: **直接在节点上预览生成的视频**。这些节点没有数据输出端口。

### `即梦任务状态检查器`

  * **功能**: 手动输入`task_id`查询视频任务详情。
  * **输出**: `video_url`, `last_frame_url`, `status`, `error_message`, `model`, `created_at` (创建时间), `updated_at` (更新时间)。

### 关于种子 (Seed) 的说明

根据官方文档，即便是相同的参数和种子，每次生成的视频结果也可能存在细微差异。因此，固定种子的主要用途是在保持整体风格和动态一致的前提下，通过修改提示词来进行迭代创作，而不是为了精确复现某一结果。

-----

## 来源与致谢

本项目 Fork 自 [xuhongming251/ComfyUI-Jimeng](https://github.com/xuhongming251/ComfyUI-Jimeng)，并在其基础上进行了功能开发和修改。感谢原作者 `xuhongming251` 的工作。
