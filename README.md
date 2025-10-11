# ComfyUI 即梦 API 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型 API。用户可以通过这些节点在 ComfyUI 中使用文生图、图生图以及多种视频生成功能。

## 功能节点列表

- **图像生成**:
  - `即梦 Seedream 3`: **统一的基础图像生成节点**。支持**文生图**（不连接图像输入时）和**图生图编辑**（连接图像输入时），智能切换 `doubao-seedream-3.0` 和 `doubao-seededit-3.0` 模型。
  - `即梦 Seedream 4`: 使用最新的 `doubao-seedream-4.0` 模型，支持高达 4K 分辨率的文生图、多图参考以及强大的**组图生成**功能。
- **视频生成**:
  - `即梦视频生成`: **统一的核心视频生成节点**。通过不同的输入组合，支持**文生视频**、**图生视频（首帧）**以及**首尾帧生视频**三种模式，并可智能切换 `lite` 模型版本。
  - `即梦参考图生视频 (Ref-I2V)`: 使用 1 至 4 张参考图像影响视频的生成结果（最高支持 `720p`）。
- **工具节点**:
  - `即梦API客户端 (Jimeng Client)`: **（必须）** 从配置文件 `api_keys.json` 读取密钥（[从此获取 Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)），并通过下拉菜单选择使用哪个密钥。
  - `即梦任务状态检查器`: 手动查询视频任务的状态，可用于恢复因意外中断的任务。

## 安装与配置

1.  **克隆仓库**
    在终端中，导航至 `ComfyUI/custom_nodes/` 目录，然后运行：

    ```bash
    git clone https://github.com/fkxianzhou/ComfyUI-Jimeng-API.git
    ```

2.  **安装依赖**
    进入 ComfyUI 的根目录，使用其内置的 Python 环境分别安装所需的库：

    ```bash
    # 安装火山引擎SDK (用于视频)
    .\python_embeded\python.exe -m pip install --upgrade "volcengine-python-sdk[ark]"

    # 安装OpenAI库 (用于图片)
    .\python_embeded\python.exe -m pip install --upgrade openai
    ```

3.  **配置 API 密钥 (重要)**

    - 进入 `ComfyUI/custom_nodes/ComfyUI-Jimeng-API/` 目录。
    - 找到名为 `api_keys.json.example` 的文件，**将其复制并重命名为 `api_keys.json`**。
    - 用文本编辑器打开 `api_keys.json` 文件，按照内部格式填入你的火山方舟 API Key（[从此获取 Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）。你可以配置多个 Key。
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

### `即梦API客户端 (Jimeng Client)`

此节点是所有操作的起点，负责管理你的 API 密钥。

- **输入**: 节点上只有一个下拉菜单 `key_name`，它会自动读取你在 `api_keys.json` 中配置的所有 `"customName"`。
- **输出**:
  - `client`: **将此统一的客户端输出连接到所有其他的即梦节点** (`Seedream 3`, `Seedream 4`, `视频生成` 等)。

### `即梦视频生成`

统一的视频生成入口，功能强大且智能。

- **核心模式**:
  1.  **文生视频**: 不连接 `首帧图像` 和 `尾帧图像` 输入。
  2.  **图生视频**: 只连接 `首帧图像` 输入，它将作为视频的起始画面。
  3.  **首尾帧生视频**: 同时连接 `首帧图像` 和 `尾帧图像` 输入。
- **智能模型切换**:
  - 当 `model_choice` 选择 `doubao-seedance-1-0-lite` 时，节点会根据有无 `首帧图像` 输入，自动调用正确的 `t2v` 或 `i2v` 模型。
- **输出**: **直接在节点上预览生成的视频**。同时会输出视频的最后一帧图像，可用于连接其他节点。

### `即梦 Seedream 3`

统一的基础图像生成节点。

- **核心模式**:
  - **文生图**: 不连接 `图像 (可选)` 输入。
  - **图生图 (编辑)**: 连接 `图像 (可选)` 输入。此时，`尺寸`参数将自动适应输入图像，无需手动设置。
- **输出**: `image`, `seed`。

### `即梦 Seedream 4`

功能最强大的图像节点。

- **功能**: 高级文生图/图生图，支持单图和组图模式。
- **重要输入**:
  - `generation_mode`: 切换 `生成单图 (disabled)` 或 `生成组图 (auto)` 模式。
  - `max_images`: 在“生成组图”模式下，设置期望生成的最大图片数量。
  - `images` (可选): 连接单张或多张（批次）图片作为参考。
- **输出**: `images` (单张或一组图片), `seed`。

### 关于种子 (Seed) 的说明

根据官方文档，即便是相同的参数和种子，每次生成的视频结果也可能存在细微差异。因此，固定种子的主要用途是在保持整体风格和动态一致的前提下，通过修改提示词来进行迭代创作，而不是为了精确复现某一结果。
