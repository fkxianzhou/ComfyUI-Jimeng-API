# ComfyUI 即梦 API 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了一系列自定义节点，用于对接火山方舟（即梦）的多模态模型API。用户可以通过这些节点在 ComfyUI 中使用文生图、图生图以及多种视频生成功能。

## 功能节点列表

  * **图像生成**:
      * `即梦文生图 (Seedream 3)`: 使用 `doubao-seedream-3.0` 模型进行文生图。
      * `即梦图像编辑 (Seededit 3)`: 使用 `doubao-seededit-3.0` 模型进行图生图编辑。
      * `即梦高级图像 (Seedream 4)`: 使用最新的 `doubao-seedream-4.0` 模型，支持高达4K分辨率的文生图、多图参考以及强大的**组图生成**功能。
  * **视频生成**:
      * `即梦视频生成`: **统一的视频生成节点**，支持**文生视频**和**图生视频**两种模式，并可智能切换 `lite` 模型版本。
      * `即梦首尾帧生视频 (F&L-I2V)`: 使用首、尾两张图像控制视频的开始和结束，支持 `1080p` 分辨率。
      * `即梦参考图生视频 (Ref-I2V)`: 使用1至4张参考图像影响视频的生成结果（最高支持 `720p`）。
  * **工具节点**:
      * `即梦API客户端 (Jimeng Client)`: **从配置文件 `api_keys.json` 读取密钥（[从此获取Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）**，并通过下拉菜单选择使用哪个密钥。
      * `即梦任务状态检查器`: 手动查询视频任务的状态，可用于恢复因意外中断的任务。

## 安装与配置

1.  **克隆仓库**
    在终端中，导航至 `ComfyUI/custom_nodes/` 目录，然后运行：

    ```bash
    git clone https://github.com/fkxianzhou/ComfyUI-Jimeng-API.git
    ```

2.  **安装依赖**
    进入ComfyUI的根目录，使用其内置的Python环境分别安装所需的库：

    ```bash
    # 安装火山引擎SDK (用于视频)
    .\python_embeded\python.exe -m pip install --upgrade "volcengine-python-sdk[ark]"

    # 安装OpenAI库 (用于图片)
    .\python_embeded\python.exe -m pip install --upgrade openai
    ```

3.  **配置API密钥 (重要)**

      * 进入 `ComfyUI/custom_nodes/ComfyUI-Jimeng-API/` 目录。
      * 找到名为 `api_keys.json.example` 的文件，**将其复制并重命名为 `api_keys.json`**。
      * 用文本编辑器打开 `api_keys.json` 文件，按照内部格式填入你的火山方舟API Key（[从此获取Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）。你可以配置多个Key。
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

此节点是所有操作的起点，负责管理你的API密钥。

  * **输入**: 节点上只有一个下拉菜单 `key_name`，它会自动读取你在 `api_keys.json` 中配置的所有 `"customName"`。
  * **输出**:
      * `openai_client`: **请将此输出连接到所有“图片”生成节点** (`文生图`, `图像编辑`, `高级图像`)。
      * `ark_client`: **请将此输出连接到所有“视频”生成节点** (`视频生成`, `首尾帧`, `参考图`) 及 `任务状态检查器`。

### `即梦视频生成`

统一的视频生成入口，功能强大且智能。

  * **功能**:
      * **文生视频**: 不连接 `image` 输入。
      * **图生视频**: 连接 `image` 输入作为视频的首帧。
  * **智能模型切换**:
      * 当 `model_choice` 选择 `doubao-seedance-1-0-lite` 时，节点会根据有无 `image` 输入，自动调用正确的 `t2v` 或 `i2v` 模型。
  * **输出**: **直接在节点上预览生成的视频**。该节点没有数据输出端口。

### `即梦首尾帧生视频 (F&L-I2V)` & `即梦参考图生视频 (Ref-I2V)`

  * **功能**: 分别用于首尾帧和参考图的视频生成。
  * **注意**:
      * `首尾帧` 节点最高支持 **`1080p`** 分辨率。
      * `参考图` 节点最高支持 **`720p`** 分辨率。
  * **输出**: **直接在节点上预览生成的视频**。

### `即梦高级图像 (Seedream 4)`

功能最强大的图像节点。

  * **功能**: 高级文生图/图生图，支持单图和组图模式。
  * **重要输入**:
      * `generation_mode`: 切换 `生成单图 (disabled)` 或 `生成组图 (auto)` 模式。
      * `max_images`: 在“生成组图”模式下，设置期望生成的最大图片数量。
      * `images` (可选): 连接单张或多张（批次）图片作为参考。
  * **输出**: `images` (单张或一组图片), `seed`。

### 关于种子 (Seed) 的说明

根据官方文档，即便是相同的参数和种子，每次生成的视频结果也可能存在细微差异。因此，固定种子的主要用途是在保持整体风格和动态一致的前提下，通过修改提示词来进行迭代创作，而不是为了精确复现某一结果。

-----

## 来源与致谢

本项目 Fork 自 [xuhongming251/ComfyUI-Jimeng](https://github.com/xuhongming251/ComfyUI-Jimeng)，并在其基础上进行了功能开发和修改。感谢原作者 `xuhongming251` 的工作。
