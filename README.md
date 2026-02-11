# ComfyUI 即梦 AI 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了火山方舟的视觉模型（即梦/豆包） API 节点。用户可以通过这些节点在 ComfyUI 中使用多种图像生成和视频生成功能。

*项目目前已重构为 V3 架构，可能存在未知的 BUG。如在使用过程中遇到问题，请通过 [ISSUES](https://github.com/fkxianzhou/ComfyUI-Jimeng-API/issues) 反馈。

*暂不支持 ComfyUI 的 Nodes 2.0 新版 UI，如遇到显示异常：
 1.  点击左侧/顶部 ComfyUI 图标 打开菜单。
 2.  找到并关闭 Nodes 2.0 选项，切换回传统界面。

## ✨ 项目特性

- **多 Key 管理**：支持在配置文件中设置多个 API Key，并在节点中灵活切换，方便管理不同账户或配额。
- **异步与并发**：所有核心节点均支持任务的异步提交和并发生成，无需阻塞队列，大幅提升批量生成效率。
- **友好交互**：提供清晰的控制台进度提示和完善的异常处理机制，报错信息直观，便于快速排查问题。

## 📦 安装

方式1：  **克隆仓库**:
    打开终端，`cd` 到 ComfyUI 的 `custom_nodes` 目录，运行：
    ```bash
    git clone https://github.com/fkxianzhou/ComfyUI-Jimeng-API
    ```

方式2： **使用ComfyUI Manager下载**。

## ⚙️ 设置：配置 API 密钥

### 方式 1：手动配置

1.  在插件根目录中找到 `api_keys.json.example` 文件。
2.  将其**重命名**为 `api_keys.json`。
3.  打开文件并填入您的密钥信息（[从此获取 Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）。

### 方式 2：节点内配置

1.  在 ComfyUI 中添加 **Jimeng API Client** 节点。
2.  在 `key_name` 下拉框中选择 **Custom**。
3.  在弹出的输入框中填入您的 API Key。
4.  （可选）在 `new_key_name` 中填入一个名称（如 "MyKey"），运行一次后该 Key 将被自动保存。
    *   *注意：保存后需刷新浏览器页面，新密钥才会显示在下拉列表中。*

## 📖 功能节点列表

(所有节点均位于 `JimengAI` 菜单下)

- **基础设置**:
  - `火山方舟 API 客户端`: **(必须)** 用于加载并创建一个可供其他节点使用的客户端实例。
- **图像生成**:
  - `图像生成（Seedream 3）`: 支持**文生图**和**图生图**模式的基础生成节点。
  - `图像生成（Seedream 4）`: 高级图像生成节点，支持**组图生成**、多图输入及最新的 4.5 模型。
- **视频生成**:
  - `视频生成（Seedance 1.0）`: 核心视频生成节点，支持文生视频、图生视频（首/尾帧）。
  - `视频生成（Seedance 1.5 Pro）`: 支持**音频生成**和**智能时长**的高级视频生成节点。
  - `参考图生视频（Seedance 1.0 lite）`: 根据 1-4 张**参考图像**生成视频。
  - `视频生成任务列表查询`: 用于查询和管理在 API 上运行的任务历史。

## 📑 节点详解

### `火山方舟 API 客户端 (Jimeng API Client)`
加载 `api_keys.json` 中的密钥配置。这是所有工作流的起点。
- **输入**: `密钥名称` (在 JSON 中配置的 customName)。
- **输出**: `客户端` 实例。

---

### `图像生成（Seedream 3）`
- **模式**: 不连接 `输入图像` 为文生图，连接则为图生图。
- **特性**: 支持自定义尺寸，支持引导系数调节。

**示例工作流**:
![Seedream 3 Workflow](example_workflows/Seedream%203.jpg)

---

### `图像生成（Seedream 4）`
支持最新的 `doubao-seedream-4.5`。
- **启用组图生成**: 开启后可一次性生成多张内容关联的图片。
- **输入图像**: 支持单张或多张（Batch）图像作为参考。

**示例工作流**:
![Seedream 4 Workflow](example_workflows/Seedream%204.jpg)

---

### `视频生成（Seedance 1.0 / 1.5 Pro）`
- **1.0 特性**: 支持首/尾帧视频生成，支持非阻塞异步模式。
- **1.5 Pro 特性**: 支持**音效生成**和**智能时长**控制。

### `参考图生视频（Seedance 1.0 lite）`
支持上传 1-4 张图像作为风格或内容参考生成视频。

### `视频生成任务列表查询`
支持按状态、模型版本或任务 ID 过滤查询任务历史。

**示例工作流**:
![Seedance 1 Workflow](example_workflows/Seedance%201.jpg)

## 📓 示例工作流


您可以在 `example_workflows` 目录中找到所有节点的示例工作流。
