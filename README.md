<div align="center">
  <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.74.0/files/dark/jimeng-color.png" width="120" />
</div>

# ComfyUI 即梦 API 节点

本项目为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 提供了火山方舟的视觉模型（即梦/豆包） API 节点。用户可以通过这些节点在 ComfyUI 中使用多种图像生成和视频生成功能。

- 项目已支持 `Seedance 2.0`、`Seedance 2.0 Fast`、`Seedance 2.0 Mini` 与 `Seedance 2.5`；2.0 标准版最高支持 4K，2.5 最长支持 30 秒。
- 如在使用过程中遇到问题，请通过 [ISSUES](https://github.com/fkxianzhou/ComfyUI-Jimeng-API/issues) 反馈。
- Classic Canvas 与 Nodes 2.0（Vue）均受支持，最低支持 ComfyUI `0.25.1`。

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

1. 在插件根目录中找到 `api_keys.json.example` 文件。
2. 将其**重命名**为 `api_keys.json`。
3. 打开文件并填入您的密钥信息（[从此获取 Key](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)）。

### 方式 2：节点内配置

1. 在 ComfyUI 中添加 **Jimeng API Client** 节点。
2. 在 `key_name` 下拉框中选择 **Custom**。
3. 在弹出的输入框中填入您的 API Key。
4. （可选）在 `new_key_name` 中填入一个名称（如 "MyKey"），运行一次后该 Key 将被自动保存。
   - *注意：保存后需刷新浏览器页面，新密钥才会显示在下拉列表中。*

## 📖 功能节点列表

(所有节点均位于 `JimengAI` 菜单下)

- **基础设置**:
  - `火山方舟 API 客户端`: **(必须)** 用于加载并创建一个可供其他节点使用的客户端实例。
  - `Jimeng 配额设置`: 用于设置图像和视频生成的配额限制，防止意外消耗过多资源。
- **图像生成**:
  - `图像生成（Seedream 3）`: 基础图像生成节点。
  - `图像生成（Seedream 4）`: 高级图像生成节点，支持多图输入、**组图生成**以及 4.5 模型。
  - `图像生成（Seedream 5）`: 支持 Seedream 5 Pro/Lite；Pro 提供提示词优化，Lite 支持组图与**联网搜索**。
- **视频生成**:
  - `视频生成（Seedance 1.0）`: 核心视频生成节点，支持文生视频、图生视频（首/尾帧）。
  - `视频生成（Seedance 1.5 Pro）`: 支持**音频生成**和**智能时长**的高级视频生成节点。
  - `视频生成（Seedance 2 / 2.5）`: 支持**图片/视频/音频多模态参考**、**视频编辑/延长**与**联网搜索增强**。
  - `视频生成（参考图生视频）`: 根据 1-4 张**参考图像**生成视频。
  - `视频生成任务列表查询`: 用于查询和管理在 API 上运行的任务历史。
- **视觉理解**:
  - `视觉理解（Visual Understanding）`: 默认使用 Seed 2.1 Pro，并保留 Seed 2.0 Pro/Lite/Mini 兼容旧工作流。

## 📑 节点详解

### `火山方舟 API 客户端 (Jimeng API Client)`

加载 `api_keys.json` 中的密钥配置。这是所有工作流的起点。

- **输入**: `密钥名称` (在 JSON 中配置的 customName)。
- **输出**: `客户端` 实例。

### `Jimeng 配额设置 (Jimeng Quota Settings)`

允许为当前客户端设置图像（张数）和视频（Tokens）的使用上限。

- **特性**: 当达到限额时自动停止任务并抛出提示，防止额度超支。
- **示例工作流**（点击预览图打开 JSON）：

  [![Quota Settings Workflow](./example_workflows/QuotaSettings.jpg)](./example_workflows/QuotaSettings.json)

***

### `图像生成（Seedream 4）`

支持 `doubao-seedream-4.5` 与 `doubao-seedream-4.0`。

- **输入图像**: 支持单张或多张（Batch）图像作为参考。
- **启用组图生成**: 开启后可一次性生成多张内容关联的图片。
- **提示词优化**: Seedream 4.0 可通过开关启用；Seedream 4.5 不发送此参数。

**示例工作流**（点击预览图打开 JSON）：

[![Seedream 4 Workflow](./example_workflows/Seedream%204.jpg)](./example_workflows/Seedream%204.json)

***

### `图像生成（Seedream 5）`

支持 `doubao-seedream-5.0-pro` 与 `doubao-seedream-5.0-lite`。

- **Pro**: 最多 10 张参考图；保留种子与水印；默认开启提示词优化，使用参考图时必须开启；输出通过 URL 下载。
- **Lite**: 保留流式 Base64、组图生成、联网搜索与种子功能。
- **自定义尺寸**: Pro 要求宽高为 16 的倍数、比例在 1:16–16:1、总像素为 921600–4194304。

**示例工作流**（点击预览图打开 JSON）：

[![Seedream 5 Workflow](./example_workflows/Seedream%205.jpg)](./example_workflows/Seedream%205.json)

***

### `视频生成（Seedance 1.0/1.5 Pro）`

支持文生视频与首/尾帧图生视频；在 1.0 能力基础上，1.5 Pro 支持**音效生成**与**智能时长**控制。

**示例工作流**（点击预览图打开 JSON）：

[![Seedance 1 Workflow](./example_workflows/Seedance%201.jpg)](./example_workflows/Seedance%201.json)

### `视频生成（Seedance 2 / 2.5）`

可覆盖文生视频、多模态参考生视频、视频编辑、视频延长与联网搜索增强。

- **Seedance 2.5**: 480p / 720p、4–30 秒；最多 30 张参考图、10 段参考视频和 10 段参考音频，并允许纯音频参考。
- **Seedance 2.0 标准版**: 480p / 720p / 1080p / 4K、4–15 秒；最多 9 张参考图、3 段参考视频和 3 段参考音频。
- **Seedance 2.0 Fast、Mini**: 480p / 720p、4–15 秒；参考数量限制与 2.0 标准版相同。
- **参考视频**: 最大 200 MB、409600–8295044 像素、24–60 FPS；可识别编码元数据时要求 H.264/H.265 视频与 AAC/MP3 音频。
- **参考媒体时长**: Seedance 2.5 单个及同类素材总时长上限为 30.2 秒；Seedance 2.0 系列为 15.2 秒。
- **请求大小**: 最终紧凑 UTF-8 JSON 请求体不得超过 64 MiB。

**示例工作流**（点击预览图打开 JSON）：

[![Seedance 2 Workflow](./example_workflows/Seedance%202.jpg)](./example_workflows/Seedance%202.json)

***

### `视频生成任务列表查询`

支持按状态、模型版本或任务 ID 过滤查询任务历史。

***

### `视觉理解（Seed 2.1 / 2.0）`

默认使用 `doubao-seed-2-1-pro`，并保留 `doubao-seed-2.0` 系列模型。

- **多模态输入**: 支持上传图片或视频进行理解和问答。
- **多轮对话**: 支持开启多轮对话模式，保持上下文。
- **深度思考**: 支持开启深度思考模式，提升复杂问题的推理能力。

**示例工作流**（点击预览图打开 JSON）：

[![Visual Understanding Workflow](./example_workflows/VisualUnderstanding.jpg)](./example_workflows/VisualUnderstanding.json)

## 📓 示例工作流

您可以在 `example_workflows` 目录中找到所有节点的示例工作流。

## 🧩 ComfyUI 兼容性

| ComfyUI / 前端 | Classic Canvas | Nodes 2.0 | 说明 |
|---|---:|---:|---|
| 0.25.1 / 1.45.15 | 支持 | 支持 | 最低支持版本；包含旧平铺工作流迁移 |
| 0.28.0 / 1.45.21 | 支持 | 支持 | 官方稳定分支目标 |
| 前端 1.46.3+ | 支持 | 支持 | 已覆盖 DynamicCombo 保存与恢复路径 |
