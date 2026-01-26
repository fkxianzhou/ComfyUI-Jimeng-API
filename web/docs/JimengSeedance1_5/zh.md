### Jimeng Seedance 1.5 Pro
高级视频生成节点，支持音频和智能时长。

*   **Node ID**: `JimengSeedance1_5`
*   **Python Class**: `JimengSeedance1_5`

#### 输入 (Inputs)

除了包含 Seedance 1.0 的大部分参数外，还包含：

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `generate_audio` | `BOOLEAN` | 是 | True | 是否同步生成音频。 |
| `auto_duration` | `BOOLEAN` | 是 | False | 是否自动决定时长 (忽略 `duration` 参数)。 |
| `draft_mode` | `BOOLEAN` | 是 | False | 是否为草稿模式 (快速预览)。 |
| `reuse_last_draft_task` | `BOOLEAN` | 是 | False | 是否复用上一次生成的样片任务 ID。 |
| `draft_task_id` | `STRING` | 否 | "" | 用于基于草稿继续生成的任务 ID。 |

#### 输出 (Outputs)

同 Seedance 1.0。