### Jimeng Video Query Tasks
查询历史视频生成任务的状态和结果。

*   **Node ID**: `JimengVideoQueryTasks`
*   **Python Class**: `JimengVideoQueryTasks`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 是 | - | API 客户端。 |
| `page_num` | `INT` | 是 | 1 | 页码。 |
| `page_size` | `INT` | 是 | 10 | 每页数量。 |
| `status` | `COMBO` | 是 | "all" | 过滤任务状态 (succeeded, failed, running 等)。 |
| `service_tier` | `COMBO` | 是 | "default" | 服务等级 (default/flex)。 |
| `task_ids` | `STRING` | 否 | "" | 指定任务 ID 列表 (每行一个)。 |
| `model_version` | `COMBO` | 是 | "all" | 过滤模型版本。 |

#### 输出 (Outputs)

| 输出名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `task_list_json` | `STRING` | 任务列表的 JSON 字符串。 |
| `total_tasks` | `INT` | 符合查询条件的任务总数。 |