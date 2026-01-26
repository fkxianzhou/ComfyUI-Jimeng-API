### Jimeng API Client
加载并初始化火山方舟 (Volcengine Ark) API 客户端。

*   **Node ID**: `JimengAPIClient`
*   **Python Class**: `JimengAPIClient`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `key_name` | `COMBO` | 是 | - | 从 `api_keys.json` 中选择 API 密钥配置名称。 |

#### 输出 (Outputs)

| 输出名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 已初始化的 API 客户端实例，供其他节点使用。 |

---