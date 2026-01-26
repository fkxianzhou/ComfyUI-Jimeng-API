### Jimeng Reference to Video
参考图生视频节点 (Seedance 1.0 lite)。

*   **Node ID**: `JimengReferenceImage2Video`
*   **Python Class**: `JimengReferenceImage2Video`

#### 输入 (Inputs)

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `JIMENG_CLIENT` | 是 | - | API 客户端。 |
| `prompt` | `STRING` | 是 | "" | 提示词。 |
| `ref_image_1` | `IMAGE` | 否 | - | 参考图像 1 (至少需要一张参考图)。 |
| `ref_image_2` | `IMAGE` | 否 | - | 参考图像 2。 |
| `ref_image_3` | `IMAGE` | 否 | - | 参考图像 3。 |
| `ref_image_4` | `IMAGE` | 否 | - | 参考图像 4。 |
| ... | ... | ... | ... | 其他通用视频参数 (duration, resolution, seed 等)。 |

#### 输出 (Outputs)

同 Seedance 1.0。