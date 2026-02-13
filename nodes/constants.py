# Jimeng API 常量与多语言配置

# 通用常量
MAX_SEED = 2147483647
MIN_SEED = -1
DEFAULT_GUIDANCE_SCALE = 5.0
MAX_GENERATION_COUNT = 2048

# 图像分辨率常量
MIN_IMAGE_PIXELS_DEFAULT = 512 * 512
MAX_IMAGE_PIXELS_DEFAULT = 2048 * 2048
MIN_IMAGE_PIXELS_V4_5 = 3686400  # 2560x1440
MAX_IMAGE_PIXELS_V4 = 4096 * 4096
MIN_ASPECT_RATIO = 1.0 / 16.0
MAX_ASPECT_RATIO = 16.0

# 视频常量
VIDEO_MAX_SEED = 4294967295
VIDEO_DEFAULT_TIMEOUT = 172800
VIDEO_MIN_TIMEOUT = 3600
VIDEO_MAX_TIMEOUT = 259200
VIDEO_FRAME_RATE = 24.0
VIDEO_MIN_FRAMES = 29
VIDEO_MAX_FRAMES = 289
VIDEO_FRAME_STEP = 4.0
VIDEO_BASE_FRAMES = 25.0
VIDEO_RESOLUTIONS = ["480p", "720p", "1080p"]
# 视频分辨率对应的像素总量估算
VIDEO_RESOLUTION_PIXELS = {
    "480p": 409920,
    "720p": 921600,
    "1080p": 2073600,
}
DEFAULT_FILENAME_PREFIX = "Jimeng/Video/Batch/Seedance"

LOG_TRANSLATIONS = {
    "zh": {
        # 初始化/依赖检查相关
        "init_sdk_ver_low": "当前 SDK 版本 {current} 过低，正在自动更新至 >= {min}...",
        "init_sdk_update_ok": "SDK 更新完成。",
        "init_sdk_not_found": "未检测到 {pkg},正在自动安装...",
        "init_sdk_install_ok": "SDK 安装完成。",
        "init_sdk_install_fail": "自动安装失败，请手动安装: {e}",
        "init_dep_check_err": "依赖检查出错: {e}",
        
        # 控制台信息
        "api_file_not_found": "提示: 未找到 API 密钥文件。请将 'api_keys.json.example' 重命名为 'api_keys.json' 并填入您的密钥。",
        "api_file_empty": "警告: 'api_keys.json' 为空或格式不正确。",
        "api_load_error": "错误: 加载 'api_keys.json' 失败: {e}",
        "api_key_not_found": "错误: 未找到 '{key_name}' 的 API Key，请检查配置文件。",
        "est_fallback": "默认兜底",
        "est_history": "历史均值",
        "est_regression": "线性回归",
        "est_recent": "近期负载调整",
        "task_submitted_est": "任务已提交。预估生成时间: {time}s (预估方式: {method})",
        "task_info_simple": "任务 ID: {task_id} | 模型: {model}",
        "batch_submit_start": "正在提交 {count} 个任务...",
        "batch_submit_result": "提交完成。成功: {created}，失败: {failed}。",
        "batch_failed_summary": "⚠️ {count} 个任务创建失败。原因汇总:",
        "batch_failed_reason": "  - {msg}: {count} 次",
        
        # 轮询提示
        "polling_single": "任务 {task_id}: 运行中... 已耗时 {elapsed}s / 预估 {max}s",
        "polling_single_waiting": "任务排队等待资源中... (状态: {status})",
        "polling_batch_stats": "批量进度: 完成 {done}/{total}。等待中: {pending}... (耗时 {elapsed}s / 预估 {max}s) [运行中: {running}，排队: {queued}]",

        "interrupted": "\n" + "用户中断了处理。正在尝试取消挂起任务...",
        "cancel_task_success": "已成功取消任务: {task_id}",
        "cancel_task_failed": "取消任务 {task_id} 失败: {msg}",
        "cancel_batch_summary": "批量取消完成。成功取消: {success} 个，取消失败: {failed} 个。",
        "cancel_batch_reason": "  - {msg}",
        "task_finished_single": "任务执行成功。",
        "batch_finished_stats": "批量处理完成。成功: {success}，失败: {failed}。",
        "batch_handling": "正在处理 {count} 个成功任务，按 Seed 排序并下载...",
        "batch_copying": "正在保存文件到输出目录: {path}",
        "err_download_url": "异步下载或转换失败,URL: {url}，错误: {e}",
        "err_task_create": "任务创建失败: {e}",
        "err_task_check": "检查任务状态失败 ({tid}): {e}",
        "err_task_fail_msg": "任务 {tid} 失败: {msg}",
        "err_batch_fail_all": "批量任务失败: 无任务成功。",
        "err_copy_fail": "复制文件失败: {path}，错误: {e}",
        "err_convert_tensor": "转换图片 Tensor 失败: {e}",
        "err_check_status_batch": "检查非阻塞任务状态失败: {e}",
        "download_retry": "警告: 下载失败 (尝试 {attempt}/{total})。{delay}秒后重试... 错误: {e}",
        
        # 流式生成
        "stream_recv_image": "流式接收: 第 {index} 张图片生成完毕，正在后台下载...",
        "stream_partial_fail": "流式警告: 第 {index} 张图片生成失败: {msg}",

        # 弹窗提示
        "popup_req_failed": "请求失败: {msg}",
        "popup_task_failed": "任务 {task_id} 失败: {msg}",
        "popup_batch_pending": "批量任务 ({count} 个) 处理中。请再次运行以检查结果。",
        "popup_task_pending": "任务 {task_id} 状态为 {status}。请再次运行以检查结果。",
        "popup_param_not_allowed": "参数错误: 提示词中不允许包含参数 '--{param}'。请使用节点的组件进行设置。",
        "popup_first_frame_missing": "参数错误: 使用尾帧图片时必须提供首帧图片。",
        "popup_ref_missing": "参数错误: 必须提供至少一张参考图。",
        "popup_prepare_failed": "任务准备失败: {e}",

        # 图片节点
        "err_pixels_range": "参数错误: 总像素数必须在 {min} ({min_desc}) 和 {max} ({max_desc}) 之间。当前值: {current}",
        "err_aspect_ratio": "参数错误: 宽高比必须在 {min} 和 {max} 之间。当前值: {current}",
        "err_download_img": "错误: 下载生成的图像失败。",
        "err_gen_model": "模型 {model} 生成失败: {e}",
        "err_img_limit_10": "参数错误: 输入图像数量不能超过 10 张。",
        "err_img_limit_15": "参数错误: 输入图像数 ({n}) 与最大生成数 ({max}) 之和不能超过 15。",
        "err_img_limit_group_15": "参数错误: 在组图模式下，输入参考图数量 ({n}) 加 生成图片数量 ({max}) 的总和 ({total}) 不能超过 15 张。",

        # Keys
        "popup_key_valid_err": "配置错误: 所选密钥 '{key}' 无效或未找到。请检查 api_keys.json。",
        "err_new_key_empty": "配置错误: 已启用手动输入，但 API Key 为空。",
        "err_new_key_invalid": "鉴权失败: 输入的 API Key 无效，请检查 Key 是否正确。",
        "info_new_key_saved": "提示: 新密钥 '{name}' 已通过鉴权并保存到 api_keys.json。",

        "quota_exceeded": "配额限制: 模型 {model} 的使用量已达到上限 ({used}/{limit})。预计消耗: {estimated}。限制已自动解除，请重新运行或设置新的配额。",
        "quota_update_failed": "警告: 更新配额用量失败: {e}",
        "quota_set_log": "设置配额: {model} -> {limit} ({type})",
        "quota_update_log": "更新用量: {model} +{cost} (总计: {total})",
        
        # API 错误码映射
        "api_errors": {
            "AuthenticationError": "API 鉴权失败。请检查 api_keys.json 文件：1. API Key 是否填写正确；2. JSON 格式是否因编辑而损坏。",
            "AccessDenied": "没有访问该资源的权限，请检查权限设置，或联系管理员添加白名单。",
            "AccountOverdueError": "当前账号欠费（余额<0），如需继续调用，请前往 https://console.volcengine.com/finance/fund/recharge 进行充值，详细操作参见 https://www.volcengine.com/docs/6269/100434 。",
            "ServiceOverdue": "您的账单已逾期，不支持该操作。请前往火山费用中心充值。",
            "ServiceNotOpen": "模型服务不可用，不支持该操作。请前往火山方舟控制台激活模型服务，或提交工单联系我们。",
            "ModelNotOpen": "当前账号暂未开通对应模型服务，请前往火山方舟控制台开通管理页开通对应模型服务。",
            "TaskRunningCannotCancel": "任务正在运行中，不支持取消操作。",
            "RateLimitExceeded": "请求超出 RPM/TPM 限制，请稍后重试。",
            "QuotaExceeded": "当前账号免费试用额度已消耗完毕，或排队任务数已超限，请前往控制台查看。",
            "ServerOverloaded": "服务资源紧张，请您稍后重试。常出现在调用流量突增或刚开始调用长时间未使用的推理接入点。",
            "InternalServiceError": "内部系统异常，请您稍后重试。",
            "SensitiveContentDetected": "输入文本可能包含敏感信息，请您使用其他 prompt。",
            "InputTextSensitiveContentDetected": "输入文本可能包含敏感信息，请您更换后重试。",
            "InputImageSensitiveContentDetected": "输入图像可能包含敏感信息，请您更换后重试。",
            "OutputTextSensitiveContentDetected": "生成的文字可能包含敏感信息，请您更换输入内容后重试。",
            "OutputImageSensitiveContentDetected": "生成的图像可能包含敏感信息，请您更换输入内容后重试。",
            "OutputVideoSensitiveContentDetected": "生成的视频可能包含敏感信息，请您更换输入内容后重试。",
            "InvalidImageURL": "无法解析或处理图片，可能是 Base64 格式不正确、图片数据损坏或格式不支持。",
            "InvalidImageDetail": "image_url 中的 detail 参数值无效，只接受 'auto', 'high', 'low'。",
            "MissingParameter": "请求缺少必要参数，请查阅 API 文档。",
            "InvalidParameter": "请求包含非法参数，请查阅 API 文档。",
            "LastFrameNotSupported": "当前模型不支持尾帧控制，请移除尾帧图片或更换支持的模型。",
            "RefImageNotSupported": "当前模型不支持该类型的参考图输入。",
            "PromptEmpty": "提示词不能为空，请输入提示词。",
        }
    },
    "en": {
        "init_sdk_ver_low": "Current SDK version {current} is too low. Auto-updating to >= {min}...",
        "init_sdk_update_ok": "SDK update completed.",
        "init_sdk_not_found": "Package {pkg} not found. Installing automatically...",
        "init_sdk_install_ok": "SDK installation completed.",
        "init_sdk_install_fail": "Auto-install failed. Please install manually: {e}",
        "init_dep_check_err": "Dependency check error: {e}",
        "api_file_not_found": "Info: API keys file not found. Please rename 'api_keys.json.example' to 'api_keys.json' and fill in your keys.",
        "api_file_empty": "Warning: 'api_keys.json' is empty or not formatted correctly.",
        "api_load_error": "Error: Failed to load 'api_keys.json': {e}",
        "api_key_not_found": "Error: API Key for '{key_name}' not found.",
        "est_fallback": "Fallback Default",
        "est_history": "History Average",
        "est_regression": "Linear Regression",
        "est_recent": "Recent Load Adjustment",
        "task_submitted_est": "Task submitted. Est. time: {time}s (Method: {method})",
        "task_info_simple": "Task ID: {task_id} | Model: {model}",
        "batch_submit_start": "Submitting batch of {count} tasks (Model: {model})...",
        "batch_submit_result": "Submission complete. Created: {created}, Failed: {failed}.",
        "batch_failed_summary": "⚠️ {count} tasks failed to create. Reason summary:",
        "batch_failed_reason": "  - {msg}: {count} times",
        
        # Updated
        "polling_single": "Task {task_id}: Running... {elapsed}s / {max}s elapsed",
        "polling_single_waiting": "Task {task_id}: Queued and waiting for resources... (Status: {status})",
        "polling_batch_stats": "Batch Progress: {done}/{total} done. {pending} pending... (Elapsed {elapsed}s / {max}s) [Run: {running}, Queue: {queued}]",

        "interrupted": "\n" + "Processing interrupted by user. Cancelling pending tasks...",
        "cancel_task_success": "Successfully cancelled task: {task_id}",
        "cancel_task_failed": "Failed to cancel task {task_id}: {msg}",
        "cancel_batch_summary": "Batch cancellation finished. Success: {success}, Failed: {failed}.",
        "cancel_batch_reason": "  - {msg}",
        "task_finished_single": "Task completed successfully.",
        "batch_finished_stats": "Batch finished. Success: {success}, Failed: {failed}.",
        "batch_handling": "Handling {count} successful tasks. Sorting by seed and downloading...",
        "batch_copying": "Copying files to output directory: {path}",
        "err_download_url": "Async download failed, URL: {url}, Error: {e}",
        "err_task_create": "Task creation failed: {e}",
        "err_task_check": "Failed to check status for {tid}: {e}",
        "err_task_fail_msg": "Task {tid} failed: {msg}",
        "err_batch_fail_all": "Batch failed: No tasks succeeded.",
        "err_copy_fail": "Failed to copy file: {path}. Error: {e}",
        "err_convert_tensor": "Failed to convert frame to tensor: {e}",
        "err_check_status_batch": "API Error checking batch status: {e}",
        "download_retry": "Warning: Download failed (Attempt {attempt}/{total}). Retrying in {delay}s... Error: {e}",
        "stream_recv_image": "Streaming: Image {index} generated, downloading in background...",
        "stream_partial_fail": "Streaming Warning: Image {index} failed: {msg}",
        "popup_req_failed": "Request failed: {msg}",
        "popup_task_failed": "Task {task_id} failed: {msg}",
        "popup_batch_pending": "Batch ({count} tasks) is pending. Run again to check results.",
        "popup_task_pending": "Task {task_id} is {status}. Run again to check results.",
        "popup_param_not_allowed": "Parameter Error: Parameter '--{param}' is not allowed in the prompt. Please use the node's widget for this value.",
        "popup_first_frame_missing": "Parameter Error: A first frame image must be provided when using a last frame image.",
        "popup_ref_missing": "Parameter Error: At least one reference image must be provided.",
        "popup_prepare_failed": "Failed to prepare task: {e}",
        "err_pixels_range": "Parameter Error: Total pixels must be between {min} ({min_desc}) and {max} ({max_desc}). Your current: {current}",
        "err_aspect_ratio": "Parameter Error: Aspect ratio must be between {min} and {max}. Your current: {current}",
        "err_download_img": "Error: Failed to download the generated image.",
        "err_gen_model": "Failed to generate image with model {model}: {e}",
        "err_img_limit_10": "Parameter Error: The number of input images cannot exceed 10.",
        "err_img_limit_15": "Parameter Error: The sum of input images ({n}) and max generated images ({max}) cannot exceed 15.",
        "err_img_limit_group_15": "Parameter Error: The sum of input images ({n}) and max generated images ({max}) cannot exceed 15 in group mode (Total: {total}).",
        "popup_key_valid_err": "Config Error: Selected key '{key}' is invalid or not found. Please check api_keys.json.",
        "err_new_key_empty": "Config Error: Manual entry enabled but API Key is empty.",
        "err_new_key_invalid": "Auth Failed: Input API Key is invalid. Connection rejected by server.",
        "info_new_key_saved": "Info: New key '{name}' verified and saved to api_keys.json.",
        "quota_exceeded": "Quota Exceeded: Usage limit for model {model} reached ({used}/{limit}). Estimated cost: {estimated}. Limit has been automatically removed. Please run again or set a new quota.",
        "quota_update_failed": "Warning: Failed to update quota usage: {e}",
        "quota_set_log": "Set quota for {model}: {limit} ({type})",
        "quota_update_log": "Updated usage for {model}: +{cost} (Total: {total})",
        "api_errors": {
            "AuthenticationError": "Invalid API Key (401). Please check api_keys.json.",
            "AccessDenied": "Access Denied (403). No permission or IP whitelist issue.",
            "AccountOverdueError": "Account Overdue (403). Please recharge your account.",
            "ServiceOverdue": "Service Overdue (403). Please recharge.",
            "ServiceNotOpen": "Service Not Open (403). Please activate the model service.",
            "ModelNotOpen": "Model Not Open (404). Please activate the model.",
            "TaskRunningCannotCancel": "Task is currently running and cannot be cancelled (409).",
            "RateLimitExceeded": "Rate Limit Exceeded (429). Please try again later.",
            "QuotaExceeded": "Quota Exceeded (429). Free trial or concurrency limit reached.",
            "ServerOverloaded": "Server Overloaded (429). Please try again later.",
            "InternalServiceError": "Internal Service Error (500). Please try again later.",
            "SensitiveContentDetected": "Sensitive Content Detected (400). Please change your prompt.",
            "InputTextSensitiveContentDetected": "Input text contains sensitive content (400).",
            "InputImageSensitiveContentDetected": "Input image contains sensitive content (400).",
            "OutputTextSensitiveContentDetected": "Generated text contains sensitive content (400).",
            "OutputImageSensitiveContentDetected": "Generated image contains sensitive content (400).",
            "OutputVideoSensitiveContentDetected": "Generated video contains sensitive content (400).",
            "InvalidImageURL": "Invalid Image URL (400).",
            "InvalidImageDetail": "Invalid Image Detail parameter (400).",
            "MissingParameter": "Missing Parameter (400).",
            "InvalidParameter": "Invalid Parameter (400).",
            "LastFrameNotSupported": "Last frame input is not supported by this model. Please remove it or use the Pro model.",
            "RefImageNotSupported": "Reference image input is not supported by this model.",
            "PromptEmpty": "Prompt cannot be empty (400).",
        }
    }
}

ERROR_TEXT_MATCH_RULES = {
    "output image may contain sensitive information": "OutputImageSensitiveContentDetected",
    "input text may contain sensitive information": "InputTextSensitiveContentDetected",
    "input image may contain sensitive information": "InputImageSensitiveContentDetected", 
    "output video may contain sensitive information": "OutputVideoSensitiveContentDetected",
    "generated text contains sensitive content": "OutputTextSensitiveContentDetected",
    "API key or AK/SK in the request is missing or invalid": "AuthenticationError",
    "account has an overdue balance": "AccountOverdueError",
    "exceeded the quota": "QuotaExceeded",
    "limit of the associated endpoint": "RateLimitExceeded",
    "Request failed because it is missing": "MissingParameter",
    "parameters specified in the request are not valid": "InvalidParameter",
    "not permitted to access": "AccessDenied",
    "service is unavailable": "ServiceNotOpen",
    "does not support last frame image": "LastFrameNotSupported",
    "does not support reference image": "RefImageNotSupported",
    "prompt cannot be empty": "PromptEmpty",
    "text content must contain a prompt description": "PromptEmpty",
    "because it is currently running": "TaskRunningCannotCancel",
}