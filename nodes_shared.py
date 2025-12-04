import os
import io
import base64
import locale
import json
import asyncio
import aiohttp
import numpy
import PIL.Image
import torch
from openai import AsyncOpenAI
from volcenginesdkarkruntime import Ark

GLOBAL_CATEGORY = "JimengAI"

DEFAULT_DOWNLOAD_TIMEOUT = 60  # 下载超时时间（秒）
DEFAULT_DOWNLOAD_RETRIES = 3   # 下载重试次数

jimeng_api_dir = os.path.dirname(os.path.abspath(__file__))
API_KEYS_FILE = os.path.join(jimeng_api_dir, "api_keys.json")

API_KEYS_CONFIG = []
CURRENT_LANG = "en"

LOG_TRANSLATIONS = {
    "zh": {
        "api_file_not_found": "[JimengAI] 提示: 未找到 API 密钥文件。请将 'api_keys.json.example' 重命名为 'api_keys.json' 并填入您的密钥。",
        "api_file_empty": "[JimengAI] 警告: 'api_keys.json' 为空或格式不正确。",
        "api_load_error": "[JimengAI] 错误: 加载 'api_keys.json' 失败: {e}",
        "api_key_not_found": "[JimengAI] 错误: 未找到 '{key_name}' 的 API Key，请检查配置文件。",
        "est_fallback": "默认兜底",
        "est_history": "历史均值",
        "est_regression": "线性回归",
        "est_recent": "近期负载调整",
        "task_submitted_est": "[JimengAI] 任务已提交。预估生成时间: {time}s (预估方式: {method})",
        "task_info_simple": "[JimengAI] 任务 ID: {task_id} | 模型: {model}",
        "batch_submit_start": "[JimengAI] 正在提交 {count} 个任务 (模型: {model})...",
        "batch_submit_result": "[JimengAI] 提交完成。成功: {created}，失败: {failed}。",
        "polling_single": "[JimengAI] 任务 {task_id}: 已耗时 {elapsed}s / 预估 {max}s ...",
        "polling_batch": "[JimengAI] 批量进度: 完成 {done}/{total}。等待中: {pending}... (耗时 {elapsed}s / {max}s)",
        "interrupted": "\n[JimengAI] 用户中断了处理。正在尝试取消挂起任务...",
        "task_finished_single": "[JimengAI] 任务执行成功。",
        "batch_finished_stats": "[JimengAI] 批量处理完成。成功: {success}，失败: {failed}。",
        "batch_handling": "[JimengAI] 正在处理 {count} 个成功任务，按 Seed 排序并下载...",
        "batch_copying": "[JimengAI] 正在保存文件到输出目录: {path}",
        "err_download_url": "[JimengAI] 异步下载或转换失败，URL: {url}，错误: {e}",
        "err_task_create": "[JimengAI] 任务创建失败: {e}",
        "err_task_check": "[JimengAI] 检查任务状态失败 ({tid}): {e}",
        "err_task_fail_msg": "[JimengAI] 任务 {tid} 失败: {msg}",
        "err_batch_fail_all": "[JimengAI] 批量任务失败: 无任务成功。",
        "err_copy_fail": "[JimengAI] 复制文件失败: {path}，错误: {e}",
        "err_convert_tensor": "[JimengAI] 转换图片 Tensor 失败: {e}",
        "err_check_status_batch": "[JimengAI] 检查非阻塞任务状态失败: {e}",
        "download_retry": "[JimengAI] 警告: 下载失败 (尝试 {attempt}/{total})。{delay}秒后重试... 错误: {e}",
        
        # 前端弹窗提示
        "popup_req_failed": "[JimengAI] 请求失败: {msg}",
        "popup_task_failed": "[JimengAI] 任务 {task_id} 失败: {msg}",
        "popup_batch_pending": "[JimengAI] 批量任务 ({count} 个) 处理中。请再次运行以检查结果。",
        "popup_task_pending": "[JimengAI] 任务 {task_id} 状态为 {status}。请再次运行以检查结果。",
        "popup_param_not_allowed": "[JimengAI] 参数错误: 提示词中不允许包含参数 '--{param}'。请使用节点的组件进行设置。",
        "popup_first_frame_missing": "[JimengAI] 参数错误: 使用尾帧图片时必须提供首帧图片。",
        "popup_ref_missing": "[JimengAI] 参数错误: 必须提供至少一张参考图。",
        "popup_prepare_failed": "[JimengAI] 任务准备失败: {e}",

        # 图片节点提示
        "err_pixels_range": "[JimengAI] 参数错误: 总像素数必须在 {min} ({min_desc}) 和 {max} ({max_desc}) 之间。当前值: {current}",
        "err_aspect_ratio": "[JimengAI] 参数错误: 宽高比必须在 {min} 和 {max} 之间。当前值: {current}",
        "err_download_img": "[JimengAI] 错误: 下载生成的图像失败。",
        "err_gen_model": "[JimengAI] 模型 {model} 生成失败: {e}",
        "err_img_limit_10": "[JimengAI] 参数错误: 输入图像数量不能超过 10 张。",
        "err_img_limit_15": "[JimengAI] 参数错误: 输入图像数 ({n}) 与最大生成数 ({max}) 之和不能超过 15。",
        "err_img_limit_group_15": "[JimengAI] 参数错误: 在组图模式下，输入参考图数量 ({n}) 加 生成图片数量 ({max}) 的总和 ({total}) 不能超过 15 张。",

        # API Key 相关
        "combo_no_keys": "未找到密钥 (请配置 api_keys.json)",
        "popup_key_valid_err": "[JimengAI] 配置错误: 所选密钥 '{key}' 无效或未找到。请检查 api_keys.json。",
    },
    "en": {
        "api_file_not_found": "[JimengAI] Info: API keys file not found. Please rename 'api_keys.json.example' to 'api_keys.json' and fill in your keys.",
        "api_file_empty": "[JimengAI] Warning: 'api_keys.json' is empty or not formatted correctly.",
        "api_load_error": "[JimengAI] Error: Failed to load 'api_keys.json': {e}",
        "api_key_not_found": "[JimengAI] Error: API Key for '{key_name}' not found.",
        "est_fallback": "Fallback Default",
        "est_history": "History Average",
        "est_regression": "Linear Regression",
        "est_recent": "Recent Load Adjustment",
        "task_submitted_est": "[JimengAI] Task submitted. Est. time: {time}s (Method: {method})",
        "task_info_simple": "[JimengAI] Task ID: {task_id} | Model: {model}",
        "batch_submit_start": "[JimengAI] Submitting batch of {count} tasks (Model: {model})...",
        "batch_submit_result": "[JimengAI] Submission complete. Created: {created}, Failed: {failed}.",
        "polling_single": "[JimengAI] Task {task_id}: {elapsed}s / {max}s elapsed...",
        "polling_batch": "[JimengAI] Batch Progress: {done}/{total} done. {pending} pending... ({elapsed}s / {max}s Est.)",
        "interrupted": "\n[JimengAI] Processing interrupted by user. Cancelling pending tasks...",
        "task_finished_single": "[JimengAI] Task completed successfully.",
        "batch_finished_stats": "[JimengAI] Batch finished. Success: {success}, Failed: {failed}.",
        "batch_handling": "[JimengAI] Handling {count} successful tasks. Sorting by seed and downloading...",
        "batch_copying": "[JimengAI] Copying files to output directory: {path}",
        "err_download_url": "[JimengAI] Async download failed, URL: {url}, Error: {e}",
        "err_task_create": "[JimengAI] Task creation failed: {e}",
        "err_task_check": "[JimengAI] Failed to check status for {tid}: {e}",
        "err_task_fail_msg": "[JimengAI] Task {tid} failed: {msg}",
        "err_batch_fail_all": "[JimengAI] Batch failed: No tasks succeeded.",
        "err_copy_fail": "[JimengAI] Failed to copy file: {path}. Error: {e}",
        "err_convert_tensor": "[JimengAI] Failed to convert frame to tensor: {e}",
        "err_check_status_batch": "[JimengAI] API Error checking batch status: {e}",
        "download_retry": "[JimengAI] Warning: Download failed (Attempt {attempt}/{total}). Retrying in {delay}s... Error: {e}",
        
        # Popup Messages
        "popup_req_failed": "[JimengAI] Request failed: {msg}",
        "popup_task_failed": "[JimengAI] Task {task_id} failed: {msg}",
        "popup_batch_pending": "[JimengAI] Batch ({count} tasks) is pending. Run again to check results.",
        "popup_task_pending": "[JimengAI] Task {task_id} is {status}. Run again to check results.",
        "popup_param_not_allowed": "[JimengAI] Parameter Error: Parameter '--{param}' is not allowed in the prompt. Please use the node's widget for this value.",
        "popup_first_frame_missing": "[JimengAI] Parameter Error: A first frame image must be provided when using a last frame image.",
        "popup_ref_missing": "[JimengAI] Parameter Error: At least one reference image must be provided.",
        "popup_prepare_failed": "[JimengAI] Failed to prepare task: {e}",

        # Image Nodes
        "err_pixels_range": "[JimengAI] Parameter Error: Total pixels must be between {min} ({min_desc}) and {max} ({max_desc}). Your current: {current}",
        "err_aspect_ratio": "[JimengAI] Parameter Error: Aspect ratio must be between {min} and {max}. Your current: {current}",
        "err_download_img": "[JimengAI] Error: Failed to download the generated image.",
        "err_gen_model": "[JimengAI] Failed to generate image with model {model}: {e}",
        "err_img_limit_10": "[JimengAI] Parameter Error: The number of input images cannot exceed 10.",
        "err_img_limit_15": "[JimengAI] Parameter Error: The sum of input images ({n}) and max generated images ({max}) cannot exceed 15.",
        "err_img_limit_group_15": "[JimengAI] Parameter Error: The sum of input images ({n}) and max generated images ({max}) cannot exceed 15 in group mode (Total: {total}).",

        # API Key
        "combo_no_keys": "No Keys Found (Check api_keys.json)",
        "popup_key_valid_err": "[JimengAI] Config Error: Selected key '{key}' is invalid or not found. Please check api_keys.json.",
    }
}

def detect_system_language():
    try:
        lang_code, _ = locale.getdefaultlocale()
        if lang_code and lang_code.startswith('zh'):
            return 'zh'
    except:
        pass
    return 'en'

def get_text(key):
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    return mapping.get(key, LOG_TRANSLATIONS["en"].get(key, key))

def log_msg(key, default_msg="", **kwargs):
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    msg = mapping.get(key, None)
    if msg is None:
        msg = LOG_TRANSLATIONS["en"].get(key, default_msg)
    
    if msg:
        try:
            print(msg.format(**kwargs))
        except Exception:
            print(msg)

def load_api_keys():
    global API_KEYS_CONFIG, CURRENT_LANG
    API_KEYS_CONFIG = []
    
    CURRENT_LANG = detect_system_language()

    if not os.path.exists(API_KEYS_FILE):
        log_msg("api_file_not_found")
        return

    try:
        with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
            keys_data = json.load(f)
            if isinstance(keys_data, list):
                for item in keys_data:
                    if "customName" in item and "apiKey" in item:
                        API_KEYS_CONFIG.append(item)
            if not API_KEYS_CONFIG:
                log_msg("api_file_empty")
    except Exception as e:
        log_msg("api_load_error", e=e)

async def _fetch_data_from_url_async(session: aiohttp.ClientSession, url: str, timeout: int = DEFAULT_DOWNLOAD_TIMEOUT, retries: int = DEFAULT_DOWNLOAD_RETRIES) -> bytes:
    for attempt in range(1, retries + 2):
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.get(url, timeout=client_timeout) as response:
                response.raise_for_status()
                return await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt > retries:
                raise e
            retry_delay = 2
            log_msg("download_retry", attempt=attempt, total=retries+1, delay=retry_delay, e=e)
            await asyncio.sleep(retry_delay)
    return b"" 

def _tensor2images(tensor: torch.Tensor) -> list:
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]

def _image_to_base64(image: torch.Tensor) -> str:
    if image is None: return None
    with io.BytesIO() as bytes_io:
        _tensor2images(image)[0].save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return base64.b64encode(data_bytes).decode("utf-8")

async def _download_url_to_image_tensor_async(session: aiohttp.ClientSession, url: str) -> torch.Tensor | None:
    if not url: return None
    try:
        image_data = await _fetch_data_from_url_async(session, url)
        i = PIL.Image.open(io.BytesIO(image_data))
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        return torch.from_numpy(image)[None,]
    except Exception as e:
        log_msg("err_download_url", url=url, e=e)
        return None

class JimengClients:
    def __init__(self, openai_client, ark_client):
        self.openai = openai_client
        self.ark = ark_client

class JimengAPIClient:
    # 即梦 API 客户端节点
    @classmethod
    def INPUT_TYPES(s):
        load_api_keys()
        key_names = [key["customName"] for key in API_KEYS_CONFIG]
        if not key_names:
            # 汉化：列表为空时的占位符
            key_names = [get_text("combo_no_keys")]
        return { "required": { "key_name": (key_names,), } }
    
    RETURN_TYPES = ("JIMENG_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_clients"
    CATEGORY = GLOBAL_CATEGORY

    def create_clients(self, key_name):
        api_key = None
        # 检查是否选择了占位符
        if key_name == get_text("combo_no_keys"):
            # 抛出中文弹窗，提示去配置 Keys
            raise RuntimeError(get_text("api_file_not_found"))

        for key_info in API_KEYS_CONFIG:
            if key_info["customName"] == key_name:
                api_key = key_info["apiKey"]
                break
        
        if not api_key:
            log_msg("api_key_not_found", key_name=key_name)
            # 抛出中文弹窗，提示 Key 无效
            raise RuntimeError(get_text("popup_key_valid_err").format(key=key_name))
            
        openai_client = AsyncOpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
        ark_client = Ark(api_key=api_key)
        
        clients = JimengClients(openai_client, ark_client)
        return (clients,)

NODE_CLASS_MAPPINGS = {
    "JimengAPIClient": JimengAPIClient,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengAPIClient": "Jimeng API Client",
}