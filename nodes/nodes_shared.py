import os
import io
import base64
import locale
import json
import re
import folder_paths
import time
import numpy
import PIL.Image
import torch
import torch.nn.functional as F
import requests
import cv2
import asyncio
from volcenginesdkarkruntime import Ark

from comfy_api.latest import io as comfy_io

import logging

from .constants import LOG_TRANSLATIONS, ERROR_TEXT_MATCH_RULES, JIMENG_API_BASE_URL

LOG_PREFIX = "[JimengAI] "

def patch_log_translations():
    """
    自动为日志消息添加前缀。
    """
    ignore_keys = {"api_errors"}
    
    for lang in LOG_TRANSLATIONS:
        trans_map = LOG_TRANSLATIONS[lang]
        for key, value in trans_map.items():
            if key in ignore_keys or key.startswith("est_"):
                continue
            
            if isinstance(value, str):
                if value.strip().startswith("-"):
                    continue
                
                if value.startswith("\n"):
                    if LOG_PREFIX.strip() not in value:
                        trans_map[key] = "\n" + LOG_PREFIX + value[1:]
                else:
                    if not value.startswith(LOG_PREFIX):
                        trans_map[key] = LOG_PREFIX + value

patch_log_translations()

logger = logging.getLogger("JimengAI")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

GLOBAL_CATEGORY = "JimengAI"

jimeng_api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEYS_FILE = os.path.join(jimeng_api_dir, "api_keys.json")

API_KEYS_CONFIG = []
CURRENT_LANG = "en"

JimengClientType = comfy_io.Custom("JIMENG_CLIENT")


def detect_system_language():
    """
    检测系统语言，如果是中文环境则返回 'zh'，否则返回 'en'。
    """
    try:
        lang_code, _ = locale.getdefaultlocale()
        if lang_code and lang_code.startswith("zh"):
            return "zh"
    except:
        pass
    return "en"


def get_text(key, **kwargs):
    """
    获取指定 key 的本地化文本。
    """
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    msg = mapping.get(key, LOG_TRANSLATIONS["en"].get(key, key))
    if kwargs:
        try:
            return msg.format(**kwargs)
        except:
            pass
    return msg


def log_msg(key, default_msg="", **kwargs):
    """
    记录本地化日志信息。
    """
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    msg = mapping.get(key, None)
    if not msg:
        msg = LOG_TRANSLATIONS["en"].get(key, default_msg)
    if msg:
        try:
            logger.info(msg.format(**kwargs))
        except:
            logger.info(msg)


def get_node_count_in_workflow(class_type, prompt=None):
    """
    获取当前工作流中指定类型节点的数量。
    """
    try:
        if prompt is None:
            from server import PromptServer
            prompt = PromptServer.instance.prompt
        
        if not prompt:
            return 0
        
        count = 0
        for key, value in prompt.items():
            if value.get("class_type") == class_type:
                count += 1
        return count
    except Exception:
        return 0


def format_api_error(e):
    """
    格式化 API 错误信息。
    尝试解析错误代码并返回对应的本地化错误描述。
    """
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    error_map = mapping.get("api_errors", {})
    fallback_map = LOG_TRANSLATIONS["en"].get("api_errors", {})

    err_code = None
    err_msg = str(e)
    detected_code = None

    code_match = re.search(r"'code':\s*'([^']+)'", err_msg)
    if code_match:
        err_code = code_match.group(1)

    msg_match = re.search(r"'message':\s*'([^']+)'", err_msg)
    if msg_match:
        extracted_msg = msg_match.group(1)
        if len(extracted_msg) > 0:
            err_msg = extracted_msg

    for keyword, mapped_code in ERROR_TEXT_MATCH_RULES.items():
        if keyword.lower() in err_msg.lower():
            detected_code = mapped_code
            break

    final_code = detected_code if detected_code else err_code

    if final_code:
        final_code = str(final_code)
        matched_msg = None

        if final_code in error_map:
            matched_msg = error_map[final_code]
        elif final_code in fallback_map:
            matched_msg = fallback_map[final_code]

        if not matched_msg:
            for key in error_map:
                if final_code.startswith(key):
                    matched_msg = error_map[key]
                    break
            if not matched_msg:
                for key in fallback_map:
                    if final_code.startswith(key):
                        matched_msg = fallback_map[key]
                        break

        if matched_msg:
            return f"[JimengAI] {matched_msg} (Code: {final_code})"

    return f"[JimengAI] Error: {err_msg}"


def load_api_keys():
    """
    加载 API 密钥配置文件 (api_keys.json)。
    """
    global API_KEYS_CONFIG, CURRENT_LANG
    API_KEYS_CONFIG = []
    CURRENT_LANG = detect_system_language()

    if not os.path.exists(API_KEYS_FILE):
        pass

    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                keys_data = json.load(f)
                if isinstance(keys_data, list):
                    for item in keys_data:
                        if "customName" in item and "apiKey" in item:
                            API_KEYS_CONFIG.append(item)
    except Exception as e:
        log_msg("api_load_error", e=e)


def save_api_key(name, key):
    """
    保存新的 API Key 到配置文件。
    """
    global API_KEYS_CONFIG
    
    updated = False
    for item in API_KEYS_CONFIG:
        if item["customName"] == name:
            item["apiKey"] = key
            updated = True
            break
    
    if not updated:
        API_KEYS_CONFIG.append({"customName": name, "apiKey": key})
    
    try:
        with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(API_KEYS_CONFIG, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved API Key: {name}")
    except Exception as e:
        logger.error(f"Failed to save API key: {e}")


def validate_api_key(api_key: str) -> bool:
    """
    验证 API Key 是否有效。
    """
    try:
        url = JIMENG_API_BASE_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 401:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"API Key validation error: {e}")
        return False


FILES_UPLOAD_CACHE = {}


async def upload_file_to_ark(client, file_path, fps=None):
    """
    使用 client.ark.files.create 上传文件。
    """
    global FILES_UPLOAD_CACHE
    
    if file_path in FILES_UPLOAD_CACHE:
        log_msg("visual_found_file", path=file_path)
        return FILES_UPLOAD_CACHE[file_path]
        
    try:
        log_msg("visual_uploading", path=file_path)
        if hasattr(client.ark, "files"):
            with open(file_path, "rb") as f:
                
                upload_kwargs = {
                    "file": f,
                    "purpose": "user_data"
                }
                
                if fps is not None:
                    preprocess_config = {
                        "video": {
                            "fps": float(fps)
                        }
                    }
                    upload_kwargs["preprocess_configs"] = preprocess_config

                file_obj = await asyncio.to_thread(
                    client.ark.files.create,
                    **upload_kwargs
                )
                
                if hasattr(file_obj, "id"):
                    file_id = file_obj.id
                    log_msg("visual_uploaded", id=file_id, status=getattr(file_obj, 'status', 'unknown'))
                    
                    await wait_for_file_active(client, file_id)
                    
                    FILES_UPLOAD_CACHE[file_path] = file_id
                    return file_id
                else:
                    raise JimengException("Upload failed: No file ID returned.")
        else:
             raise JimengException("SDK does not support files.create.")
             
    except Exception as e:
        logger.error(f"Upload failed for {file_path}: {e}")
        raise JimengException(f"File upload failed: {e}")


async def wait_for_file_active(client, file_id):
    """
    轮询文件状态，直到其变为“active”状态。
    """
    log_msg("visual_wait_active", id=file_id)
    max_retries = 60
    
    for _ in range(max_retries):
        try:
            file_info = await asyncio.to_thread(
                client.ark.files.retrieve,
                file_id=file_id
            )
            
            status = getattr(file_info, "status", "unknown")
            log_msg("visual_file_status", id=file_id, status=status)
            
            if status == "active":
                return True
            elif status == "error":
                raise JimengException(f"File processing failed: {file_id}")
            
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error checking file status: {e}")
            await asyncio.sleep(1)
            
    raise JimengException(f"Timeout waiting for file {file_id} to become active.")


def _tensor2images(tensor: torch.Tensor) -> list:
    """
    将 PyTorch Tensor 转换为 PIL Image 列表。
    """
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]


def _image_to_base64(image: torch.Tensor) -> str:
    """
    将单张图片 Tensor 转换为 Base64 编码字符串 (JPEG 格式)。
    """
    if image is None:
        return None
    with io.BytesIO() as bytes_io:
        _tensor2images(image)[0].save(bytes_io, format="JPEG")
        data_bytes = bytes_io.getvalue()
    return base64.b64encode(data_bytes).decode("utf-8")


def create_white_image_tensor(width=1024, height=1024):
    """
    创建一个指定大小的纯白图片 Tensor。
    Shape: [1, height, width, 3]
    """
    return torch.ones((1, height, width, 3), dtype=torch.float32)


def safe_cat_tensors(tensors, dim=0):
    """
    安全地将多个 Tensor 拼接在一起。
    主要用于处理 Adaptive Size 返回不同尺寸图片的情况。
    """
    if not tensors:
        return None

    if not isinstance(tensors, list):
        return tensors

    if len(tensors) == 0:
        return None

    target_tensor = tensors[0]
    target_h, target_w = target_tensor.shape[1], target_tensor.shape[2]

    processed_tensors = []
    for t in tensors:
        if t.shape[1] != target_h or t.shape[2] != target_w:
            t_permuted = t.permute(0, 3, 1, 2)
            t_resized = F.interpolate(
                t_permuted, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            t_final = t_resized.permute(0, 2, 3, 1)
            processed_tensors.append(t_final)
        else:
            processed_tensors.append(t)

    return torch.cat(processed_tensors, dim=dim)


def create_white_video_file(filename_prefix, width=1024, height=1024):
    """
    创建一个指定大小的 1 帧纯白视频文件 (H.264 MP4)。
    返回视频文件的绝对路径。
    """
    import tempfile
    try:
        import cv2
    except ImportError:
        return None

    try:
        temp_dir = folder_paths.get_temp_directory()
        timestamp = int(time.time() * 1000)
        
        flat_prefix = filename_prefix.replace("/", "_").replace("\\", "_")
        filename = f"{flat_prefix}_dummy_{timestamp}.mp4"
        
        filepath = os.path.join(temp_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 24.0
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        if not out.isOpened():
            log_msg("err_create_dummy_video", e="Failed to open VideoWriter with mp4v")
            return None

        white_frame = numpy.ones((height, width, 3), dtype=numpy.uint8) * 255
        
        out.write(white_frame)
        out.release()
        
        if not os.path.exists(filepath):
             log_msg("err_create_dummy_video", e="File was not created on disk")
             return None
        
        return filepath
    except Exception as e:
        log_msg("err_create_dummy_video", e=e)
        return None


class JimengException(Exception):
    """
    Jimeng 自定义异常类。
    设置 jimeng_suppress_traceback = True 以在打印时抑制堆栈跟踪。
    """
    def __init__(self, message):
        super().__init__(message)
        self.jimeng_suppress_traceback = True


class JimengClients:
    """
    包装 Ark 客户端的容器类。
    """
    def __init__(self, ark_client, api_key=None):
        self.ark = ark_client
        self.api_key = api_key

    def check_quota(self, model: str, estimated_cost: int):
        if not self.api_key:
            return
        from .quota import QuotaManager
        QuotaManager.instance().check_quota(self.api_key, model, estimated_cost)

    def update_usage(self, model: str, actual_cost: int):
        if not self.api_key:
            return
        from .quota import QuotaManager
        QuotaManager.instance().update_usage(self.api_key, model, actual_cost)


class JimengAPIClient(comfy_io.ComfyNode):
    """
    Jimeng API 客户端节点。
    负责加载 API 密钥并初始化 Ark 客户端。
    """
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        load_api_keys()
        key_names = [key["customName"] for key in API_KEYS_CONFIG]
        key_names.append("Custom")

        return comfy_io.Schema(
            node_id="JimengAPIClient",
            display_name="Jimeng API Client",
            category=GLOBAL_CATEGORY,
            inputs=[
                comfy_io.String.Input("new_api_key", default=""),
                comfy_io.String.Input("new_key_name", default=""),
                comfy_io.Combo.Input("key_name", options=key_names),
            ],
            outputs=[JimengClientType.Output(display_name="client")],
        )

    @classmethod
    def execute(
        cls, key_name, new_api_key="", new_key_name=""
    ) -> comfy_io.NodeOutput:
        api_key = None

        if key_name == "Custom":
            if not new_api_key or not new_api_key.strip():
                raise JimengException(get_text("err_new_key_empty"))
            
            api_key = new_api_key.strip()
            
            if not validate_api_key(api_key):
                raise JimengException(get_text("err_new_key_invalid"))
            
            if new_key_name and new_key_name.strip():
                save_api_key(new_key_name.strip(), api_key)
                print(get_text("info_new_key_saved", name=new_key_name.strip()))

        else:
            for key_info in API_KEYS_CONFIG:
                if key_info["customName"] == key_name:
                    api_key = key_info["apiKey"]
                    break

        if not api_key:
            log_msg("api_key_not_found", key_name=key_name)
            raise JimengException(get_text("popup_key_valid_err").format(key=key_name))

        ark_client = Ark(
            api_key=api_key, base_url=JIMENG_API_BASE_URL
        )

        return comfy_io.NodeOutput(JimengClients(ark_client, api_key))
