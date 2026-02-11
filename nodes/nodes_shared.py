import os
import io
import base64
import locale
import json
import re
import numpy
import PIL.Image
import torch
import requests
from volcenginesdkarkruntime import Ark

from comfy_api.latest import io as comfy_io

import logging

from .constants import LOG_TRANSLATIONS, ERROR_TEXT_MATCH_RULES

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
        url = "https://ark.cn-beijing.volces.com/api/v3"
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
    def __init__(self, ark_client):
        self.ark = ark_client


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
            api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

        return comfy_io.NodeOutput(JimengClients(ark_client))
