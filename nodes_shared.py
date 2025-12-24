import os
import io
import base64
import locale
import json
import re
import numpy
import PIL.Image
import torch
from volcenginesdkarkruntime import Ark

from comfy_api.latest import io as comfy_io

# 导入常量
from .constants import LOG_TRANSLATIONS, ERROR_TEXT_MATCH_RULES

GLOBAL_CATEGORY = "JimengAI"

jimeng_api_dir = os.path.dirname(os.path.abspath(__file__))
API_KEYS_FILE = os.path.join(jimeng_api_dir, "api_keys.json")

API_KEYS_CONFIG = []
CURRENT_LANG = "en"

# 定义自定义类型,供本文件和其他文件使用
JimengClientType = comfy_io.Custom("JIMENG_CLIENT")


def detect_system_language():
    try:
        lang_code, _ = locale.getdefaultlocale()
        if lang_code and lang_code.startswith("zh"):
            return "zh"
    except:
        pass
    return "en"


def get_text(key):
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    return mapping.get(key, LOG_TRANSLATIONS["en"].get(key, key))


def log_msg(key, default_msg="", **kwargs):
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    msg = mapping.get(key, None)
    if not msg:
        msg = LOG_TRANSLATIONS["en"].get(key, default_msg)
    if msg:
        try:
            print(msg.format(**kwargs))
        except:
            print(msg)


def format_api_error(e):
    # 格式化 API 错误
    global CURRENT_LANG
    mapping = LOG_TRANSLATIONS.get(CURRENT_LANG, LOG_TRANSLATIONS["en"])
    error_map = mapping.get("api_errors", {})
    fallback_map = LOG_TRANSLATIONS["en"].get("api_errors", {})

    err_code = None
    err_msg = str(e)
    detected_code = None

    # 提取错误代码与消息
    code_match = re.search(r"'code':\s*'([^']+)'", err_msg)
    if code_match:
        err_code = code_match.group(1)

    msg_match = re.search(r"'message':\s*'([^']+)'", err_msg)
    if msg_match:
        extracted_msg = msg_match.group(1)
        if len(extracted_msg) > 0:
            err_msg = extracted_msg

    # 匹配虚拟错误代码
    for keyword, mapped_code in ERROR_TEXT_MATCH_RULES.items():
        if keyword.lower() in err_msg.lower():
            detected_code = mapped_code
            break

    # 确定错误代码
    final_code = detected_code if detected_code else err_code

    # 查找错误翻译
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
    global API_KEYS_CONFIG, CURRENT_LANG
    API_KEYS_CONFIG = []
    CURRENT_LANG = detect_system_language()

    if not os.path.exists(API_KEYS_FILE):
        log_msg("api_file_not_found")
        return

    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            keys_data = json.load(f)
            if isinstance(keys_data, list):
                for item in keys_data:
                    if "customName" in item and "apiKey" in item:
                        API_KEYS_CONFIG.append(item)
            if not API_KEYS_CONFIG:
                log_msg("api_file_empty")
    except Exception as e:
        log_msg("api_load_error", e=e)


def _tensor2images(tensor: torch.Tensor) -> list:
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]


def _image_to_base64(image: torch.Tensor) -> str:
    if image is None:
        return None
    with io.BytesIO() as bytes_io:
        _tensor2images(image)[0].save(bytes_io, format="JPEG")
        data_bytes = bytes_io.getvalue()
    return base64.b64encode(data_bytes).decode("utf-8")


class JimengClients:
    def __init__(self, ark_client):
        self.ark = ark_client


class JimengAPIClient(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        load_api_keys()
        key_names = [key["customName"] for key in API_KEYS_CONFIG]
        if not key_names:
            key_names = [get_text("combo_no_keys")]

        return comfy_io.Schema(
            node_id="JimengAPIClient",
            display_name="Jimeng API Client",
            category=GLOBAL_CATEGORY,
            inputs=[comfy_io.Combo.Input("key_name", options=key_names)],
            outputs=[JimengClientType.Output(display_name="client")],
        )

    @classmethod
    def execute(cls, key_name) -> comfy_io.NodeOutput:
        api_key = None
        if key_name == get_text("combo_no_keys"):
            raise RuntimeError(get_text("api_file_not_found"))

        for key_info in API_KEYS_CONFIG:
            if key_info["customName"] == key_name:
                api_key = key_info["apiKey"]
                break

        if not api_key:
            log_msg("api_key_not_found", key_name=key_name)
            raise RuntimeError(get_text("popup_key_valid_err").format(key=key_name))

        ark_client = Ark(
            api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

        return comfy_io.NodeOutput(JimengClients(ark_client))
