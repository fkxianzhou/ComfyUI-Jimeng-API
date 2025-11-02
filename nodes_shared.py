import os
import io
import time
import base64
import random
import datetime
import asyncio
import aiohttp
import json

import numpy
import PIL.Image
import torch

from openai import AsyncOpenAI
from volcenginesdkarkruntime import Ark
import folder_paths
import comfy.model_management

GLOBAL_CATEGORY = "JimengAI"

jimeng_api_dir = os.path.dirname(os.path.abspath(__file__))
API_KEYS_FILE = os.path.join(jimeng_api_dir, "api_keys.json")

API_KEYS_CONFIG = []

def load_api_keys():
    """从 api_keys.json 文件加载API密钥。"""
    global API_KEYS_CONFIG
    API_KEYS_CONFIG = []
    if not os.path.exists(API_KEYS_FILE):
        print(f"[JimengAI] Info: API keys file not found. Please rename 'api_keys.json.example' to 'api_keys.json' and fill in your keys.")
        return

    try:
        with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
            keys_data = json.load(f)
            if isinstance(keys_data, list):
                for item in keys_data:
                    if "customName" in item and "apiKey" in item:
                        API_KEYS_CONFIG.append(item)
            if not API_KEYS_CONFIG:
                print(f"[JimengAI] Warning: 'api_keys.json' is empty or not formatted correctly.")
    except Exception as e:
        print(f"[JimengAI] Error: Failed to load 'api_keys.json': {e}")

load_api_keys()

async def _fetch_data_from_url_async(session: aiohttp.ClientSession, url: str) -> bytes:
    """【异步】从给定的URL下载数据。"""
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()

def _tensor2images(tensor: torch.Tensor) -> list:
    """将输入的PyTorch Tensor转换为PIL Image对象列表。"""
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]

def _image_to_base64(image: torch.Tensor) -> str:
    """将单个图像Tensor转换为Base64编码的字符串。"""
    if image is None: return None
    with io.BytesIO() as bytes_io:
        _tensor2images(image)[0].save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return base64.b64encode(data_bytes).decode("utf-8")

async def _download_url_to_image_tensor_async(session: aiohttp.ClientSession, url: str) -> torch.Tensor | None:
    """【异步】从URL下载图片并将其转换为ComfyUI的IMAGE Tensor格式。"""
    if not url: return None
    try:
        image_data = await _fetch_data_from_url_async(session, url)
        i = PIL.Image.open(io.BytesIO(image_data))
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        return torch.from_numpy(image)[None,]
    except Exception as e:
        print(f"异步下载或转换图片失败，URL: {url}，错误: {e}")
        return None

class JimengClients:
    """持有OpenAI和Ark两种API客户端的容器。"""
    def __init__(self, openai_client, ark_client):
        self.openai = openai_client
        self.ark = ark_client

class JimengAPIClient:
    """加载API密钥并创建统一的客户端。"""
    @classmethod
    def INPUT_TYPES(s):
        key_names = [key["customName"] for key in API_KEYS_CONFIG]
        if not key_names:
            key_names = ["No Keys Found in api_keys.json"]
        
        return { "required": { "key_name": (key_names,), } }
    
    RETURN_TYPES = ("JIMENG_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_clients"
    CATEGORY = GLOBAL_CATEGORY

    def create_clients(self, key_name):
        api_key = None
        for key_info in API_KEYS_CONFIG:
            if key_info["customName"] == key_name:
                api_key = key_info["apiKey"]
                break
        
        if not api_key:
            raise ValueError(f"API Key for '{key_name}' not found. Please check your 'api_keys.json' file.")
            
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