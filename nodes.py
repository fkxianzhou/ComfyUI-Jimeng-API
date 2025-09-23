# --- 1. 导入所需模块 ---
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

# 导入OpenAI的异步客户端和火山引擎的原生SDK客户端
from openai import AsyncOpenAI
from volcenginesdkarkruntime import Ark
import folder_paths

# --- 2. 全局设置 ---
# 为所有节点定义一个统一的分类名，方便在ComfyUI菜单中查找。
GLOBAL_CATEGORY = "JimengAI"

# --- 3. 配置文件和辅助函数 ---

# 获取当前文件所在的目录，用于定位配置文件
jimeng_api_dir = os.path.dirname(os.path.abspath(__file__))
# 定义配置文件的路径
API_KEYS_FILE = os.path.join(jimeng_api_dir, "api_keys.json")

# 用于缓存从文件中读取的API密钥
API_KEYS_CONFIG = []

def load_api_keys():
    """从 api_keys.json 文件加载API密钥。"""
    global API_KEYS_CONFIG
    API_KEYS_CONFIG = [] # 每次加载前先清空
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

# ComfyUI启动时执行一次，加载密钥
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

async def _download_and_save_video_async(session: aiohttp.ClientSession, video_url: str, filename_prefix: str, seed: int | None) -> dict:
    """【异步】下载视频并保存，返回UI预览数据。"""
    if not video_url: return {"ui": {"video": []}}
    if seed is not None: filename_prefix = f"{filename_prefix}_seed_{seed}"
    output_dir = folder_paths.get_output_directory()
    (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
    file_ext = video_url.split('.')[-1].split('?')[0]
    final_filename = f"{filename}_{random.randint(1, 10000)}.{file_ext}"
    final_path = os.path.join(full_output_folder, final_filename)
    data = await _fetch_data_from_url_async(session, video_url)
    with open(final_path, "wb") as f: f.write(data)
    preview_data = [{"filename": final_filename, "subfolder": subfolder, "type": "output"}]
    return {"ui": {"images": preview_data, "animated": (True,)}}

def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    """检查提示词中是否包含了不应出现的命令行风格参数。"""
    for i in text_params:
        if f"--{i}" in prompt: raise ValueError(f"Parameter '--{i}' is not allowed in the prompt. Please use the node's widget for this value.")

async def _poll_task_until_completion_async(client, task_id: str, timeout=600, interval=5):
    """【异步】轮询任务状态，直到任务完成、失败或超时。"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            get_result = await asyncio.to_thread(client.content_generation.tasks.get, task_id=task_id)
            if get_result.status == "succeeded": return get_result
            elif get_result.status in ["failed", "cancelled"]:
                error = getattr(get_result, 'error', None)
                if error: raise RuntimeError(f"Task failed with code: {error.code}, message: {error.message}")
                else: raise RuntimeError(f"Task failed with status: {get_result.status}")
        except Exception as e:
            if isinstance(e, RuntimeError): raise e
            print(f"Failed to get task status, retrying... Error: {e}")
        await asyncio.sleep(interval)
    raise TimeoutError(f"Task polling timed out after {timeout} seconds for task_id: {task_id}")

# --- 4. 节点类定义 ---

class JimengAPIClient:
    """节点功能：从配置文件加载API密钥，并创建客户端供后续节点使用。"""
    @classmethod
    def INPUT_TYPES(s):
        key_names = [key["customName"] for key in API_KEYS_CONFIG]
        if not key_names:
            key_names = ["No Keys Found in api_keys.json"]
        
        return { "required": { "key_name": (key_names,), } }
    
    RETURN_TYPES = ("JIMENG_OPENAI_CLIENT", "JIMENG_ARK_CLIENT")
    RETURN_NAMES = ("openai_client", "ark_client")
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
        return (openai_client, ark_client)

class JimengText2Image:
    """节点功能：文本生成图片。"""
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "client": ("JIMENG_OPENAI_CLIENT",), "prompt": ("STRING", {"multiline": True, "default": ""}), "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", "1248x832", "1512x648"],), "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}), "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}), "watermark": ("BOOLEAN", {"default": False}), }, }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, size, seed, guidance_scale, watermark):
        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        async with aiohttp.ClientSession() as session:
            try:
                resp = await client.images.generate(model="doubao-seedream-3-0-t2i-250415", prompt=prompt, response_format="url", size=size, extra_body={"seed": actual_seed, "guidance_scale": guidance_scale, "watermark": watermark})
                image_tensor = await _download_url_to_image_tensor_async(session, resp.data[0].url)
                if image_tensor is None: raise RuntimeError("Failed to download the generated image.")
                return (image_tensor, actual_seed)
            except Exception as e:
                raise RuntimeError(f"Failed to generate image: {e}")

class JimengImageEdit:
    """节点功能：根据文本指令编辑一张图片。"""
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "client": ("JIMENG_OPENAI_CLIENT",), "image": ("IMAGE",), "prompt": ("STRING", {"multiline": True, "default": ""}), "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}), "guidance_scale": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 10.0, "step": 0.1}), "watermark": ("BOOLEAN", {"default": False}), }, }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, image, prompt, seed, guidance_scale, watermark):
        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        image_b64 = f"data:image/jpeg;base64,{_image_to_base64(image)}"
        async with aiohttp.ClientSession() as session:
            try:
                resp = await client.images.generate(model="doubao-seededit-3-0-i2i-250628", prompt=prompt, extra_body={"image": image_b64, "size": "adaptive", "seed": actual_seed, "guidance_scale": guidance_scale, "watermark": watermark})
                image_tensor = await _download_url_to_image_tensor_async(session, resp.data[0].url)
                if image_tensor is None: raise RuntimeError("Failed to download the edited image.")
                return (image_tensor, actual_seed)
            except Exception as e:
                raise RuntimeError(f"Failed to edit image: {e}")

class JimengSeedream4:
    """节点功能：使用Seedream4模型进行文生图、图生图，并支持单图/组图模式。"""
    RECOMMENDED_SIZES = [ "2048x2048 (1:1)", "2304x1728 (4:3)", "1728x2304 (3:4)", "2560x1440 (16:9)", "1440x2560 (9:16)", "2496x1664 (3:2)", "1664x2496 (2:3)", "3024x1296 (21:9)", "4096x4096 (1:1)" ]
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "client": ("JIMENG_OPENAI_CLIENT",), "prompt": ("STRING", {"multiline": True, "default": ""}), "generation_mode": (["Single Image (disabled)", "Image Group (auto)"],), "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}), "size": (s.RECOMMENDED_SIZES,), "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}), "watermark": ("BOOLEAN", {"default": False}), }, "optional": { "images": ("IMAGE",), } }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, generation_mode, max_images, size, seed, watermark, images=None):
        sequential_image_generation = "disabled" if "disabled" in generation_mode else "auto"
        n_input_images = 0
        if images is not None: n_input_images = images.shape[0]
        if n_input_images > 10: raise ValueError("The number of input images cannot exceed 10.")
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(f"The sum of input images ({n_input_images}) and max generated images ({max_images}) cannot exceed 15.")

        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        size_str = size.split(" ")[0]
        
        image_param = None
        if images is not None:
            image_b64_list = [_image_to_base64(images[i:i+1]) for i in range(n_input_images)]
            if n_input_images == 1: image_param = f"data:image/jpeg;base64,{image_b64_list[0]}"
            else: image_param = [f"data:image/jpeg;base64,{b64}" for b64 in image_b64_list]

        extra_body = {"seed": actual_seed, "watermark": watermark, "sequential_image_generation": sequential_image_generation}
        if image_param: extra_body["image"] = image_param
        if sequential_image_generation == "auto":
            extra_body['sequential_image_generation_options'] = {"max_images": max_images}

        async with aiohttp.ClientSession() as session:
            try:
                resp = await client.images.generate(model="doubao-seedream-4-0-250828", prompt=prompt, size=size_str, response_format="url", extra_body=extra_body)
                download_tasks = [_download_url_to_image_tensor_async(session, item.url) for item in resp.data]
                output_tensors = await asyncio.gather(*download_tasks)
                valid_tensors = [t for t in output_tensors if t is not None]
                if not valid_tensors: raise RuntimeError("Failed to download any of the generated images.")
                return (torch.cat(valid_tensors, dim=0), actual_seed)
            except Exception as e:
                raise RuntimeError(f"Failed to generate with Seedream 4: {e}")

class JimengVideoGeneration:
    """节点功能：根据文本或图片输入生成视频，并直接预览结果。"""
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "client": ("JIMENG_ARK_CLIENT",),
                "model_choice": (["doubao-seedance-1-0-pro", "doubao-seedance-1-0-lite"], {"default": "doubao-seedance-1-0-pro"}), 
                "prompt": ("STRING", {"multiline": True, "default": ""}), 
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}), 
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}), 
                "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
                "camerafixed": ("BOOLEAN", {"default": True}), 
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}), 
            },
            "optional": { "image": ("IMAGE",), } 
        }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, model_choice, prompt, duration, resolution, aspect_ratio, camerafixed, seed, image=None):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "camerafixed", "seed"])
        
        final_model_name = ""
        if model_choice == "doubao-seedance-1-0-pro":
            final_model_name = "doubao-seedance-1-0-pro-250528"
        elif model_choice == "doubao-seedance-1-0-lite":
            if image is None: final_model_name = "doubao-seedance-1-0-lite-t2v-250428"
            else: final_model_name = "doubao-seedance-1-0-lite-i2v-250428"
        
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} --dur {duration} --camerafixed {'true' if camerafixed else 'false'} --seed {actual_seed}"
        
        content = [{"type": "text", "text": prompt_string}]
        if image is not None:
            base64_str = _image_to_base64(image)
            data_url = f"data:image/jpeg;base64,{base64_str}"
            content.append({"type": "image_url", "image_url": {"url": data_url}, "role": "first_frame"})

        async with aiohttp.ClientSession() as session:
            try:
                create_result = await asyncio.to_thread(client.content_generation.tasks.create, model=final_model_name, content=content)
                task_id = create_result.id
            except Exception as e: 
                raise RuntimeError(f"Failed to create task: {e}")
            try:
                final_result = await _poll_task_until_completion_async(client, task_id)
                video_url = final_result.content.video_url
                filename_prefix = "Jimeng_T2V" if image is None else "Jimeng_I2V"
                return await _download_and_save_video_async(session, video_url, filename_prefix, actual_seed)
            except (RuntimeError, TimeoutError) as e:
                print(e)
                return {"ui": {"video": []}}

class JimengFirstLastFrame2Video:
    """节点功能：根据首尾两帧图片生成过渡视频，并直接预览。"""
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
            "client": ("JIMENG_ARK_CLIENT",), 
            "first_frame_image": ("IMAGE",), 
            "last_frame_image": ("IMAGE",), 
            "model": (["doubao-seedance-1-0-lite-i2v-250428"],), 
            "prompt": ("STRING", {"multiline": True, "default": ""}), 
            "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}), 
            "resolution": (["480p", "720p", "1080p"], {"default": "720p"}), 
            "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}), 
            "camerafixed": ("BOOLEAN", {"default": True}), 
            "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}), 
        }, }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, first_frame_image, last_frame_image, model, prompt, duration, resolution, aspect_ratio, camerafixed, seed):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "camerafixed", "seed"])
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        first_frame_data_url = f"data:image/jpeg;base64,{_image_to_base64(first_frame_image)}"
        last_frame_data_url = f"data:image/jpeg;base64,{_image_to_base64(last_frame_image)}"
        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} --dur {duration} --camerafixed {'true' if camerafixed else 'false'} --seed {actual_seed}"
        content = [ {"type": "text", "text": prompt_string}, {"type": "image_url", "image_url": {"url": first_frame_data_url}, "role": "first_frame"}, {"type": "image_url", "image_url": {"url": last_frame_data_url}, "role": "last_frame"} ]
        async with aiohttp.ClientSession() as session:
            try:
                create_result = await asyncio.to_thread( client.content_generation.tasks.create, model=model, content=content )
                task_id = create_result.id
            except Exception as e: raise RuntimeError(f"Failed to create task: {e}")
            try:
                final_result = await _poll_task_until_completion_async(client, task_id)
                return await _download_and_save_video_async(session, final_result.content.video_url, "Jimeng_F&L-I2V", actual_seed)
            except (RuntimeError, TimeoutError) as e:
                print(e)
                return {"ui": {"video": []}}

class JimengReferenceImage2Video:
    """节点功能：根据一张或多张参考图生成视频，并直接预览。"""
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
            "client": ("JIMENG_ARK_CLIENT",), 
            "prompt": ("STRING", {"multiline": True, "default": ""}), 
            "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}), 
            "resolution": (["480p", "720p"], {"default": "720p"}), 
            "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}), 
            "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}), 
        }, "optional": { 
            "ref_image_1": ("IMAGE",), 
            "ref_image_2": ("IMAGE",), 
            "ref_image_3": ("IMAGE",), 
            "ref_image_4": ("IMAGE",), 
        } }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, duration, resolution, aspect_ratio, seed, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "seed"])
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        model = "doubao-seedance-1-0-lite-i2v-250428"
        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} --dur {duration} --seed {actual_seed}"
        content = [{"type": "text", "text": prompt_string}]
        ref_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        for img_tensor in ref_images:
            if img_tensor is not None: content.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img_tensor)}"}, "role": "reference_image" })
        if len(content) == 1: raise ValueError("At least one reference image must be provided.")
        async with aiohttp.ClientSession() as session:
            try:
                create_result = await asyncio.to_thread( client.content_generation.tasks.create, model=model, content=content )
                task_id = create_result.id
            except Exception as e: raise RuntimeError(f"Failed to create task: {e}")
            try:
                final_result = await _poll_task_until_completion_async(client, task_id)
                return await _download_and_save_video_async(session, final_result.content.video_url, "Jimeng_Ref-I2V", actual_seed)
            except (RuntimeError, TimeoutError) as e:
                print(e)
                return {"ui": {"video": []}}

class JimengTaskStatusChecker:
    """节点功能：手动输入一个任务ID来查询其状态和结果。"""
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "client": ("JIMENG_ARK_CLIENT",), "task_id": ("STRING", {"forceInput": True}), } }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "last_frame_url", "status", "error_message", "model", "created_at", "updated_at")
    FUNCTION = "check_status"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    async def check_status(self, client, task_id):
        if not task_id: return ("", "", "no task_id provided", "Error: Task ID is empty.", "", "", "")
        try:
            get_result = await asyncio.to_thread(client.content_generation.tasks.get, task_id=task_id)
            status, model = get_result.status, get_result.model
            created_at = datetime.datetime.fromtimestamp(get_result.created_at).strftime('%Y-%m-%d %H:%M:%S')
            updated_at = datetime.datetime.fromtimestamp(get_result.updated_at).strftime('%Y-%m-%d %H:%M:%S')
            video_url, last_frame_url, error_message = "", "", ""
            if status == "succeeded":
                video_url = get_result.content.video_url
                last_frame_url = getattr(get_result.content, 'last_frame_url', "")
            elif status == "failed" and get_result.error:
                error_message = f"Code: {get_result.error.code}, Message: {get_result.error.message}"
            return (video_url, last_frame_url, status, error_message, model, created_at, updated_at)
        except Exception as e:
            return ("", "", "api_error", f"API Error: {e}", "", "", "")

# --- 5. 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "JimengAPIClient": JimengAPIClient,
    "JimengText2Image": JimengText2Image,
    "JimengImageEdit": JimengImageEdit,
    "JimengSeedream4": JimengSeedream4,
    "JimengVideoGeneration": JimengVideoGeneration,
    "JimengFirstLastFrame2Video": JimengFirstLastFrame2Video,
    "JimengReferenceImage2Video": JimengReferenceImage2Video,
    "JimengTaskStatusChecker": JimengTaskStatusChecker,
}

# 节点在UI中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengAPIClient": "Jimeng API Client",
    "JimengText2Image": "Jimeng Text to Image (Seedream 3)",
    "JimengImageEdit": "Jimeng Image Edit (Seededit 3)",
    "JimengSeedream4": "Jimeng Advanced Image (Seedream 4)",
    "JimengVideoGeneration": "Jimeng Video Generation",
    "JimengFirstLastFrame2Video": "Jimeng F&L Frame to Video",
    "JimengReferenceImage2Video": "Jimeng Reference to Video",
    "JimengTaskStatusChecker": "Jimeng Task Status Checker",
}