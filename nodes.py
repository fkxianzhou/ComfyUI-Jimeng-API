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

async def _download_and_save_video_async(session: aiohttp.ClientSession, video_url: str, filename_prefix: str, seed: int | None, save_path: str) -> dict:
    """【异步】下载视频并保存，返回UI预览数据。"""
    if not video_url: return {"ui": {"video": []}}
    if seed is not None: filename_prefix = f"{filename_prefix}_seed_{seed}"

    if save_path:
        filename_prefix = os.path.join(save_path, filename_prefix)

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
    """【异步】轮询任务状态，支持中断。"""
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            comfy.model_management.throw_exception_if_processing_interrupted()

            try:
                get_result = await asyncio.to_thread(client.content_generation.tasks.get, task_id=task_id)
                if get_result.status == "succeeded": return get_result
                elif get_result.status in ["failed", "cancelled"]:
                    error = getattr(get_result, 'error', None)
                    if error: raise RuntimeError(f"Task failed with code: {error.code}, message: {error.message}")
                    else: raise RuntimeError(f"Task failed with status: {get_result.status}")
            except Exception as e:
                if isinstance(e, RuntimeError): raise e
                if isinstance(e, comfy.model_management.InterruptProcessingException): raise e
                print(f"Failed to get task status, retrying... Error: {e}")
            
            await asyncio.sleep(interval)
        
        raise TimeoutError(f"Task polling timed out after {timeout} seconds for task_id: {task_id}")
    
    except comfy.model_management.InterruptProcessingException as e:
        print(f"[JimengAI] Info: Interruption detected for task {task_id}. Attempting to cancel task on API...")
        try:
            await asyncio.to_thread(client.content_generation.tasks.delete, task_id=task_id)
            print(f"[JimengAI] Info: Sent cancellation request for task {task_id}.")
        except Exception as delete_e:
            print(f"[JimengAI] Warning: Failed to send cancellation request for task {task_id}. This is OK. Error: {delete_e}")
        
        raise e

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

class JimengSeedream3:
    """Seedream 3 & Seededit 3 图像生成。"""
    RECOMMENDED_SIZES = ["Custom", "1024x1024 (1:1)", "864x1152 (3:4)", "1152x864 (4:3)", "1280x720 (16:9)", "720x1280 (9:16)", "832x1248 (2:3)", "1248x832 (3:2)", "1512x648 (21:9)"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (s.RECOMMENDED_SIZES,),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, size, width, height, seed, guidance_scale, watermark, image=None):
        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        openai_client = client.openai

        model_id = ""
        extra_body = {
            "seed": actual_seed,
            "guidance_scale": guidance_scale,
            "watermark": watermark
        }
        
        if size == "Custom":
            total_pixels = width * height
            min_pixels = 512 * 512  
            max_pixels = 2048 * 2048 
            if not (min_pixels <= total_pixels <= max_pixels):
                raise ValueError(f"Total pixels must be between {min_pixels} (512x512) and {max_pixels} (2048x2048). Your current: {total_pixels}")

            aspect_ratio = width / height
            if not (1/16 <= aspect_ratio <= 16):
                raise ValueError(f"Aspect ratio must be between 1/16 and 16. Your current: {aspect_ratio}")
                
            size_param = f"{width}x{height}"
        else:
            size_param = size.split(" ")[0]

        if image is None:
            model_id = "doubao-seedream-3-0-t2i-250415"
        else:
            model_id = "doubao-seededit-3-0-i2i-250628"
            image_b64 = f"data:image/jpeg;base64,{_image_to_base64(image)}"
            extra_body["image"] = image_b64
            size_param = "adaptive"

        async with aiohttp.ClientSession() as session:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                resp = await openai_client.images.generate(
                    model=model_id,
                    prompt=prompt,
                    size=size_param,
                    response_format="url",
                    extra_body=extra_body
                )
                
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                image_tensor = await _download_url_to_image_tensor_async(session, resp.data[0].url)
                if image_tensor is None:
                    raise RuntimeError("Failed to download the generated image.")
                return (image_tensor, actual_seed)
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to generate image with model {model_id}: {e}")

class JimengSeedream4:
    """Seedream4 文生图与图生图。"""
    RECOMMENDED_SIZES = [ "Custom", "2048x2048 (1:1)", "2304x1728 (4:3)", "1728x2304 (3:4)", "2560x1440 (16:9)", "1440x2560 (9:16)", "2496x1664 (3:2)", "1664x2496 (2:3)", "3024x1296 (21:9)", "4096x4096 (1:1)" ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "generation_mode": (["Single Image (disabled)", "Image Group (auto)"],),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "size": (s.RECOMMENDED_SIZES,),
                "width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, generation_mode, max_images, size, width, height, seed, watermark, images=None):
        sequential_image_generation = "disabled" if "disabled" in generation_mode else "auto"
        n_input_images = 0
        if images is not None: n_input_images = images.shape[0]
        if n_input_images > 10: raise ValueError("The number of input images cannot exceed 10.")
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(f"The sum of input images ({n_input_images}) and max generated images ({max_images}) cannot exceed 15.")

        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        
        if size == "Custom":
            total_pixels = width * height
            min_pixels = 1280 * 720
            max_pixels = 4096 * 4096
            if not (min_pixels <= total_pixels <= max_pixels):
                raise ValueError(f"Total pixels must be between {min_pixels} and {max_pixels}. Your current: {total_pixels}")

            aspect_ratio = width / height
            if not (1/16 <= aspect_ratio <= 16):
                raise ValueError(f"Aspect ratio must be between 1/16 and 16. Your current: {aspect_ratio}")

            size_str = f"{width}x{height}"
        else:
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

        openai_client = client.openai
        async with aiohttp.ClientSession() as session:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                resp = await openai_client.images.generate(model="doubao-seedream-4-0-250828", prompt=prompt, size=size_str, response_format="url", extra_body=extra_body)
                
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                download_tasks = [_download_url_to_image_tensor_async(session, item.url) for item in resp.data]
                output_tensors = await asyncio.gather(*download_tasks)
                valid_tensors = [t for t in output_tensors if t is not None]
                if not valid_tensors: raise RuntimeError("Failed to download any of the generated images.")
                return (torch.cat(valid_tensors, dim=0), actual_seed)
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to generate with Seedream 4: {e}")

class JimengVideoGeneration:
    """使用 doubao-seedance 模型生成视频。"""
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "model_choice": (["doubao-seedance-1-0-pro", "doubao-seedance-1-0-lite"], {"default": "doubao-seedance-1-0-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "Jimeng"}),
                "image": ("IMAGE",),
                "last_frame_image": ("IMAGE",)
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frame",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, model_choice, prompt, duration, resolution, aspect_ratio, camerafixed, seed, save_path, image=None, last_frame_image=None):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "camerafixed", "seed"])

        final_model_name = ""
        if model_choice == "doubao-seedance-1-0-pro":
            final_model_name = "doubao-seedance-1-0-pro-250528"
        elif model_choice == "doubao-seedance-1-0-lite":
            if image is None:
                final_model_name = "doubao-seedance-1-0-lite-t2v-250428"
            else:
                final_model_name = "doubao-seedance-1-0-lite-i2v-250428"
        
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} --dur {duration} --camerafixed {'true' if camerafixed else 'false'} --seed {actual_seed}"
        
        content = [{"type": "text", "text": prompt_string}]
        
        if image is not None:
            first_frame_b64 = _image_to_base64(image)
            first_frame_url = f"data:image/jpeg;base64,{first_frame_b64}"
            content.append({"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"})
        
        if last_frame_image is not None:
            if image is None:
                raise ValueError("A first frame image must be provided when using a last frame image.")
            last_frame_b64 = _image_to_base64(last_frame_image)
            last_frame_url = f"data:image/jpeg;base64,{last_frame_b64}"
            content.append({"type": "image_url", "image_url": {"url": last_frame_url}, "role": "last_frame"})

        ark_client = client.ark
        async with aiohttp.ClientSession() as session:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                create_result = await asyncio.to_thread(ark_client.content_generation.tasks.create, model=final_model_name, content=content, return_last_frame=True)
                task_id = create_result.id
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to create task: {e}")
            
            try:
                final_result = await _poll_task_until_completion_async(ark_client, task_id)
                
                video_url = final_result.content.video_url
                last_frame_url = getattr(final_result.content, 'last_frame_url', None)
                
                filename_prefix = "Jimeng_VideoGen"
                ui_data = await _download_and_save_video_async(session, video_url, filename_prefix, actual_seed, save_path)
                
                last_frame_tensor = await _download_url_to_image_tensor_async(session, last_frame_url)
                
                return {"ui": ui_data["ui"], "result": (last_frame_tensor,)}
                
            except (RuntimeError, TimeoutError) as e:
                print(e)
                return {"ui": {"video": []}}
            except comfy.model_management.InterruptProcessingException as e:
                raise e

class JimengReferenceImage2Video:
    """根据参考图生成视频。"""
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "client": ("JIMENG_CLIENT",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
            "resolution": (["480p", "720p"], {"default": "720p"}),
            "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
        }, "optional": {
            "save_path": ("STRING", {"default": "Jimeng"}),
            "ref_image_1": ("IMAGE",),
            "ref_image_2": ("IMAGE",),
            "ref_image_3": ("IMAGE",),
            "ref_image_4": ("IMAGE",),
        } }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frame",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, duration, resolution, aspect_ratio, seed, save_path, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "seed"])
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        model = "doubao-seedance-1-0-lite-i2v-250428"
        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} --dur {duration} --seed {actual_seed}"
        content = [{"type": "text", "text": prompt_string}]
        ref_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        for img_tensor in ref_images:
            if img_tensor is not None: content.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img_tensor)}"}, "role": "reference_image" })
        if len(content) == 1: raise ValueError("At least one reference image must be provided.")
        ark_client = client.ark
        async with aiohttp.ClientSession() as session:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                create_result = await asyncio.to_thread( ark_client.content_generation.tasks.create, model=model, content=content, return_last_frame=True )
                task_id = create_result.id
            except Exception as e: 
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to create task: {e}")
            try:
                final_result = await _poll_task_until_completion_async(ark_client, task_id)
                video_url = final_result.content.video_url
                last_frame_url = getattr(final_result.content, 'last_frame_url', None)

                ui_data = await _download_and_save_video_async(session, video_url, "Jimeng_Ref-I2V", actual_seed, save_path)
                
                last_frame_tensor = await _download_url_to_image_tensor_async(session, last_frame_url)
                
                return {"ui": ui_data["ui"], "result": (last_frame_tensor,)}

            except (RuntimeError, TimeoutError) as e:
                print(e)
                return {"ui": {"video": []}}
            except comfy.model_management.InterruptProcessingException as e:
                raise e

class JimengTaskStatusChecker:
    """手动查询任务ID的状态和结果。"""
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "client": ("JIMENG_CLIENT",), "task_id": ("STRING", {"forceInput": True}), } }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "last_frame_url", "status", "error_message", "model", "created_at", "updated_at")
    FUNCTION = "check_status"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    async def check_status(self, client, task_id):
        if not task_id: return ("", "", "no task_id provided", "Error: Task ID is empty.", "", "", "")
        ark_client = client.ark
        try:
            get_result = await asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=task_id)
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

NODE_CLASS_MAPPINGS = {
    "JimengAPIClient": JimengAPIClient,
    "JimengSeedream3": JimengSeedream3,
    "JimengSeedream4": JimengSeedream4,
    "JimengVideoGeneration": JimengVideoGeneration,
    "JimengReferenceImage2Video": JimengReferenceImage2Video,
    "JimengTaskStatusChecker": JimengTaskStatusChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengAPIClient": "Jimeng API Client",
    "JimengSeedream3": "Jimeng Seedream 3",
    "JimengSeedream4": "Jimeng Seedream 4",
    "JimengVideoGeneration": "Jimeng Video Generation",
    "JimengReferenceImage2Video": "Jimeng Reference to Video",
    "JimengTaskStatusChecker": "Jimeng Task Status Checker",

}