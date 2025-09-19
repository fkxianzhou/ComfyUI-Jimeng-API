# --- 1. 导入所需模块 ---
import os
import io
import time
import base64
import random
import datetime
import asyncio
import aiohttp

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

# --- 3. 辅助函数 ---

async def _fetch_data_from_url_async(session: aiohttp.ClientSession, url: str) -> bytes:
    """【异步】从给定的URL下载数据，需要从外部传入一个 aiohttp.ClientSession 对象。"""
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
    """【异步】下载视频，保存到ComfyUI的输出文件夹，并返回用于UI预览的字典。"""
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
        if f"--{i}" in prompt: raise ValueError(f"参数 '--{i}' 不应出现在提示词中。请使用节点上对应的控件来设置此值。")

async def _poll_task_until_completion_async(client, task_id: str, timeout=600, interval=5):
    """【异步】轮询任务状态，直到任务完成、失败或超时。"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            get_result = await asyncio.to_thread(client.content_generation.tasks.get, task_id=task_id)
            if get_result.status == "succeeded": return get_result
            elif get_result.status in ["failed", "cancelled"]:
                error = getattr(get_result, 'error', None)
                if error: raise RuntimeError(f"任务失败，代码: {error.code}，信息: {error.message}")
                else: raise RuntimeError(f"任务失败，状态: {get_result.status}")
        except Exception as e:
            if isinstance(e, RuntimeError): raise e
            print(f"获取任务状态失败，正在重试... 错误: {e}")
        await asyncio.sleep(interval)
    raise TimeoutError(f"任务轮询超时（{timeout}秒），任务ID: {task_id}")

# --- 4. 节点类定义 ---

class JimengAPIClient:
    """节点功能：创建API客户端，供后续节点使用。"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"api_key": ("STRING", {"multiline": False, "default": ""})}}
    
    RETURN_TYPES = ("JIMENG_OPENAI_CLIENT", "JIMENG_ARK_CLIENT")
    RETURN_NAMES = ("openai_client", "ark_client")
    FUNCTION = "create_clients"
    CATEGORY = GLOBAL_CATEGORY

    def create_clients(self, api_key):
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
                if image_tensor is None: raise RuntimeError("下载生成的图片失败。")
                return (image_tensor, actual_seed)
            except Exception as e:
                raise RuntimeError(f"生成图片失败: {e}")

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
                if image_tensor is None: raise RuntimeError("下载编辑后的图片失败。")
                return (image_tensor, actual_seed)
            except Exception as e:
                raise RuntimeError(f"编辑图片失败: {e}")

class JimengSeedream4:
    """节点功能：使用Seedream4模型进行文生图、图生图，并支持单图/组图模式。"""
    RECOMMENDED_SIZES = [ "2048x2048 (1:1)", "2304x1728 (4:3)", "1728x2304 (3:4)", "2560x1440 (16:9)", "1440x2560 (9:16)", "2496x1664 (3:2)", "1664x2496 (2:3)", "3024x1296 (21:9)", "4096x4096 (1:1)" ]
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "client": ("JIMENG_OPENAI_CLIENT",), "prompt": ("STRING", {"multiline": True, "default": ""}), "generation_mode": (["生成单图 (disabled)", "生成组图 (auto)"],), "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}), "size": (s.RECOMMENDED_SIZES,), "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}), "watermark": ("BOOLEAN", {"default": False}), }, "optional": { "images": ("IMAGE",), } }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, generation_mode, max_images, size, seed, watermark, images=None):
        sequential_image_generation = generation_mode.split(" ")[1].strip("()")
        n_input_images = 0
        if images is not None: n_input_images = images.shape[0]
        if n_input_images > 10: raise ValueError("输入图片数量不能超过10张。")
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(f"输入图片数量({n_input_images}) + 最大生成数量({max_images}) 的总和不能超过15。")

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
                if not valid_tensors: raise RuntimeError("下载所有生成的图片均失败。")
                return (torch.cat(valid_tensors, dim=0), actual_seed)
            except Exception as e:
                raise RuntimeError(f"使用Seedream 4生成失败: {e}")

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
                raise RuntimeError(f"创建任务失败: {e}")
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
            except Exception as e: raise RuntimeError(f"创建任务失败: {e}")
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
        if len(content) == 1: raise ValueError("至少需要提供一张参考图片。")
        async with aiohttp.ClientSession() as session:
            try:
                create_result = await asyncio.to_thread( client.content_generation.tasks.create, model=model, content=content )
                task_id = create_result.id
            except Exception as e: raise RuntimeError(f"创建任务失败: {e}")
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
        if not task_id: return ("", "", "no task_id provided", "错误: 任务ID为空。", "", "", "")
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
                error_message = f"代码: {get_result.error.code}, 信息: {get_result.error.message}"
            return (video_url, last_frame_url, status, error_message, model, created_at, updated_at)
        except Exception as e:
            return ("", "", "api_error", f"API错误: {e}", "", "", "")

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

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengAPIClient": "即梦API客户端 (Jimeng Client)",
    "JimengText2Image": "即梦文生图 (Seedream 3)",
    "JimengImageEdit": "即梦图像编辑 (Seededit 3)",
    "JimengSeedream4": "即梦高级图像 (Seedream 4)",
    "JimengVideoGeneration": "即梦视频生成",
    "JimengFirstLastFrame2Video": "即梦首尾帧生视频 (F&L-I2V)",
    "JimengReferenceImage2Video": "即梦参考图生视频 (Ref-I2V)",
    "JimengTaskStatusChecker": "即梦任务状态检查器",
}