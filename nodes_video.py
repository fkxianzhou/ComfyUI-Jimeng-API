import os
import io
import time
import base64
import random
import datetime
import asyncio
import aiohttp
import json
import math
import logging

import folder_paths
import comfy.model_management
from server import PromptServer

from .nodes_shared import (
    GLOBAL_CATEGORY, 
    _image_to_base64, 
    _download_url_to_image_tensor_async,
    _fetch_data_from_url_async
)

logging.getLogger("volcenginesdkarkruntime").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    print(f"[JimengAI] Warning: 'comfy_api' not found. Video output nodes may not function correctly.")
    print(f"[JimengAI] Info: This is normal if you are running ComfyUI in --disable-api-save-load mode.")
    class VideoFromFile:
        def __init__(self, *args, **kwargs):
            raise ImportError("Failed to import 'comfy_api.input_impl.VideoFromFile'. Please ensure comfy_api is available.")


DEFAULT_FALLBACK_PER_SEC = 12
DEFAULT_FALLBACK_BASE = 20
HISTORY_PAGE_SIZE = 50
MIN_DATA_POINTS = 3
OUTLIER_STD_DEV_FACTOR = 2.0 

async def _get_api_estimated_time_async(ark_client, model_name: str, duration: int) -> int:
    """
    【异步】通过查询 API 历史任务来获取预估时间。
    会自动过滤掉与当前参数不符的任务，并剔除统计异常值。
    """
    fallback_time = (int(duration) * DEFAULT_FALLBACK_PER_SEC) + DEFAULT_FALLBACK_BASE
    
    print(f"[JimengAI] Info: Fetching task history for '{model_name}' (duration≈{duration}s) to estimate time...")
    
    try:
        resp = await asyncio.to_thread(
            ark_client.content_generation.tasks.list,
            status="succeeded",
            model=model_name,
            page_size=HISTORY_PAGE_SIZE
        )

        if not resp.items:
            print(f"[JimengAI] Info: No recent task history found for '{model_name}'. Using fallback estimate.")
            return fallback_time

        timings = []
        for item in resp.items:
            if item.status == "succeeded" and hasattr(item, 'duration') and item.duration == int(duration):
                task_time = item.updated_at - item.created_at
                if task_time > 0:
                    timings.append(task_time)
        
        if len(timings) < MIN_DATA_POINTS:
            print(f"[JimengAI] Info: Not enough historical data (<{MIN_DATA_POINTS} runs) found for duration {duration}s. Using fallback estimate.")
            return fallback_time

        mean = sum(timings) / len(timings)
        variance = sum([(x - mean) ** 2 for x in timings]) / len(timings)
        std_dev = math.sqrt(variance)

        filtered_timings = []
        outlier_count = 0
        threshold = std_dev * OUTLIER_STD_DEV_FACTOR

        for t in timings:
            if abs(t - mean) < threshold:
                filtered_timings.append(t)
            else:
                outlier_count += 1
        
        if outlier_count > 0:
            print(f"[JimengAI] Info: Ignored {outlier_count} abnormally long task(s) (e.g., review, server lag) from estimation.")

        if not filtered_timings:
            print(f"[JimengAI] Warning: All historical data was filtered as outliers. Using fallback estimate.")
            return fallback_time

        final_avg_time = sum(filtered_timings) / len(filtered_timings)
        print(f"[JimengAI] Info: Estimated generation time based on {len(filtered_timings)} valid past run(s): {int(final_avg_time)}s")
        return int(final_avg_time)

    except Exception as e:
        print(f"[JimengAI] Warning: Failed to fetch or analyze task history: {e}. Using fallback estimate.")
        return fallback_time


async def _download_and_save_video_async_return_path(session: aiohttp.ClientSession, video_url: str, filename_prefix: str, seed: int | None, save_path: str) -> str | None:
    """【异步】下载视频并保存到临时目录，返回完整文件路径。"""
    if not video_url: return None
    if seed is not None: filename_prefix = f"{filename_prefix}_seed_{seed}"

    output_dir = folder_paths.get_temp_directory()
    
    if save_path:
        output_dir = os.path.join(output_dir, save_path)
        
    (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
    
    os.makedirs(full_output_folder, exist_ok=True)
    
    file_ext = video_url.split('.')[-1].split('?')[0]
    if not file_ext: file_ext = "mp4"
    
    final_filename = f"{filename}_{random.randint(1, 10000)}.{file_ext}"
    final_path = os.path.join(full_output_folder, final_filename)
    
    try:
        data = await _fetch_data_from_url_async(session, video_url)
        with open(final_path, "wb") as f: f.write(data)
        return final_path
    except Exception as e:
        print(f"[JimengAI] Error: Failed to download or save video to temp path: {final_path}. Error: {e}")
        return None


def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    """检查提示词中是否包含了不应出现的命令行风格参数。"""
    for i in text_params:
        if f"--{i}" in prompt: raise ValueError(f"Parameter '--{i}' is not allowed in the prompt. Please use the node's widget for this value.")

def _calculate_duration_and_frames_args(duration: float) -> (str, int):
    """
    根据输入的时长（可以是小数），计算出 --dur 或 --frames 参数，
    并返回一个用于历史估计的整数时长。

    返回: (str: api_argument, int: estimation_duration)
    """
    if duration == int(duration):
        int_duration = int(duration)
        return (f"--dur {int_duration}", int_duration)
    else:
        FPS = 24.0
        target_frames = duration * FPS
        
        n = round((target_frames - 25.0) / 4.0)
        
        final_frames = 25 + 4 * n
        
        final_frames = int(max(29, min(289, final_frames)))
        
        api_arg = f"--frames {final_frames}"
        
        calculated_duration_sec = final_frames / FPS
        estimation_duration = int(round(calculated_duration_sec))
        
        print(f"[JimengAI] Info: Duration {duration}s requested. Using closest valid frames: {final_frames} (~{calculated_duration_sec:.3f}s).")
        
        return (api_arg, estimation_duration)


async def _poll_task_until_completion_async(client, task_id: str, node_id: str, estimated_max: int, model_name: str, duration: int, timeout=600, interval=5):
    """【异步】轮询任务状态，支持中断和模拟进度。"""
    start_time = time.time()
    info_printed = False
    
    if estimated_max <= 0: 
        estimated_max = 1 
        
    ps_instance = PromptServer.instance 

    try:
        while time.time() - start_time < timeout:
            comfy.model_management.throw_exception_if_processing_interrupted()

            try:
                get_result = await asyncio.to_thread(client.content_generation.tasks.get, task_id=task_id)
                
                if not info_printed:
                    task_created_at_str = "N/A"
                    if hasattr(get_result, 'created_at') and get_result.created_at:
                        task_created_at_str = datetime.datetime.fromtimestamp(get_result.created_at).strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"[JimengAI] Task submitted to API. Polling started.")
                    print(f"    - TASK ID:    {task_id}")
                    print(f"    - Created at: {task_created_at_str}")
                    print(f"    - Model:      {model_name}")
                    print(f"    - Duration:   ~{duration}s") 
                    print(f"    - Est. Time:  {estimated_max}s")
                    info_printed = True

                if get_result.status == "succeeded":
                    print(f"[JimengAI] Task {task_id}: Succeeded. (Total time: {int(time.time() - start_time)}s)          ")
                    if node_id and ps_instance:
                        ps_instance.send_sync("progress", {"value": estimated_max, "max": estimated_max, "node": node_id})
                    return get_result
                elif get_result.status in ["failed", "cancelled"]:
                    print(f"[JimengAI] Task {task_id}: Failed or Cancelled.                                                ")
                    if node_id and ps_instance:
                        ps_instance.send_sync("progress", {"value": 0, "max": estimated_max, "node": node_id})
                    
                    error = getattr(get_result, 'error', None)
                    if error: raise RuntimeError(f"Task failed with code: {error.code}, message: {error.message}")
                    else: raise RuntimeError(f"Task failed with status: {get_result.status}")
            except Exception as e:
                if isinstance(e, RuntimeError): raise e
                if isinstance(e, comfy.model_management.InterruptProcessingException): raise e
                
                if not info_printed:
                    print(f"[JimengAI] Task {task_id}: Waiting for task details from API...")
                    info_printed = True
                
                print(f"Failed to get task status, retrying... Error: {e}")
            
            elapsed = time.time() - start_time
            current_progress_value = min(int(elapsed), estimated_max) 
            
            if node_id and ps_instance:
                ps_instance.send_sync("progress", { 
                    "value": current_progress_value,
                    "max": estimated_max,
                    "node": node_id
                })
            
            if info_printed:
                print(f"[JimengAI] Polling Task {task_id}: {current_progress_value}s / {estimated_max}s elapsed...", end="\r")

            await asyncio.sleep(interval)
        
        print(f"\n[JimengAI] Task {task_id}: Polling Timed Out.                                                ")
        if node_id and ps_instance: 
            ps_instance.send_sync("progress", {"value": 0, "max": estimated_max, "node": node_id})
        raise TimeoutError(f"Task polling timed out after {timeout} seconds for task_id: {task_id}")
    
    except comfy.model_management.InterruptProcessingException as e:
        print(f"\n[JimengAI] Task {task_id}: Interrupted by user.                                             ")
        if node_id and ps_instance: 
            ps_instance.send_sync("progress", {"value": 0, "max": estimated_max, "node": node_id})
        
        print(f"[JimengAI] Info: Interruption detected for task {task_id}. Attempting to cancel task on API...")
        try:
            await asyncio.to_thread(client.content_generation.tasks.delete, task_id=task_id)
            print(f"[JimengAI] Info: Sent cancellation request for task {task_id}.")
        except Exception as delete_e:
            print(f"[JimengAI] Warning: Failed to send cancellation request for task {task_id}. This is OK. Error: {delete_e}")
        
        raise e


class JimengVideoGeneration:
    """使用 doubao-seedance 模型生成视频。"""
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "model_choice": (["doubao-seedance-1-0-pro", "doubao-seedance-1-0-pro-fast", "doubao-seedance-1-0-lite"], {"default": "doubao-seedance-1-0-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.208,
                    "max": 12.042,
                    "step": (4/24),
                    "display": "number"
                }),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
            },
            "optional": {
                "save_path_in_temp": ("STRING", {"default": "Jimeng"}),
                "image": ("IMAGE",),
                "last_frame_image": ("IMAGE",)
            },
            "hidden": { "node_id": "UNIQUE_ID" }
        }
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate"
    OUTPUT_NODE = False 
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, model_choice, prompt, duration, resolution, aspect_ratio, camerafixed, seed, save_path_in_temp, image=None, last_frame_image=None, node_id=None):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "frames", "camerafixed", "seed"])

        final_model_name = ""
        if model_choice == "doubao-seedance-1-0-pro":
            final_model_name = "doubao-seedance-1-0-pro-250528"
        elif model_choice == "doubao-seedance-1-0-pro-fast":
            final_model_name = "doubao-seedance-1-0-pro-fast-251015"
        elif model_choice == "doubao-seedance-1-0-lite":
            if image is None:
                final_model_name = "doubao-seedance-1-0-lite-t2v-250428"
            else:
                final_model_name = "doubao-seedance-1-0-lite-i2v-250428"
        
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        
        (duration_or_frames_arg, estimation_duration) = _calculate_duration_and_frames_args(duration)
        
        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} {duration_or_frames_arg} --camerafixed {'true' if camerafixed else 'false'} --seed {actual_seed}"
        
        content = [{"type": "text", "text": prompt_string}]
        
        if image is not None:
            first_frame_b64 = _image_to_base64(image)
            first_frame_url = f"data:image/jpeg;base64,{first_frame_b64}"
            content.append({"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"})
        
        if last_frame_image is not None:
            if model_choice == "doubao-seedance-1-0-pro-fast":
                raise ValueError(f"Model '{model_choice}' does not support last_frame_image. Please disconnect the last_frame_image input.")
            if image is None:
                raise ValueError("A first frame image must be provided when using a last frame image.")
            last_frame_b64 = _image_to_base64(last_frame_image)
            last_frame_url = f"data:image/jpeg;base64,{last_frame_b64}"
            content.append({"type": "image_url", "image_url": {"url": last_frame_url}, "role": "last_frame"})

        ark_client = client.ark

        estimated_max_time = await _get_api_estimated_time_async(ark_client, final_model_name, estimation_duration)
        
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
                final_result = await _poll_task_until_completion_async(
                    ark_client, task_id, node_id, estimated_max_time, final_model_name, estimation_duration
                )
                
                video_url = final_result.content.video_url
                last_frame_url = getattr(final_result.content, 'last_frame_url', None)
                
                filename_prefix = "Jimeng_VideoGen"
                video_path = await _download_and_save_video_async_return_path(session, video_url, filename_prefix, actual_seed, save_path_in_temp)
                
                if video_path is None:
                    raise RuntimeError("Failed to download or save video from API.")

                last_frame_tensor = await _download_url_to_image_tensor_async(session, last_frame_url)
                
                return (VideoFromFile(video_path), last_frame_tensor, final_result.model_dump_json())
                
            except (RuntimeError, TimeoutError) as e:
                print(e)
                raise e
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
            "duration": ("FLOAT", {
                "default": 5.0,
                "min": 1.208,
                "max": 12.042,
                "step": (4/24),
                "display": "number"
            }),
            "resolution": (["480p", "720p"], {"default": "720p"}),
            "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
        }, "optional": {
            "save_path_in_temp": ("STRING", {"default": "Jimeng"}),
            "ref_image_1": ("IMAGE",),
            "ref_image_2": ("IMAGE",),
            "ref_image_3": ("IMAGE",),
            "ref_image_4": ("IMAGE",),
        },
        "hidden": { "node_id": "UNIQUE_ID" } 
      }
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, duration, resolution, aspect_ratio, seed, save_path_in_temp, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None, node_id=None):
        _raise_if_text_params(prompt, ["resolution", "ratio", "dur", "frames", "seed"])
        
        actual_seed = random.randint(0, 4294967295) if seed == -1 else seed
        model = "doubao-seedance-1-0-lite-i2v-250428"
        
        (duration_or_frames_arg, estimation_duration) = _calculate_duration_and_frames_args(duration)

        prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} {duration_or_frames_arg} --seed {actual_seed}"
        
        content = [{"type": "text", "text": prompt_string}]
        ref_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        for img_tensor in ref_images:
            if img_tensor is not None: content.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img_tensor)}"}, "role": "reference_image" })
        if len(content) == 1: raise ValueError("At least one reference image must be provided.")
        
        ark_client = client.ark

        estimated_max_time = await _get_api_estimated_time_async(ark_client, model, estimation_duration) 
        
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
                final_result = await _poll_task_until_completion_async(
                    ark_client, task_id, node_id, estimated_max_time, model, estimation_duration
                )
                
                video_url = final_result.content.video_url
                last_frame_url = getattr(final_result.content, 'last_frame_url', None)

                video_path = await _download_and_save_video_async_return_path(session, video_url, "Jimeng_Ref-I2V", actual_seed, save_path_in_temp)

                if video_path is None:
                    raise RuntimeError("Failed to download or save video from API.")

                last_frame_tensor = await _download_url_to_image_tensor_async(session, last_frame_url)
                
                return (VideoFromFile(video_path), last_frame_tensor, final_result.model_dump_json())

            except (RuntimeError, TimeoutError) as e:
                print(e)
                raise e
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
    "JimengVideoGeneration": JimengVideoGeneration,
    "JimengReferenceImage2Video": JimengReferenceImage2Video,
    "JimengTaskStatusChecker": JimengTaskStatusChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengVideoGeneration": "Jimeng Video Generation",
    "JimengReferenceImage2Video": "Jimeng Reference to Video",
    "JimengTaskStatusChecker": "Jimeng Task Status Checker",
}