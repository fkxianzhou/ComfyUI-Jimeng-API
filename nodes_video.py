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
RECENT_TASK_COUNT = 5
RECENT_SPIKE_FACTOR = 1.1

NON_BLOCKING_TASK_CACHE = {}

async def _get_api_estimated_time_async(ark_client, model_name: str, duration: int, resolution: str) -> int:
    fallback_time = (int(duration) * DEFAULT_FALLBACK_PER_SEC) + DEFAULT_FALLBACK_BASE
    
    print(f"[JimengAI] Info: Fetching task history for '{model_name}' (duration≈{duration}s, resolution={resolution}) to estimate time...")
    
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

        exact_timings = []
        recent_exact_timings = []
        all_data_points = []

        for item in resp.items:
            if not (item.status == "succeeded" and hasattr(item, 'resolution') and item.resolution == resolution):
                continue
            
            item_duration = getattr(item, 'duration', 0)
            task_time = item.updated_at - item.created_at

            if task_time <= 0 or item_duration <= 0:
                continue
                
            all_data_points.append((float(item_duration), float(task_time)))

            if item_duration == int(duration):
                exact_timings.append(task_time)
                if len(recent_exact_timings) < RECENT_TASK_COUNT:
                    recent_exact_timings.append(task_time)
        
        if len(exact_timings) >= MIN_DATA_POINTS:
            print(f"[JimengAI] Info: Found {len(exact_timings)} exact duration match(es). Using standard estimation.")
            
            mean = sum(exact_timings) / len(exact_timings)
            variance = sum([(x - mean) ** 2 for x in exact_timings]) / len(exact_timings)
            std_dev = math.sqrt(variance)
            threshold = std_dev * OUTLIER_STD_DEV_FACTOR

            filtered_timings = []
            for t in exact_timings:
                if abs(t - mean) < threshold:
                    filtered_timings.append(t)
            
            if not filtered_timings:
                print(f"[JimengAI] Warning: All exact matches were outliers.")
            else:
                historical_avg_time = sum(filtered_timings) / len(filtered_timings)
                recent_avg_time = 0
                if recent_exact_timings:
                    recent_avg_time = sum(recent_exact_timings) / len(recent_exact_timings)
                
                final_avg_time = historical_avg_time
                if recent_avg_time > historical_avg_time * RECENT_SPIKE_FACTOR:
                    print(f"[JimengAI] Info: Recent tasks (avg {int(recent_avg_time)}s) are slower than historical avg ({int(historical_avg_time)}s).")
                    print(f"[JimengAI] Info: Adjusting estimate to {int(recent_avg_time)}s due to high API load.")
                    final_avg_time = recent_avg_time
                
                print(f"[JimengAI] Info: Estimated generation time based on {len(filtered_timings)} valid past run(s): {int(final_avg_time)}s")
                return int(final_avg_time)

        print(f"[JimengAI] Info: Not enough exact data for {duration}s. Attempting linear regression from {len(all_data_points)} other tasks...")

        if len(all_data_points) < MIN_DATA_POINTS:
            print(f"[JimengAI] Info: Not enough data for regression. Using fallback estimate.")
            return fallback_time

        all_times = [t for d, t in all_data_points]
        mean_t = sum(all_times) / len(all_times)
        std_dev_t = math.sqrt(sum([(t - mean_t) ** 2 for t in all_times]) / len(all_times))
        threshold_t = std_dev_t * OUTLIER_STD_DEV_FACTOR
        
        filtered_data_points = []
        for d, t in all_data_points:
            if abs(t - mean_t) < threshold_t:
                filtered_data_points.append((d, t))

        if len(filtered_data_points) < MIN_DATA_POINTS:
            print(f"[JimengAI] Info: Not enough regression data after outlier removal. Using fallback estimate.")
            return fallback_time

        x_list = [d for d, t in filtered_data_points]
        y_list = [t for d, t in filtered_data_points]
        n = float(len(x_list))
        mean_x = sum(x_list) / n
        mean_y = sum(y_list) / n
        
        numer = sum((x_list[i] - mean_x) * (y_list[i] - mean_y) for i in range(int(n)))
        denom = sum((x_list[i] - mean_x) ** 2 for i in range(int(n)))

        if denom == 0:
            print(f"[JimengAI] Warning: Regression failed (zero variance in duration). Using fallback estimate.")
            return fallback_time

        m = numer / denom
        b = mean_y - (m * mean_x)
        
        predicted_time = m * float(duration) + b
        
        if predicted_time < b or predicted_time < DEFAULT_FALLBACK_BASE:
             print(f"[JimengAI] Info: Regression predicted an invalid time ({int(predicted_time)}s). Clamping to base cost.")
             predicted_time = max(b, DEFAULT_FALLBACK_BASE)

        print(f"[JimengAI] Info: Regression model (m={m:.2f}s/dur, b={b:.2f}s) estimated {int(predicted_time)}s for {duration}s.")
        return int(predicted_time)

    except Exception as e:
        print(f"[JimengAI] Warning: Failed to fetch or analyze task history: {e}. Using fallback estimate.")
        return fallback_time

async def _download_and_save_video_async_return_path(session: aiohttp.ClientSession, video_url: str, filename_prefix: str, seed: int | None, save_path: str) -> str | None:
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
    for i in text_params:
        if f"--{i}" in prompt: raise ValueError(f"Parameter '--{i}' is not allowed in the prompt. Please use the node's widget for this value.")

def _calculate_duration_and_frames_args(duration: float) -> (str, int):
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
    start_time = time.time()
    info_printed = False
    overtime_warning_printed = False
    
    if estimated_max <= 0: 
        estimated_max = 1 
        
    ps_instance = PromptServer.instance 
    current_max = estimated_max

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
                    final_elapsed = int(time.time() - start_time)
                    print(f"[JimengAI] Task {task_id}: Succeeded. (Total time: {final_elapsed}s)          ")
                    return get_result
                elif get_result.status in ["failed", "cancelled"]:
                    current_max = max(estimated_max, int(time.time() - start_time))
                    if node_id and ps_instance:
                        ps_instance.send_sync("progress", {"value": 0, "max": current_max, "node": node_id})
                    
                    error = getattr(get_result, 'error', None)
                    
                    if error: 
                        raise RuntimeError(f"[JimengAI] Task {task_id} Failed. Code: {error.code}, Message: {error.message}")
                    else: 
                        raise RuntimeError(f"[JimengAI] Task {task_id} Failed. Status: {get_result.status}")
            except Exception as e:
                if isinstance(e, RuntimeError): raise e
                if isinstance(e, comfy.model_management.InterruptProcessingException): raise e
                
                if not info_printed:
                    print(f"[JimengAI] Task {task_id}: Waiting for task details from API...")
                    info_printed = True
                
                print(f"Failed to get task status, retrying... Error: {e}")
            
            elapsed = time.time() - start_time
            current_progress_value = int(elapsed)
            
            if current_progress_value > current_max and not overtime_warning_printed:
                print(f"\n[JimengAI] Warning: Task {task_id} is taking longer than estimated ({current_max}s). Continuing to wait...")
                overtime_warning_printed = True
            
            value_to_send = min(current_progress_value, current_max - 1)

            if node_id and ps_instance:
                ps_instance.send_sync("progress", {
                    "value": value_to_send,
                    "max": current_max,
                    "node": node_id
                })
            
            if info_printed:
                if not overtime_warning_printed:
                    print(f"[JimengAI] Polling Task {task_id}: {current_progress_value}s / {current_max}s elapsed...", end="\r")
                else:
                    print(f"[JimengAI] Polling Task {task_id}: {current_progress_value}s / {current_max}s (Est.) elapsed...", end="\r")

            await asyncio.sleep(interval)
        
        current_max = max(estimated_max, int(time.time() - start_time))
        print(f"\n[JimengAI] Task {task_id}: Polling Timed Out.                                                ")
        if node_id and ps_instance: 
            ps_instance.send_sync("progress", {"value": 0, "max": current_max, "node": node_id})
        raise TimeoutError(f"Task polling timed out after {timeout} seconds for task_id: {task_id}")
    
    except comfy.model_management.InterruptProcessingException as e:
        current_max = max(estimated_max, int(time.time() - start_time))
        print(f"\n[JimengAI] Task {task_id}: Interrupted by user.                                             ")
        if node_id and ps_instance: 
            ps_instance.send_sync("progress", {"value": 0, "max": current_max, "node": node_id})

        
        print(f"[JimengAI] Info: Interruption detected for task {task_id}. Attempting to cancel task on API...")
        try:
            await asyncio.to_thread(client.content_generation.tasks.delete, task_id=task_id)
            print(f"[JimengAI] Info: Sent cancellation request for task {task_id}.")
        except Exception as delete_e:
            print(f"[JimengAI] Warning: Failed to send cancellation request for task {task_id}. This is OK. Error: {delete_e}")
        
        raise e


class JimengVideoGeneration:
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    NON_BLOCKING_TASK_CACHE = NON_BLOCKING_TASK_CACHE
    
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
                "non_blocking": ("BOOLEAN", {"default": False}),
            },
            "optional": {
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

    def _create_failure_json(self, error_message, task_id=None):
        print(f"[JimengAI] Error: {error_message}")
        raise RuntimeError(f"[JimengAI] Task {task_id or 'N/A'} failed: {error_message}")

    def _create_pending_json(self, status, task_id):
        print(f"[JimengAI] Info: Task {task_id} is {status}. Run again to check results.")
        
        status_message_map = {
            "submitted": "Task submitted. Execution paused. Please run the workflow again to check status.",
            "queued": "Task is queued (waiting). Execution paused. Please run the workflow again in a moment.",
            "pending": "Task is pending (in queue). Execution paused. Please run the workflow again in a moment.",
            "running": "Task is actively running. Execution paused. Please run the workflow again in a moment.",
            "processing": "Task is actively processing. Execution paused. Please run the workflow again in a moment.",
        }
        
        specific_message = status_message_map.get(status, 
            f"Task is in an intermediate state ({status}). Execution paused. Please run the workflow again in a moment."
        )
        
        message = f"[JimengAI] Task {task_id}: {specific_message}"
        raise RuntimeError(message)
        
    async def _handle_task_success_async(self, final_result, session):
        video_url = final_result.content.video_url
        last_frame_url = getattr(final_result.content, 'last_frame_url', None)
        
        seed_from_api = getattr(final_result, 'seed', random.randint(0, 4294967295))

        filename_prefix = "Jimeng_VideoGen"
        video_path = await _download_and_save_video_async_return_path(session, video_url, filename_prefix, seed_from_api, "Jimeng")
        
        if video_path is None:
            raise RuntimeError("Failed to download or save video from API.")

        last_frame_tensor = await _download_url_to_image_tensor_async(session, last_frame_url)
        
        usage_info = getattr(final_result, 'usage', None)
        
        output_response = {
            "task_id": final_result.id,
            "model": final_result.model,
            "status": final_result.status,
            "seed": seed_from_api,
            "resolution": getattr(final_result, 'resolution', None),
            "ratio": getattr(final_result, 'ratio', None),
            "duration": getattr(final_result, 'duration', None),
            "framespersecond": getattr(final_result, 'framespersecond', None),
            "usage": usage_info.model_dump() if usage_info else None,
            "created_at": datetime.datetime.fromtimestamp(final_result.created_at).strftime('%Y-%m-%d %H:%M:%S'),
            "updated_at": datetime.datetime.fromtimestamp(final_result.updated_at).strftime('%Y-%m-%d %H:%M:%S'),
        }
        return (VideoFromFile(video_path), last_frame_tensor, json.dumps(output_response, indent=2))

    async def generate(self, client, model_choice, prompt, duration, resolution, aspect_ratio, camerafixed, seed, non_blocking=False, image=None, last_frame_image=None, node_id=None):
        ark_client = client.ark
        cached_task = self.NON_BLOCKING_TASK_CACHE.get(node_id)
        
        async with aiohttp.ClientSession() as session:
            if non_blocking and cached_task:
                task_id = cached_task["task_id"]
                print(f"[JimengAI] Checking status for non-blocking task: {task_id}")
                
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    get_result = await asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=task_id)
                    
                    status = get_result.status
                    
                    if status == "succeeded":
                        print(f"[JimengAI] Task {task_id} succeeded. Fetching results.")
                        del self.NON_BLOCKING_TASK_CACHE[node_id]
                        return await self._handle_task_success_async(get_result, session)
                    
                    elif status in ["failed", "cancelled"]:
                        del self.NON_BLOCKING_TASK_CACHE[node_id]
                        error = getattr(get_result, 'error', None)
                        error_msg = f"Code: {error.code}, Message: {error.message}" if error else "Failed with no error details."
                        self._create_failure_json(error_msg, task_id)
                        
                    else:
                        self._create_pending_json(status, task_id)

                except Exception as e:
                    if isinstance(e, comfy.model_management.InterruptProcessingException):
                        raise e
                    if isinstance(e, RuntimeError):
                        raise e
                    del self.NON_BLOCKING_TASK_CACHE[node_id]
                    self._create_failure_json(f"API Error checking status for {task_id}: {e}", task_id)
                
                return

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

            comfy.model_management.throw_exception_if_processing_interrupted()
            
            try:
                create_result = await asyncio.to_thread(ark_client.content_generation.tasks.create, model=final_model_name, content=content, return_last_frame=True)
                task_id = create_result.id
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to create task: {e}")

            if non_blocking:
                self.NON_BLOCKING_TASK_CACHE[node_id] = {"task_id": task_id}
                print(f"[JimengAI] Non-Blocking task submitted. ID: {task_id}")
                self._create_pending_json("submitted", task_id)
            
            else:
                try:
                    estimated_max_time = await _get_api_estimated_time_async(ark_client, final_model_name, estimation_duration, resolution)
                    final_result = await _poll_task_until_completion_async(
                        ark_client, task_id, node_id, estimated_max_time, final_model_name, estimation_duration
                    )
                    return await self._handle_task_success_async(final_result, session)
                except (RuntimeError, TimeoutError) as e:
                    raise e
                except comfy.model_management.InterruptProcessingException as e:
                    raise e


class JimengReferenceImage2Video:
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    NON_BLOCKING_TASK_CACHE = NON_BLOCKING_TASK_CACHE
    
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
            "non_blocking": ("BOOLEAN", {"default": False}),
        }, "optional": {
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

    def _create_failure_json(self, error_message, task_id=None):
        print(f"[JimengAI] Error: {error_message}")
        raise RuntimeError(f"[JimengAI] Task {task_id or 'N/A'} failed: {error_message}")

    def _create_pending_json(self, status, task_id):
        print(f"[JimengAI] Info: Task {task_id} is {status}. Run again to check results.")
        
        status_message_map = {
            "submitted": "Task submitted. Execution paused. Please run the workflow again to check status.",
            "queued": "Task is queued (waiting). Execution paused. Please run the workflow again in a moment.",
            "pending": "Task is pending (in queue). Execution paused. Please run the workflow again in a moment.",
            "running": "Task is actively running. Execution paused. Please run the workflow again in a moment.",
            "processing": "Task is actively processing. Execution paused. Please run the workflow again in a moment.",
        }
        
        specific_message = status_message_map.get(status, 
            f"Task is in an intermediate state ({status}). Execution paused. Please run the workflow again in a moment."
        )
        
        message = f"[JimengAI] Task {task_id}: {specific_message}"
        raise RuntimeError(message)
        
    async def _handle_task_success_async(self, final_result, session):
        video_url = final_result.content.video_url
        last_frame_url = getattr(final_result.content, 'last_frame_url', None)

        seed_from_api = getattr(final_result, 'seed', random.randint(0, 4294967295))

        video_path = await _download_and_save_video_async_return_path(session, video_url, "Jimeng_Ref-I2V", seed_from_api, "Jimeng")

        if video_path is None:
            raise RuntimeError("Failed to download or save video from API.")

        last_frame_tensor = await _download_url_to_image_tensor_async(session, last_frame_url)
        
        usage_info = getattr(final_result, 'usage', None)

        output_response = {
            "task_id": final_result.id,
            "model": final_result.model,
            "status": final_result.status,
            "seed": seed_from_api,
            "resolution": getattr(final_result, 'resolution', None),
            "ratio": getattr(final_result, 'ratio', None),
            "duration": getattr(final_result, 'duration', None),
            "framespersecond": getattr(final_result, 'framespersecond', None),
            "usage": usage_info.model_dump() if usage_info else None,
            "created_at": datetime.datetime.fromtimestamp(final_result.created_at).strftime('%Y-%m-%d %H:%M:%S'),
            "updated_at": datetime.datetime.fromtimestamp(final_result.updated_at).strftime('%Y-%m-%d %H:%M:%S'),
        }
        return (VideoFromFile(video_path), last_frame_tensor, json.dumps(output_response, indent=2))

    async def generate(self, client, prompt, duration, resolution, aspect_ratio, seed, non_blocking=False, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None, node_id=None):
        ark_client = client.ark
        cached_task = self.NON_BLOCKING_TASK_CACHE.get(node_id)
        
        async with aiohttp.ClientSession() as session:
            if non_blocking and cached_task:
                task_id = cached_task["task_id"]
                print(f"[JimengAI] Checking status for non-blocking task: {task_id}")
                
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    get_result = await asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=task_id)
                    
                    status = get_result.status
                    
                    if status == "succeeded":
                        print(f"[JimengAI] Task {task_id} succeeded. Fetching results.")
                        del self.NON_BLOCKING_TASK_CACHE[node_id]
                        return await self._handle_task_success_async(get_result, session)
                    
                    elif status in ["failed", "cancelled"]:
                        del self.NON_BLOCKING_TASK_CACHE[node_id]
                        error = getattr(get_result, 'error', None)
                        error_msg = f"Code: {error.code}, Message: {error.message}" if error else "Failed with no error details."
                        self._create_failure_json(error_msg, task_id)
                        
                    else:
                        self._create_pending_json(status, task_id)

                except Exception as e:
                    if isinstance(e, comfy.model_management.InterruptProcessingException):
                        raise e
                    if isinstance(e, RuntimeError):
                        raise e
                    del self.NON_BLOCKING_TASK_CACHE[node_id]
                    self._create_failure_json(f"API Error checking status for {task_id}: {e}", task_id)

                return

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
            
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            try:
                create_result = await asyncio.to_thread( ark_client.content_generation.tasks.create, model=model, content=content, return_last_frame=True )
                task_id = create_result.id
            except Exception as e: 
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to create task: {e}")

            if non_blocking:
                self.NON_BLOCKING_TASK_CACHE[node_id] = {"task_id": task_id}
                print(f"[JimengAI] Non-Blocking task submitted. ID: {task_id}")
                self._create_pending_json("submitted", task_id)
            
            else:
                try:
                    estimated_max_time = await _get_api_estimated_time_async(ark_client, model, estimation_duration, resolution) 
                    final_result = await _poll_task_until_completion_async(
                        ark_client, task_id, node_id, estimated_max_time, model, estimation_duration
                    )
                    return await self._handle_task_success_async(final_result, session)
                except (RuntimeError, TimeoutError) as e:
                    raise e
                except comfy.model_management.InterruptProcessingException as e:
                    raise e


class JimengQueryTasks:
    MODELS = [
        "all", 
        "doubao-seedance-1-0-pro-250528", 
        "doubao-seedance-1-0-pro-fast-251015",
        "doubao-seedance-1-0-lite-t2v-250428",
        "doubao-seedance-1-0-lite-i2v-250428"
    ]
    STATUSES = ["all", "succeeded", "failed", "running", "queued", "cancelled"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "page_num": ("INT", {"default": 1, "min": 1, "max": 500}),
                "page_size": ("INT", {"default": 10, "min": 1, "max": 500}),
                "status": (s.STATUSES, {"default": "all"}),
            },
            "optional": {
                "task_ids": ("STRING", {"default": "", "multiline": True}),
                "model_choice": (s.MODELS, {"default": "all"}),
                "custom_model_id": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("task_list_json", "total_tasks")
    FUNCTION = "query_tasks"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    async def query_tasks(self, client, page_num, page_size, status, task_ids=None, model_choice="all", custom_model_id=None):
        ark_client = client.ark
        
        kwargs = {
            "page_num": page_num,
            "page_size": page_size
        }

        final_model = None
        if custom_model_id and custom_model_id.strip():
            final_model = custom_model_id.strip()
            print(f"[JimengAI] Info: Using custom model ID: {final_model}")
        elif model_choice != "all":
            final_model = model_choice
        
        if final_model:
            kwargs["model"] = final_model

        if status != "all":
            kwargs["status"] = status
        
        if task_ids and task_ids.strip():
            id_list = [tid.strip() for tid in task_ids.split('\n') if tid.strip()]
            if id_list:
                kwargs["task_ids"] = id_list
        
        try:
            print(f"[JimengAI] Querying tasks with params: {kwargs}")
            resp = await asyncio.to_thread(
                ark_client.content_generation.tasks.list,
                **kwargs
            )
            
            items_list = [item.model_dump() for item in resp.items]
            
            # Convert timestamps to readable format
            for item_dict in items_list:
                if 'created_at' in item_dict and isinstance(item_dict['created_at'], (int, float)):
                    item_dict['created_at'] = datetime.datetime.fromtimestamp(item_dict['created_at']).strftime('%Y-%m-%d %H:%M:%S')
                if 'updated_at' in item_dict and isinstance(item_dict['updated_at'], (int, float)):
                    item_dict['updated_at'] = datetime.datetime.fromtimestamp(item_dict['updated_at']).strftime('%Y-%m-%d %H:%M:%S')

            items_json = json.dumps(items_list, indent=2, ensure_ascii=False)
            total = resp.total
            
            print(f"[JimengAI] Query successful. Found {total} total tasks, returning {len(items_list)} items.")
            return (items_json, total)
            
        except Exception as e:
            error_msg_str = f"Failed to query tasks: {e}"
            print(f"[JimengAI] Error: {error_msg_str}")
            error_msg = json.dumps({"error": error_msg_str}, indent=2, ensure_ascii=False)
            return (error_msg, 0)


NODE_CLASS_MAPPINGS = {
    "JimengVideoGeneration": JimengVideoGeneration,
    "JimengReferenceImage2Video": JimengReferenceImage2Video,
    "JimengQueryTasks": JimengQueryTasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengVideoGeneration": "Jimeng Video Generation",
    "JimengReferenceImage2Video": "Jimeng Reference to Video",
    "JimengQueryTasks": "Jimeng Query Tasks",
}