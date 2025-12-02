import os
import io
import time
import random
import datetime
import asyncio
import aiohttp
import json
import math
import logging
import shutil
import ast

import folder_paths
import comfy.model_management
from server import PromptServer
import torch
import PIL.Image
import numpy

from .nodes_shared import (
    GLOBAL_CATEGORY, 
    _image_to_base64, 
    _download_url_to_image_tensor_async, 
    _fetch_data_from_url_async,
    log_msg,
    get_text
)

logging.getLogger("volcenginesdkarkruntime").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    class VideoFromFile:
        def __init__(self, *args, **kwargs):
            raise ImportError("Failed to import 'comfy_api'. Video output nodes may not work.")

DEFAULT_FALLBACK_PER_SEC = 12
DEFAULT_FALLBACK_BASE = 20
HISTORY_PAGE_SIZE = 50
MIN_DATA_POINTS = 3
OUTLIER_STD_DEV_FACTOR = 2.0
RECENT_TASK_COUNT = 5
RECENT_SPIKE_FACTOR = 1.1

NON_BLOCKING_TASK_CACHE = {}

async def _get_api_estimated_time_async(ark_client, model_name: str, duration: int, resolution: str) -> (int, str):
    # 异步获取 API 任务的预估完成时间
    fallback_time = (int(duration) * DEFAULT_FALLBACK_PER_SEC) + DEFAULT_FALLBACK_BASE
    
    try:
        resp = await asyncio.to_thread(
            ark_client.content_generation.tasks.list,
            status="succeeded",
            model=model_name,
            page_size=HISTORY_PAGE_SIZE
        )

        if not resp.items:
            return (fallback_time, "est_fallback")

        exact_timings = []
        recent_exact_timings = []
        all_data_points = []

        for item in resp.items:
            if not (item.status == "succeeded" and hasattr(item, 'resolution') and item.resolution == resolution):
                continue
            
            item_duration = getattr(item, 'duration', 0)

            t_start = item.created_at
            t_end = item.updated_at
            if hasattr(t_start, 'timestamp'): t_start = t_start.timestamp()
            if hasattr(t_end, 'timestamp'): t_end = t_end.timestamp()
            
            raw_diff = float(t_end) - float(t_start)

            try:
                local_offset = datetime.datetime.now().astimezone().utcoffset().total_seconds()
            except Exception:
                local_offset = 0

            fixed_diff = raw_diff - local_offset

            if fixed_diff > 0 and abs(fixed_diff) < abs(raw_diff):
                task_time = fixed_diff
            else:
                task_time = raw_diff

            if task_time <= 0 or item_duration <= 0:
                continue
                
            all_data_points.append((float(item_duration), float(task_time)))

            if item_duration == int(duration):
                exact_timings.append(task_time)
                if len(recent_exact_timings) < RECENT_TASK_COUNT:
                    recent_exact_timings.append(task_time)
        
        if len(exact_timings) >= MIN_DATA_POINTS:
            mean = sum(exact_timings) / len(exact_timings)
            variance = sum([(x - mean) ** 2 for x in exact_timings]) / len(exact_timings)
            std_dev = math.sqrt(variance)
            threshold = std_dev * OUTLIER_STD_DEV_FACTOR

            filtered_timings = [t for t in exact_timings if abs(t - mean) < threshold]
            
            if not filtered_timings:
                return (fallback_time, "est_fallback")
            
            historical_avg_time = sum(filtered_timings) / len(filtered_timings)
            recent_avg_time = 0
            if recent_exact_timings:
                recent_avg_time = sum(recent_exact_timings) / len(recent_exact_timings)
            
            if recent_avg_time > historical_avg_time * RECENT_SPIKE_FACTOR:
                return (int(recent_avg_time), "est_recent")
            
            return (int(historical_avg_time), "est_history")

        if len(all_data_points) < MIN_DATA_POINTS:
            return (fallback_time, "est_fallback")

        all_times = [t for d, t in all_data_points]
        mean_t = sum(all_times) / len(all_times)
        std_dev_t = math.sqrt(sum([(t - mean_t) ** 2 for t in all_times]) / len(all_times))
        threshold_t = std_dev_t * OUTLIER_STD_DEV_FACTOR
        
        filtered_data_points = [(d, t) for d, t in all_data_points if abs(t - mean_t) < threshold_t]

        if len(filtered_data_points) < MIN_DATA_POINTS:
            return (fallback_time, "est_fallback")

        x_list = [d for d, t in filtered_data_points]
        y_list = [t for d, t in filtered_data_points]
        n = float(len(x_list))
        mean_x = sum(x_list) / n
        mean_y = sum(y_list) / n
        
        numer = sum((x_list[i] - mean_x) * (y_list[i] - mean_y) for i in range(int(n)))
        denom = sum((x_list[i] - mean_x) ** 2 for i in range(int(n)))

        if denom == 0:
            return (fallback_time, "est_fallback")

        m = numer / denom
        b = mean_y - (m * mean_x)
        predicted_time = m * float(duration) + b
        
        if predicted_time < b or predicted_time < DEFAULT_FALLBACK_BASE:
             predicted_time = max(b, DEFAULT_FALLBACK_BASE)

        return (int(predicted_time), "est_regression")

    except Exception:
        return (fallback_time, "est_fallback")

async def _download_to_temp_async(session: aiohttp.ClientSession, url: str, filename_prefix: str, seed: int | None, save_path: str, file_ext: str) -> (str | None, bytes | None):
    # 下载到 TEMP 目录并返回绝对路径和数据
    if not url: return (None, None)
    if seed is not None: filename_prefix = f"{filename_prefix}_seed_{seed}"
    output_dir = folder_paths.get_temp_directory()
    if save_path: output_dir = os.path.join(output_dir, save_path)
    (full_output_folder, filename, _, _, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
    os.makedirs(full_output_folder, exist_ok=True)
    final_filename = f"{filename}_{random.randint(1, 10000)}.{file_ext}"
    final_path = os.path.join(full_output_folder, final_filename)
    try:
        data = await _fetch_data_from_url_async(session, url)
        with open(final_path, "wb") as f: f.write(data)
        return (final_path, data)
    except Exception as e:
        log_msg("err_download_url", url=url, e=e)
        return (None, None)

async def _download_and_save_video_async_return_path(session: aiohttp.ClientSession, video_url: str, filename_prefix: str, seed: int | None, save_path: str) -> str | None:
    # 下载视频到 TEMP 目录并返回绝对路径
    if not video_url: return None
    file_ext = video_url.split('.')[-1].split('?')[0] or "mp4"
    (final_path, _) = await _download_to_temp_async(session, video_url, filename_prefix, seed, save_path, file_ext)
    return final_path

async def _download_and_save_frame_async_return_tensor_and_path(session: aiohttp.ClientSession, frame_url: str, filename_prefix: str, seed: int | None, save_path: str) -> (torch.Tensor | None, str | None):
    # 下载帧到 TEMP 目录并返回 Tensor 和绝对路径
    if not frame_url: return (None, None)
    file_ext = frame_url.split('.')[-1].split('?')[0] or "jpg"
    (final_path, data) = await _download_to_temp_async(session, frame_url, filename_prefix, seed, save_path, file_ext)
    if final_path is None or data is None: return (None, None)
    try:
        i = PIL.Image.open(io.BytesIO(data))
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        return (torch.from_numpy(image)[None,], final_path)
    except Exception as e:
        log_msg("err_convert_tensor", e=e)
        return (None, final_path)

def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    # 检查提示词中是否包含了不允许的文本参数
    for i in text_params:
        if f"--{i}" in prompt: 
            raise ValueError(get_text("popup_param_not_allowed").format(param=i))

def _calculate_duration_and_frames_args(duration: float) -> (str, int):
    # 根据浮点数时长计算API所需的帧数参数
    if duration == int(duration):
        return (f"--dur {int(duration)}", int(duration))
    else:
        target_frames = duration * 24.0
        n = round((target_frames - 25.0) / 4.0)
        final_frames = int(max(29, min(289, 25 + 4 * n)))
        return (f"--frames {final_frames}", int(round(final_frames / 24.0)))

class JimengVideoBase:
    # 视频生成节点的并发逻辑基类
    NON_BLOCKING_TASK_CACHE = NON_BLOCKING_TASK_CACHE
    
    def _log_batch_task_failure(self, error_message, task_id=None):
        log_msg("err_task_fail_msg", tid=task_id or 'N/A', msg=error_message)

    def _create_failure_json(self, error_message, task_id=None):
        # 打印日志到控制台
        print(f"[JimengAI] Error: {error_message}")
        
        # 根据是否有 Task ID 获取对应的本地化文本
        if task_id:
            display_msg = get_text("popup_task_failed").format(task_id=task_id, msg=error_message)
        else:
            display_msg = get_text("popup_req_failed").format(msg=error_message)

        raise RuntimeError(display_msg)

    def _create_pending_json(self, status, task_id=None, task_count=0):
        if task_count > 0:
            msg = get_text("popup_batch_pending").format(count=task_count)
        else:
            msg = get_text("popup_task_pending").format(task_id=task_id, status=status)
        raise RuntimeError(msg)

    async def _handle_batch_success_async(self, successful_tasks: list, output_save_path: str, generation_count: int, save_last_frame_batch: bool, session: aiohttp.ClientSession):
        # 处理所有成功的任务
        if generation_count > 1:
            log_msg("batch_handling", count=len(successful_tasks))
        
        temp_save_path = "Jimeng"
        video_prefix = "Jimeng_Vid_Temp"
        frame_prefix = "Jimeng_Frame_Temp"
        tasks_with_info = []

        for task in successful_tasks:
            video_url = task.content.video_url
            last_frame_url = getattr(task.content, 'last_frame_url', None)
            seed_from_api = getattr(task, 'seed', random.randint(0, 4294967295))
            
            video_coro = _download_and_save_video_async_return_path(session, video_url, video_prefix, seed_from_api, temp_save_path)
            frame_coro = _download_and_save_frame_async_return_tensor_and_path(session, last_frame_url, frame_prefix, seed_from_api, temp_save_path)
            
            output_response = {
                "task_id": task.id,
                "model": task.model,
                "status": task.status,
                "seed": seed_from_api,
                "resolution": getattr(task, 'resolution', None),
                "video_url": video_url,
            }
            tasks_with_info.append((seed_from_api, task, video_coro, frame_coro, output_response))

        tasks_with_info.sort(key=lambda x: x[0])
        
        download_coroutines = []
        for info in tasks_with_info:
            download_coroutines.append(info[2]) 
            download_coroutines.append(info[3]) 
        
        download_results = await asyncio.gather(*download_coroutines, return_exceptions=True)

        all_responses = []
        first_video_output = None
        first_frame_output = None
        files_to_copy = [] 

        video_results = [download_results[i] for i in range(0, len(download_results), 2)]
        frame_results = [download_results[i] for i in range(1, len(download_results), 2)]

        for i in range(len(tasks_with_info)):
            all_responses.append(tasks_with_info[i][4])
            video_path_or_exc = video_results[i]
            (frame_tensor_or_exc, frame_temp_path_or_exc) = frame_results[i]
            task_id = tasks_with_info[i][1].id
            seed = tasks_with_info[i][0]

            if isinstance(video_path_or_exc, Exception):
                log_msg("err_download_url", url="video", e=video_path_or_exc)
            elif isinstance(video_path_or_exc, str):
                if first_video_output is None: first_video_output = VideoFromFile(video_path_or_exc)
                files_to_copy.append((video_path_or_exc, seed, "video"))
            
            if isinstance(frame_tensor_or_exc, torch.Tensor):
                if first_frame_output is None: first_frame_output = frame_tensor_or_exc
            if isinstance(frame_temp_path_or_exc, str):
                 files_to_copy.append((frame_temp_path_or_exc, seed, "frame"))

        if generation_count > 1:
            log_msg("batch_copying", path=output_save_path)
            output_dir_base = folder_paths.get_output_directory()
            for absolute_temp_path, seed, file_type in files_to_copy:
                if file_type == "frame" and not save_last_frame_batch: continue
                try:
                    if not os.path.exists(absolute_temp_path): continue
                    prefix = "Jimeng_Vid" if file_type == "video" else "Jimeng_Frame"
                    prefix_with_seed = f"{prefix}_seed_{seed}"
                    (full_output_folder, filename, _, _, _) = folder_paths.get_save_image_path(os.path.join(output_save_path, prefix_with_seed), output_dir_base)
                    os.makedirs(full_output_folder, exist_ok=True)
                    ext = ".mp4" if file_type == "video" else ".jpg"
                    final_output_path = os.path.join(full_output_folder, f"{filename}_{random.randint(1, 10000)}{ext}")
                    shutil.copy2(absolute_temp_path, final_output_path)
                except Exception as e:
                    log_msg("err_copy_fail", path=absolute_temp_path, e=e)

        return (first_video_output, first_frame_output, json.dumps(all_responses, indent=2))
        
    async def _execute_batch_generation(self, client, model_name, content, estimation_duration, resolution, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id, poll_interval=2):
        # 核心执行器：处理任务提交、轮询或状态检查
        ark_client = client.ark
        ps_instance = PromptServer.instance
        cached_data = self.NON_BLOCKING_TASK_CACHE.get(node_id)
            
        if non_blocking and cached_data:
            task_ids = cached_data["task_ids"]
            log_msg("check_status", count=len(task_ids))
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                get_coroutines = [asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=tid) for tid in task_ids]
                results = await asyncio.gather(*get_coroutines, return_exceptions=True)
                
                successful_tasks, failed_tasks_info, pending_tasks = [], [], []
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        failed_tasks_info.append((task_ids[i], str(res)))
                    elif res.status == "succeeded":
                        successful_tasks.append(res)
                    elif res.status in ["failed", "cancelled"]:
                        fail_reason = "Failed"
                        if hasattr(res, 'error') and res.error:
                            if hasattr(res.error, 'message'): fail_reason = res.error.message
                            elif isinstance(res.error, dict) and 'message' in res.error: fail_reason = res.error['message']
                            else: fail_reason = str(res.error)
                        failed_tasks_info.append((res.id, fail_reason))
                    else:
                        pending_tasks.append(res)

                for tid, error_msg in failed_tasks_info:
                    self._log_batch_task_failure(error_msg, tid)
                
                if pending_tasks:
                    self._create_pending_json(pending_tasks[0].status, pending_tasks[0].id, len(pending_tasks))
                else:
                    del self.NON_BLOCKING_TASK_CACHE[node_id]
                    if not successful_tasks:
                        if failed_tasks_info:
                            first_tid, first_msg = failed_tasks_info[0]
                            self._create_failure_json(first_msg, task_id=first_tid)
                        else:
                            self._create_failure_json("Batch failed: No tasks succeeded.")
                    
                    ret_results = None
                    async with aiohttp.ClientSession() as session:
                        ret_results = await self._handle_batch_success_async(successful_tasks, batch_save_path, generation_count, save_last_frame_batch, session)
                    
                    await asyncio.sleep(0.1)
                    return ret_results

            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException): raise e
                if str(e).startswith("[JimengAI]"): raise e
                del self.NON_BLOCKING_TASK_CACHE[node_id]
                log_msg("err_check_status_batch", e=e)
                self._create_failure_json(f"API Error: {e}")
        
        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_name)
        
        create_coroutines = []
        for _ in range(generation_count):
            create_coroutines.append(asyncio.to_thread(ark_client.content_generation.tasks.create, model=model_name, content=content, return_last_frame=True))
        
        comfy.model_management.throw_exception_if_processing_interrupted()
        results = await asyncio.gather(*create_coroutines, return_exceptions=True)

        tasks_to_poll = []
        creation_errors = []  
        creation_failed_count = 0
        
        for res in results:
            if isinstance(res, Exception):
                log_msg("err_task_create", e=res)
                creation_errors.append(res) 
                creation_failed_count += 1
            else:
                tasks_to_poll.append(res)

        if not tasks_to_poll:
            log_msg("err_batch_fail_all")
            final_error_msg = "All tasks failed on creation."
            if creation_errors:
                raw_error_str = str(creation_errors[0])
                final_error_msg = raw_error_str 
                try:
                    start_idx = raw_error_str.find('{')
                    end_idx = raw_error_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        dict_str = raw_error_str[start_idx : end_idx + 1]
                        error_data = ast.literal_eval(dict_str)
                        if isinstance(error_data, dict):
                            if 'error' in error_data and isinstance(error_data['error'], dict) and 'message' in error_data['error']:
                                final_error_msg = error_data['error']['message']
                            elif 'message' in error_data:
                                final_error_msg = error_data['message']
                except Exception:
                    pass
            self._create_failure_json(final_error_msg)
        
        if generation_count > 1:
            log_msg("batch_submit_result", created=len(tasks_to_poll), failed=creation_failed_count)

        if non_blocking:
            task_ids = [t.id for t in tasks_to_poll]
            self.NON_BLOCKING_TASK_CACHE[node_id] = {"task_ids": task_ids}
            self._create_pending_json("submitted", task_ids[0], len(task_ids))
        
        estimated_max_time, method_key = await _get_api_estimated_time_async(ark_client, model_name, estimation_duration, resolution)
        if estimated_max_time <= 0: estimated_max_time = 1
        
        method_name = get_text(method_key)
        
        log_msg("task_submitted_est", time=estimated_max_time, method=method_name)
        
        if generation_count == 1 and tasks_to_poll:
             log_msg("task_info_simple", task_id=tasks_to_poll[0].id, model=model_name)
        
        start_time = time.time()
        current_max = estimated_max_time
        successful_tasks = []
        failed_tasks_info = []
        tasks_to_poll_ids = [t.id for t in tasks_to_poll]
        total_tasks = len(tasks_to_poll_ids)

        try:
            while tasks_to_poll_ids:
                comfy.model_management.throw_exception_if_processing_interrupted()
                get_coroutines = [asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=tid) for tid in tasks_to_poll_ids]
                results = await asyncio.gather(*get_coroutines, return_exceptions=True)
                
                next_poll_ids = []
                for i, res in enumerate(results):
                    current_task_id = tasks_to_poll_ids[i]
                    if isinstance(res, Exception):
                        next_poll_ids.append(current_task_id) 
                    elif res.status == "succeeded":
                        successful_tasks.append(res)
                    elif res.status in ["failed", "cancelled"]:
                        fail_reason = "Failed"
                        if hasattr(res, 'error') and res.error:
                            if hasattr(res.error, 'message'):
                                fail_reason = res.error.message
                            elif isinstance(res.error, dict) and 'message' in res.error:
                                fail_reason = res.error['message']
                            else:
                                fail_reason = str(res.error)
                        elif res.status == "cancelled":
                            fail_reason = "Cancelled"
                            
                        failed_tasks_info.append((current_task_id, fail_reason)) 
                    else:
                        next_poll_ids.append(current_task_id) 
                
                tasks_to_poll_ids = next_poll_ids
                elapsed = time.time() - start_time
                value_to_send = min(int(elapsed), current_max - 1)
                if elapsed > current_max: current_max = int(elapsed) + 1 

                if node_id and ps_instance:
                    ps_instance.send_sync("progress", {"value": value_to_send, "max": current_max, "node": node_id})

                if tasks_to_poll_ids:
                    if generation_count == 1:
                        print(get_text("polling_single").format(task_id=tasks_to_poll_ids[0], elapsed=int(elapsed), max=current_max), end="\r")
                    else:
                        done_count = len(successful_tasks) + len(failed_tasks_info)
                        print(get_text("polling_batch").format(done=done_count, total=total_tasks, pending=len(tasks_to_poll_ids), elapsed=int(elapsed), max=current_max), end="\r")
                    await asyncio.sleep(poll_interval)
                else:
                    break 
        except comfy.model_management.InterruptProcessingException as e:
            log_msg("interrupted")
            cancel_coroutines = [asyncio.to_thread(ark_client.content_generation.tasks.delete, task_id=tid) for tid in tasks_to_poll_ids]
            await asyncio.gather(*cancel_coroutines, return_exceptions=True) 
            raise e 
        finally:
            print()
            if node_id and ps_instance:
                ps_instance.send_sync("progress", {"value": 0, "max": current_max, "node": node_id})

        for tid, error_msg in failed_tasks_info:
            self._log_batch_task_failure(error_msg, tid)

        if generation_count == 1:
            if successful_tasks:
                log_msg("task_finished_single")
        else:
            log_msg("batch_finished_stats", success=len(successful_tasks), failed=len(failed_tasks_info))
        
        if not successful_tasks:
             if failed_tasks_info:
                 first_tid, first_msg = failed_tasks_info[0]
                 self._create_failure_json(first_msg, task_id=first_tid)
             else:
                 self._create_failure_json("Batch failed: No tasks succeeded.")

        ret_results = None
        async with aiohttp.ClientSession() as session:
            ret_results = await self._handle_batch_success_async(successful_tasks, batch_save_path, generation_count, save_last_frame_batch, session)
        
        await asyncio.sleep(0.1)
        return ret_results

    async def _common_generation_logic(self, client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id, model_name, content, forbidden_params, prompt_extra_args=""):
        # 通用生成逻辑封装
        try:
            _raise_if_text_params(prompt, forbidden_params)
            api_seed = -1
            if seed > 0: api_seed = seed
            (duration_or_frames_arg, estimation_duration) = _calculate_duration_and_frames_args(duration)
            prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} {duration_or_frames_arg} --seed {api_seed} {prompt_extra_args.strip()}"
            content.insert(0, {"type": "text", "text": prompt_string})
            comfy.model_management.throw_exception_if_processing_interrupted()
            effective_save_path = "Jimeng" 
            if generation_count > 1: effective_save_path = batch_save_path
            
            return await self._execute_batch_generation(
                client=client, model_name=model_name, content=content, estimation_duration=estimation_duration, resolution=resolution,
                generation_count=generation_count, batch_save_path=effective_save_path, save_last_frame_batch=save_last_frame_batch,
                non_blocking=non_blocking, node_id=node_id
            )
        except Exception as e:
            if isinstance(e, comfy.model_management.InterruptProcessingException): raise e
            
            s_e = str(e)
            if s_e.startswith("[JimengAI]"):
                raise e
            
            raise RuntimeError(get_text("popup_prepare_failed").format(e=e))

class JimengVideoGeneration(JimengVideoBase):
    # 即梦视频生成节点
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "model_choice": (["doubao-seedance-1-0-pro", "doubao-seedance-1-0-pro-fast", "doubao-seedance-1-0-lite"], {"default": "doubao-seedance-1-0-pro"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.208, "max": 12.042, "step": (4/24), "display": "number"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "generation_count": ("INT", {"default": 1, "min": 1}),
                "batch_save_path": ("STRING", {"default": "Jimeng/Video"}),
                "save_last_frame_batch": ("BOOLEAN", {"default": False}),
                "non_blocking": ("BOOLEAN", {"default": False}),
            },
            "optional": { "image": ("IMAGE",), "last_frame_image": ("IMAGE",) },
            "hidden": { "node_id": "UNIQUE_ID" }
        }
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate"
    OUTPUT_NODE = True 
    CATEGORY = GLOBAL_CATEGORY
    
    async def generate(self, client, model_choice, prompt, duration, resolution, aspect_ratio, camerafixed, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking=False, image=None, last_frame_image=None, node_id=None):
        final_model_name = ""
        if model_choice == "doubao-seedance-1-0-pro": final_model_name = "doubao-seedance-1-0-pro-250528"
        elif model_choice == "doubao-seedance-1-0-pro-fast": final_model_name = "doubao-seedance-1-0-pro-fast-251015"
        elif model_choice == "doubao-seedance-1-0-lite":
            if image is None: final_model_name = "doubao-seedance-1-0-lite-t2v-250428"
            else: final_model_name = "doubao-seedance-1-0-lite-i2v-250428"
        
        prompt_extra_args = f"--camerafixed {'true' if camerafixed else 'false'}"
        content = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(image)}"}, "role": "first_frame"})
        if last_frame_image is not None:
            if image is None: raise ValueError(get_text("popup_first_frame_missing"))
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(last_frame_image)}"}, "role": "last_frame"})
            
        return await self._common_generation_logic(
            client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id,
            model_name=final_model_name, content=content, forbidden_params=["resolution", "ratio", "dur", "frames", "camerafixed", "seed"], prompt_extra_args=prompt_extra_args
        )

class JimengReferenceImage2Video(JimengVideoBase):
    # 参考图生视频节点
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "client": ("JIMENG_CLIENT",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "duration": ("FLOAT", {"default": 5.0, "min": 1.208, "max": 12.042, "step": (4/24), "display": "number"}),
            "resolution": (["480p", "720p"], {"default": "720p"}),
            "aspect_ratio": (s.ASPECT_RATIOS, {"default": "adaptive"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
            "generation_count": ("INT", {"default": 1, "min": 1}),
            "batch_save_path": ("STRING", {"default": "Jimeng/Video"}),
            "save_last_frame_batch": ("BOOLEAN", {"default": False}),
            "non_blocking": ("BOOLEAN", {"default": False}),
        }, "optional": { "ref_image_1": ("IMAGE",), "ref_image_2": ("IMAGE",), "ref_image_3": ("IMAGE",), "ref_image_4": ("IMAGE",), },
        "hidden": { "node_id": "UNIQUE_ID" } }
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "response")
    FUNCTION = "generate"
    OUTPUT_NODE = True 
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking=False, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None, node_id=None):
        content = []
        for img in [ref_image_1, ref_image_2, ref_image_3, ref_image_4]:
            if img is not None: content.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img)}"}, "role": "reference_image" })
        if len(content) == 0: raise ValueError(get_text("popup_ref_missing"))
        
        return await self._common_generation_logic(
            client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id,
            model_name="doubao-seedance-1-0-lite-i2v-250428", content=content, forbidden_params=["resolution", "ratio", "dur", "frames", "seed"], prompt_extra_args=""
        )

class JimengQueryTasks:
    # 任务查询节点
    MODELS = ["all", "doubao-seedance-1-0-pro-250528", "doubao-seedance-1-0-pro-fast-251015", "doubao-seedance-1-0-lite-t2v-250428", "doubao-seedance-1-0-lite-i2v-250428"]
    STATUSES = ["all", "succeeded", "failed", "running", "queued", "cancelled"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "client": ("JIMENG_CLIENT",), "page_num": ("INT", {"default": 1}), "page_size": ("INT", {"default": 10}), "status": (s.STATUSES, {"default": "all"}),},
            "optional": { "task_ids": ("STRING", {"default": ""}), "model_choice": (s.MODELS, {"default": "all"}), "custom_model_id": ("STRING", {"default": ""}), }
        }
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("task_list_json", "total_tasks")
    FUNCTION = "query_tasks"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    async def query_tasks(self, client, page_num, page_size, status, task_ids=None, model_choice="all", custom_model_id=None):
        ark_client = client.ark
        kwargs = { "page_num": page_num, "page_size": page_size }
        if custom_model_id and custom_model_id.strip(): kwargs["model"] = custom_model_id.strip()
        elif model_choice != "all": kwargs["model"] = model_choice
        if status != "all": kwargs["status"] = status
        if task_ids and task_ids.strip(): kwargs["task_ids"] = [tid.strip() for tid in task_ids.split('\n') if tid.strip()]
        try:
            resp = await asyncio.to_thread(ark_client.content_generation.tasks.list, **kwargs)
            items_list = [item.model_dump() for item in resp.items]
            for item_dict in items_list:
                if 'created_at' in item_dict and isinstance(item_dict['created_at'], (int, float)):
                    item_dict['created_at'] = datetime.datetime.fromtimestamp(item_dict['created_at']).strftime('%Y-%m-%d %H:%M:%S')
                if 'updated_at' in item_dict and isinstance(item_dict['updated_at'], (int, float)):
                    item_dict['updated_at'] = datetime.datetime.fromtimestamp(item_dict['updated_at']).strftime('%Y-%m-%d %H:%M:%S')
            return (json.dumps(items_list, indent=2, ensure_ascii=False), resp.total)
        except Exception as e:
            return (json.dumps({"error": str(e)}), 0)

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