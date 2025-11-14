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
import shutil 

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
    _fetch_data_from_url_async
)

# 设置相关日志记录器的级别，减少不必要的输出
logging.getLogger("volcenginesdkarkruntime").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    print(f"[JimengAI] Warning: 'comfy_api' not found. Video output nodes may not function correctly.")
    print(f"[JimengAI] Info: This is normal if you are running ComfyUI in --disable-api-save-load mode.")
    class VideoFromFile:
        # 定义一个备用的 VideoFromFile 类，以便在导入失败时提供信息
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
    # 异步获取API任务的预估完成时间
    fallback_time = (int(duration) * DEFAULT_FALLBACK_PER_SEC) + DEFAULT_FALLBACK_BASE
    
    print(f"[JimengAI] Info: Fetching task history for '{model_name}' (duration≈{duration}s, resolution={resolution}) to estimate time...")
    
    try:
        # 异步线程中获取任务历史
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

        # 遍历历史数据
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
        
        # 如果有足够多的精确匹配数据
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
                # 考虑近期API负载波动
                if recent_avg_time > historical_avg_time * RECENT_SPIKE_FACTOR:
                    print(f"[JimengAI] Info: Recent tasks (avg {int(recent_avg_time)}s) are slower than historical avg ({int(historical_avg_time)}s).")
                    print(f"[JimengAI] Info: Adjusting estimate to {int(recent_avg_time)}s due to high API load.")
                    final_avg_time = recent_avg_time
                
                print(f"[JimengAI] Info: Estimated generation time based on {len(filtered_timings)} valid past run(s): {int(final_avg_time)}s")
                return int(final_avg_time)

        print(f"[JimengAI] Info: Not enough exact data for {duration}s. Attempting linear regression from {len(all_data_points)} other tasks...")

        # 如果没有足够的精确数据，尝试线性回归
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

async def _download_to_temp_async(session: aiohttp.ClientSession, url: str, filename_prefix: str, seed: int | None, save_path: str, file_ext: str) -> (str | None, bytes | None):
    # 下载到 TEMP 目录并返回绝对路径和数据
    if not url: 
        return (None, None)
    
    if seed is not None: 
        filename_prefix = f"{filename_prefix}_seed_{seed}"

    output_dir = folder_paths.get_temp_directory()
    if save_path:
        output_dir = os.path.join(output_dir, save_path)
        
    (full_output_folder, filename, _, _, _) = folder_paths.get_save_image_path(filename_prefix, output_dir)
    os.makedirs(full_output_folder, exist_ok=True)
    
    final_filename = f"{filename}_{random.randint(1, 10000)}.{file_ext}"
    final_path = os.path.join(full_output_folder, final_filename)
    
    try:
        data = await _fetch_data_from_url_async(session, url)
        with open(final_path, "wb") as f: 
            f.write(data)
        return (final_path, data)
    except Exception as e:
        print(f"[JimengAI] Error: Failed to download or save to temp path: {final_path}. Error: {e}")
        return (None, None)

async def _download_and_save_video_async_return_path(session: aiohttp.ClientSession, video_url: str, filename_prefix: str, seed: int | None, save_path: str) -> str | None:
    # 下载视频到 TEMP 目录并返回绝对路径
    if not video_url: 
        return None
    
    file_ext = video_url.split('.')[-1].split('?')[0] or "mp4"
    (final_path, _) = await _download_to_temp_async(session, video_url, filename_prefix, seed, save_path, file_ext)
    
    return final_path

async def _download_and_save_frame_async_return_tensor_and_path(session: aiohttp.ClientSession, frame_url: str, filename_prefix: str, seed: int | None, save_path: str) -> (torch.Tensor | None, str | None):
    # 下载帧到 TEMP 目录并返回 Tensor 和绝对路径
    if not frame_url: 
        return (None, None)
        
    file_ext = frame_url.split('.')[-1].split('?')[0] or "jpg"
    (final_path, data) = await _download_to_temp_async(session, frame_url, filename_prefix, seed, save_path, file_ext)

    if final_path is None or data is None:
        return (None, None)
        
    try:
        # 转换为 Tensor
        i = PIL.Image.open(io.BytesIO(data))
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        
        return (torch.from_numpy(image)[None,], final_path)
    except Exception as e:
        print(f"[JimengAI] Error: Failed to convert downloaded frame to tensor: {e}")
        # 即使转换失败，也可能返回路径，但Tensor为None
        return (None, final_path)


def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    # 检查提示词中是否包含了不允许的文本参数（应使用节点控件输入）
    for i in text_params:
        if f"--{i}" in prompt: raise ValueError(f"Parameter '--{i}' is not allowed in the prompt. Please use the node's widget for this value.")

def _calculate_duration_and_frames_args(duration: float) -> (str, int):
    # 根据浮点数时长计算API所需的--dur或--frames参数及用于估时的整数时长
    if duration == int(duration):
        # 如果是整数
        int_duration = int(duration)
        return (f"--dur {int_duration}", int_duration)
    else:
        # 如果是浮点数，转换为最接近的有效帧数
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


class JimengVideoBase:
    # 视频生成节点的并发逻辑基类
    NON_BLOCKING_TASK_CACHE = NON_BLOCKING_TASK_CACHE
    
    def _log_batch_task_failure(self, error_message, task_id=None):
        # 内部方法：仅打印批量任务中的单个失败，不中断工作流
        print(f"\n[JimengAI] Error: Task {task_id or 'N/A'} failed: {error_message}")

    def _create_failure_json(self, error_message, task_id=None):
        # 内部方法：打印关键错误并抛出运行时异常，中断整个工作流
        print(f"[JimengAI] Error: {error_message}")
        raise RuntimeError(f"[JimengAI] Task {task_id or 'N/A'} failed: {error_message}")

    def _create_pending_json(self, status, task_id=None, task_count=0):
        # 内部方法：为非阻塞模式创建挂起状态的响应，中断工作流
        if task_count > 0:
             print(f"[JimengAI] Info: Batch of {task_count} tasks has pending tasks (e.g., {task_id}). Run again to check results.")
        else:
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
        
        if task_count > 0:
             message = f"[JimengAI] Batch ({task_count} tasks): {specific_message}"
        else:
             message = f"[JimengAI] Task {task_id}: {specific_message}"
        raise RuntimeError(message)

    async def _handle_batch_success_async(self, successful_tasks: list, output_save_path: str, generation_count: int, save_last_frame_batch: bool, session: aiohttp.ClientSession):
        # 处理所有成功的任务：下载到Temp，（可选）复制到Output，并返回第一个结果

        print(f"[JimengAI] Info: Handling {len(successful_tasks)} successful tasks. Sorting by seed...")
        
        # 所有下载都进入 temp/Jimeng 目录
        temp_save_path = "Jimeng"
        video_prefix = "Jimeng_Vid_Temp"
        frame_prefix = "Jimeng_Frame_Temp"

        tasks_with_info = []

        # 1. 收集并准备所有任务
        for task in successful_tasks:
            video_url = task.content.video_url
            last_frame_url = getattr(task.content, 'last_frame_url', None)
            seed_from_api = getattr(task, 'seed', random.randint(0, 4294967295))
            
            video_coro = _download_and_save_video_async_return_path(session, video_url, video_prefix, seed_from_api, temp_save_path)
            frame_coro = _download_and_save_frame_async_return_tensor_and_path(session, last_frame_url, frame_prefix, seed_from_api, temp_save_path)
            
            usage_info = getattr(task, 'usage', None)
            output_response = {
                "task_id": task.id,
                "model": task.model,
                "status": task.status,
                "seed": seed_from_api,
                "resolution": getattr(task, 'resolution', None),
                "ratio": getattr(task, 'ratio', None),
                "duration": getattr(task, 'duration', None),
                "framespersecond": getattr(task, 'framespersecond', None),
                "usage": usage_info.model_dump() if usage_info else None,
                "created_at": datetime.datetime.fromtimestamp(task.created_at).strftime('%Y-%m-%d %H:%M:%S'),
                "updated_at": datetime.datetime.fromtimestamp(task.updated_at).strftime('%Y-%m-%d %H:%M:%S'),
            }
            tasks_with_info.append((seed_from_api, task, video_coro, frame_coro, output_response))

        # 2. 按 Seed 排序
        tasks_with_info.sort(key=lambda x: x[0])
        
        print(f"[JimengAI] Info: Concurrently downloading {len(tasks_with_info)} results to TEMP dir in seed order...")

        # 3. 准备排序后的协程
        download_coroutines = []
        for info in tasks_with_info:
            download_coroutines.append(info[2]) # video_coro
            download_coroutines.append(info[3]) # frame_coro
        
        # 4. 并发执行下载 (到 Temp)
        download_results = await asyncio.gather(*download_coroutines, return_exceptions=True)

        all_responses = []
        first_video_output = None
        first_frame_output = None
        files_to_copy = [] # (absolute_temp_path, seed, type)

        video_results = [download_results[i] for i in range(0, len(download_results), 2)]
        frame_results = [download_results[i] for i in range(1, len(download_results), 2)]

        # 5. 按排序后的顺序处理结果
        for i in range(len(tasks_with_info)):
            all_responses.append(tasks_with_info[i][4]) # 添加排序后的 json
            
            video_path_or_exc = video_results[i]
            (frame_tensor_or_exc, frame_temp_path_or_exc) = frame_results[i]
            task_id = tasks_with_info[i][1].id
            seed = tasks_with_info[i][0]

            # 处理视频下载结果
            if isinstance(video_path_or_exc, Exception):
                self._log_batch_task_failure(f"Failed to download video: {video_path_or_exc}", task_id)
            elif isinstance(video_path_or_exc, str):
                if first_video_output is None:
                    # video_path_or_exc 是 绝对路径
                    first_video_output = VideoFromFile(video_path_or_exc)
                files_to_copy.append((video_path_or_exc, seed, "video"))
            
            # 处理帧下载结果
            if isinstance(frame_tensor_or_exc, Exception):
                self._log_batch_task_failure(f"Failed to download last frame: {frame_tensor_or_exc}", task_id)
            elif isinstance(frame_tensor_or_exc, torch.Tensor):
                if first_frame_output is None:
                    first_frame_output = frame_tensor_or_exc
            
            if isinstance(frame_temp_path_or_exc, str):
                 files_to_copy.append((frame_temp_path_or_exc, seed, "frame"))

        # 6. 将所有文件从 Temp 复制到 Output (仅当批次大于1时)
        if generation_count > 1:
            print(f"[JimengAI] Info: Copying files from Temp to Output dir: {output_save_path}")
            output_dir_base = folder_paths.get_output_directory()
            
            for absolute_temp_path, seed, file_type in files_to_copy:
                
                # 检查是否跳过保存帧
                if file_type == "frame" and not save_last_frame_batch:
                    continue
                
                try:
                    if not os.path.exists(absolute_temp_path):
                        print(f"[JimengAI] Warning: Temp file not found, skipping copy: {absolute_temp_path}")
                        continue
                    
                    prefix = "Jimeng_Vid" if file_type == "video" else "Jimeng_Frame"
                    prefix_with_seed = f"{prefix}_seed_{seed}"
                    
                    # 在 output_save_path (e.g., "Jimeng/Video") 下创建文件
                    (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(
                        os.path.join(output_save_path, prefix_with_seed), 
                        output_dir_base
                    )
                    
                    os.makedirs(full_output_folder, exist_ok=True)
                    
                    ext = ".mp4" if file_type == "video" else ".jpg"
                    final_output_path = os.path.join(full_output_folder, f"{filename}_{random.randint(1, 10000)}{ext}")
                    
                    shutil.copy2(absolute_temp_path, final_output_path)
                    
                except Exception as e:
                    self._log_batch_task_failure(f"Failed to copy file to output: {absolute_temp_path}. Error: {e}")
        else:
            print(f"[JimengAI] Info: Generation count is 1. Skipping batch save to Output directory.")


        print(f"[JimengAI] Info: Batch handling complete. Returning first successful (seed-sorted) video to node output.")
        return (first_video_output, first_frame_output, json.dumps(all_responses, indent=2))
        
    async def _execute_batch_generation(self, client, model_name, content, estimation_duration, resolution, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id, poll_interval=2):
        # 核心执行器：处理任务提交、轮询（阻塞）或状态检查（非阻塞）
        
        ark_client = client.ark
        ps_instance = PromptServer.instance
        cached_data = self.NON_BLOCKING_TASK_CACHE.get(node_id)
            
        # 1. 非阻塞模式 - 状态检查
        if non_blocking and cached_data:
            task_ids = cached_data["task_ids"]
            print(f"[JimengAI] Checking status for non-blocking batch of {len(task_ids)} tasks...")
            
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                get_coroutines = [asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=tid) for tid in task_ids]
                results = await asyncio.gather(*get_coroutines, return_exceptions=True)
                
                successful_tasks, failed_tasks_info, pending_tasks = [], [], []

                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        failed_tasks_info.append((task_ids[i], f"API Error checking status: {res}"))
                    elif res.status == "succeeded":
                        successful_tasks.append(res)
                    elif res.status in ["failed", "cancelled"]:
                        error = getattr(res, 'error', None)
                        error_msg = f"Code: {error.code}, Message: {error.message}" if error else "Failed/Cancelled with no error details."
                        failed_tasks_info.append((res.id, error_msg))
                    else:
                        pending_tasks.append(res)

                # 记录所有失败
                for tid, error_msg in failed_tasks_info:
                    self._log_batch_task_failure(error_msg, tid)
                
                if pending_tasks:
                    # 仍有任务在运行，再次挂起
                    self._create_pending_json(pending_tasks[0].status, pending_tasks[0].id, len(pending_tasks))
                else:
                    # 所有任务都已完成（成功或失败）
                    print(f"[JimengAI] Non-blocking batch finished. {len(successful_tasks)} succeeded, {len(failed_tasks_info)} failed.")
                    del self.NON_BLOCKING_TASK_CACHE[node_id]
                    
                    # 异常处理 (非阻塞)
                    if not successful_tasks:
                        if failed_tasks_info:
                            first_task_id, first_error_msg = failed_tasks_info[0]
                            error_to_raise = f"Task {first_task_id} failed: {first_error_msg}"
                            if generation_count > 1:
                                error_to_raise = f"All {generation_count} tasks failed. First error: {error_to_raise}"
                            self._create_failure_json(error_to_raise, first_task_id)
                        else:
                            self._create_failure_json("Batch failed: No tasks succeeded and no specific errors were reported.", node_id)
                    
                    # 在此处创建会话，仅用于本次下载
                    async with aiohttp.ClientSession() as session:
                        return await self._handle_batch_success_async(successful_tasks, batch_save_path, generation_count, save_last_frame_batch, session)

            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException): raise e
                if isinstance(e, RuntimeError): raise e
                del self.NON_BLOCKING_TASK_CACHE[node_id]
                self._create_failure_json(f"API Error checking batch status: {e}")
        
        # 2. 新任务提交
        print(f"[JimengAI] Submitting batch of {generation_count} tasks to API (Model: {model_name})...")
        create_coroutines = []
        for _ in range(generation_count):
            create_coroutines.append(
                asyncio.to_thread(ark_client.content_generation.tasks.create, model=model_name, content=content, return_last_frame=True)
            )
        
        comfy.model_management.throw_exception_if_processing_interrupted()
        results = await asyncio.gather(*create_coroutines, return_exceptions=True)

        tasks_to_poll = []
        creation_failed_count = 0
        for res in results:
            if isinstance(res, Exception):
                self._log_batch_task_failure(f"Task creation failed: {res}")
                creation_failed_count += 1
            else:
                tasks_to_poll.append(res)

        if not tasks_to_poll:
            self._create_failure_json("All tasks in batch failed on creation.")
        
        print(f"[JimengAI] Batch submitted. {len(tasks_to_poll)} tasks created, {creation_failed_count} failed.")

        # 3. 非阻塞模式 - 缓存并挂起
        if non_blocking:
            task_ids = [t.id for t in tasks_to_poll]
            self.NON_BLOCKING_TASK_CACHE[node_id] = {"task_ids": task_ids}
            print(f"[JimengAI] Non-Blocking batch submitted. IDs: {task_ids}")
            self._create_pending_json("submitted", task_ids[0], len(task_ids))
        
        # 4. 阻塞模式 - 并发轮询
        estimated_max_time = await _get_api_estimated_time_async(ark_client, model_name, estimation_duration, resolution)
        if estimated_max_time <= 0: estimated_max_time = 1
        
        start_time = time.time()
        current_max = estimated_max_time
        info_printed = False 
        
        successful_tasks = []
        failed_tasks_info = []
        tasks_to_poll_ids = [t.id for t in tasks_to_poll]
        total_tasks = len(tasks_to_poll_ids)
        
        if generation_count > 1:
            print(f"[JimengAI] Info: Polling {total_tasks} tasks. Base estimated time per task: {estimated_max_time}s")

        try:
            while tasks_to_poll_ids:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                get_coroutines = [asyncio.to_thread(ark_client.content_generation.tasks.get, task_id=tid) for tid in tasks_to_poll_ids]
                results = await asyncio.gather(*get_coroutines, return_exceptions=True)
                
                if generation_count == 1 and not info_printed:
                    for res in results: 
                        if not isinstance(res, Exception):
                            try:
                                get_result = res
                                task_id = get_result.id
                                task_created_at_str = "N/A"
                                if hasattr(get_result, 'created_at') and get_result.created_at:
                                    task_created_at_str = datetime.datetime.fromtimestamp(get_result.created_at).strftime('%Y-%m-%d %H:%M:%S')
                                
                                print(f"\n[JimengAI] Task submitted to API. Polling started.")
                                print(f"    - TASK ID:    {task_id}")
                                print(f"    - Created at: {task_created_at_str}")
                                print(f"    - Model:      {model_name}")
                                print(f"    - Duration:   ~{estimation_duration}s") 
                                print(f"    - Est. Time:  {estimated_max_time}s")
                                info_printed = True
                                break 
                            except Exception as e:
                                print(f"\n[JimengAI] Warning: Could not print task info: {e}")
                                info_printed = True 
                
                next_poll_ids = []
                
                for i, res in enumerate(results):
                    current_task_id = tasks_to_poll_ids[i]
                    
                    if isinstance(res, Exception):
                        print(f"\n[JimengAI] Warning: Failed to get status for task {current_task_id}, retrying... Error: {res}")
                        next_poll_ids.append(current_task_id) 
                    elif res.status == "succeeded":
                        successful_tasks.append(res)
                    elif res.status in ["failed", "cancelled"]:
                        error = getattr(res, 'error', None)
                        error_msg = f"Code: {error.code}, Message: {error.message}" if error else "Failed/Cancelled"
                        failed_tasks_info.append((current_task_id, error_msg)) 
                    else:
                        next_poll_ids.append(current_task_id) 
                
                tasks_to_poll_ids = next_poll_ids
                elapsed = time.time() - start_time
                
                value_to_send = min(int(elapsed), current_max - 1)
                if elapsed > current_max:
                     current_max = int(elapsed) + 1 

                if node_id and ps_instance:
                    ps_instance.send_sync("progress", {"value": value_to_send, "max": current_max, "node": node_id})

                if tasks_to_poll_ids:
                    if generation_count == 1:
                        if info_printed: 
                            print(f"[JimengAI] Polling Task {tasks_to_poll_ids[0]}: {int(elapsed)}s / {current_max}s elapsed...", end="\r")
                    else:
                        completed_tasks = len(successful_tasks) + len(failed_tasks_info)
                        print(f"[JimengAI] Polling Batch: {completed_tasks}/{total_tasks} done. {len(tasks_to_poll_ids)} pending... ({int(elapsed)}s / {current_max}s Est.)", end="\r")

                    await asyncio.sleep(poll_interval)
                else:
                    break 

        except comfy.model_management.InterruptProcessingException as e:
            print(f"\n[JimengAI] Batch Interrupted by user. Attempting to cancel {len(tasks_to_poll_ids)} pending tasks...")
            cancel_coroutines = [asyncio.to_thread(ark_client.content_generation.tasks.delete, task_id=tid) for tid in tasks_to_poll_ids]
            await asyncio.gather(*cancel_coroutines, return_exceptions=True) 
            print(f"[JimengAI] Info: Sent cancellation requests for pending tasks.")
            raise e 
        
        finally:
            print() 
            if node_id and ps_instance:
                ps_instance.send_sync("progress", {"value": 0, "max": current_max, "node": node_id})

        # 5. 轮询结束 - 处理结果
        for tid, error_msg in failed_tasks_info:
            self._log_batch_task_failure(error_msg, tid)

        print(f"[JimengAI] Batch finished. {len(successful_tasks)} succeeded, {len(failed_tasks_info)} failed.")
        
        # 异常处理 (阻塞)
        if not successful_tasks:
            if failed_tasks_info:
                first_task_id, first_error_msg = failed_tasks_info[0]
                error_to_raise = f"Task {first_task_id} failed: {first_error_msg}"
                if generation_count > 1:
                    error_to_raise = f"All {generation_count} tasks failed. First error: {error_to_raise}"
                
                self._create_failure_json(error_to_raise, first_task_id)
            else:
                self._create_failure_json("Batch failed: No tasks succeeded and no specific errors were reported.", node_id)

        # 在此处创建会话，仅用于本次下载
        async with aiohttp.ClientSession() as session:
            return await self._handle_batch_success_async(successful_tasks, batch_save_path, generation_count, save_last_frame_batch, session)

    # 通用生成逻辑
    async def _common_generation_logic(
        self, 
        client, 
        prompt, 
        duration, 
        resolution, 
        aspect_ratio, 
        seed, 
        generation_count, 
        batch_save_path, 
        save_last_frame_batch, 
        non_blocking, 
        node_id,
        
        # 由子类提供的特定参数
        model_name: str,
        content: list,
        forbidden_params: list[str],
        prompt_extra_args: str = ""
    ):
        # 统一处理视频生成的通用逻辑
        try:
            # 1. 检查禁止的文本参数
            _raise_if_text_params(prompt, forbidden_params)

            # 2. API Seed 逻辑
            api_seed = -1
            if seed > 0:
                api_seed = seed
            
            # 3. 计算时长参数
            (duration_or_frames_arg, estimation_duration) = _calculate_duration_and_frames_args(duration)
            
            # 4. 构建提示词字符串
            prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} {duration_or_frames_arg} --seed {api_seed} {prompt_extra_args.strip()}"
            
            # 5. 将文本提示词插入到内容列表的开头
            content.insert(0, {"type": "text", "text": prompt_string})
            
            # 6. 检查中断
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            # 7. 决定保存路径
            effective_save_path = "Jimeng" 
            if generation_count > 1:
                effective_save_path = batch_save_path
            
            # 8. 调用核心执行器
            return await self._execute_batch_generation(
                client=client,
                model_name=model_name,
                content=content,
                estimation_duration=estimation_duration,
                resolution=resolution,
                generation_count=generation_count,
                batch_save_path=effective_save_path,
                save_last_frame_batch=save_last_frame_batch,
                non_blocking=non_blocking,
                node_id=node_id
            )
        
        except (RuntimeError, TimeoutError, ValueError) as e:
            raise e
        except comfy.model_management.InterruptProcessingException as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to prepare task: {e}")


class JimengVideoGeneration(JimengVideoBase):
    # 即梦视频生成节点
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    
    @classmethod
    def INPUT_TYPES(s):
        # 定义节点的输入参数
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
                "generation_count": ("INT", {"default": 1, "min": 1}),
                "batch_save_path": ("STRING", {"default": "Jimeng/Video"}),
                "save_last_frame_batch": ("BOOLEAN", {"default": False}),
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
    OUTPUT_NODE = True 
    CATEGORY = GLOBAL_CATEGORY
    
    async def generate(self, client, model_choice, prompt, duration, resolution, aspect_ratio, camerafixed, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking=False, image=None, last_frame_image=None, node_id=None):
        
        # 1. 准备模型名称
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
        
        # 2. 准备额外参数
        prompt_extra_args = f"--camerafixed {'true' if camerafixed else 'false'}"
        
        # 3. 准备内容列表 (图像)
        content = []
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
            
        # 4. 准备禁止的参数列表
        forbidden_params = ["resolution", "ratio", "dur", "frames", "camerafixed", "seed"]

        # 5. 调用基类方法
        return await self._common_generation_logic(
            client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id,
            model_name=final_model_name,
            content=content,
            forbidden_params=forbidden_params,
            prompt_extra_args=prompt_extra_args
        )


class JimengReferenceImage2Video(JimengVideoBase):
    # 即梦参考图生成视频节点
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    
    @classmethod
    def INPUT_TYPES(s):
        # 定义节点的输入参数
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
            "generation_count": ("INT", {"default": 1, "min": 1}),
            "batch_save_path": ("STRING", {"default": "Jimeng/Video"}),
            "save_last_frame_batch": ("BOOLEAN", {"default": False}),
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
    OUTPUT_NODE = True 
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking=False, ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None, node_id=None):
        
        # 1. 准备模型名称
        model = "doubao-seedance-1-0-lite-i2v-250428"
        
        # 2. 准备内容列表 (图像)
        content = []
        ref_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        
        for img_tensor in ref_images:
            if img_tensor is not None: content.append({ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img_tensor)}"}, "role": "reference_image" })
        if len(content) == 1: raise ValueError("At least one reference image must be provided.")
        
        # 3. 准备禁止的参数列表
        forbidden_params = ["resolution", "ratio", "dur", "frames", "seed"]

        # 4. 调用基类方法
        return await self._common_generation_logic(
            client, prompt, duration, resolution, aspect_ratio, seed, generation_count, batch_save_path, save_last_frame_batch, non_blocking, node_id,
            model_name=model,
            content=content,
            forbidden_params=forbidden_params,
            prompt_extra_args="" # 此节点没有额外的提示词参数
        )


class JimengQueryTasks:
    # 即梦任务查询节点
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
        # 定义节点的输入参数
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
        # 节点的主要异步执行函数，用于查询任务列表
        ark_client = client.ark
        
        kwargs = {
            "page_num": page_num,
            "page_size": page_size
        }

        # 处理模型选择
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
        
        # 处理任务ID列表
        if task_ids and task_ids.strip():
            id_list = [tid.strip() for tid in task_ids.split('\n') if tid.strip()]
            if id_list:
                kwargs["task_ids"] = id_list
        
        try:
            print(f"[JimengAI] Querying tasks with params: {kwargs}")
            # 异步线程中执行查询
            resp = await asyncio.to_thread(
                ark_client.content_generation.tasks.list,
                **kwargs
            )
            
            items_list = [item.model_dump() for item in resp.items]
            
            # 转换时间戳为可读格式
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
            # 处理查询异常
            error_msg_str = f"Failed to query tasks: {e}"
            print(f"[JimengAI] Error: {error_msg_str}")
            error_msg = json.dumps({"error": error_msg_str}, indent=2, ensure_ascii=False)
            return (error_msg, 0)


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "JimengVideoGeneration": JimengVideoGeneration,
    "JimengReferenceImage2Video": JimengReferenceImage2Video,
    "JimengQueryTasks": JimengQueryTasks,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengVideoGeneration": "Jimeng Video Generation",
    "JimengReferenceImage2Video": "Jimeng Reference to Video",
    "JimengQueryTasks": "Jimeng Query Tasks",
}