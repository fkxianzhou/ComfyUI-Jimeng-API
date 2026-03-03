import asyncio
import time
import math
import datetime
import logging
import json
import aiohttp
import torch
import random

import comfy.model_management
from server import PromptServer
from .nodes_shared import log_msg, format_api_error, get_text, JimengException, create_white_image_tensor, create_white_video_file, safe_cat_tensors
from .models_config import VIDEO_MODEL_MAP, VIDEO_2_UI_OPTIONS
from .utils_download import download_url_to_image_tensor_async

DEFAULT_FALLBACK_PER_SEC = 12
DEFAULT_FALLBACK_BASE = 20
HISTORY_PAGE_SIZE = 50
MIN_DATA_POINTS = 3
OUTLIER_STD_DEV_FACTOR = 2.0
RECENT_TASK_COUNT = 5
RECENT_SPIKE_FACTOR = 1.1

async def _get_api_estimated_time_async(
    ark_client, model_name: str, duration: int, resolution: str
) -> (int, str):
    """
    异步获取 API 预估耗时。
    通过分析历史任务数据，使用均值、线性回归或近期负载调整来估算任务完成时间。
    """
    fallback_per_sec = DEFAULT_FALLBACK_PER_SEC

    v2_model_ids = set(VIDEO_2_UI_OPTIONS)
    for m in VIDEO_2_UI_OPTIONS:
        if m in VIDEO_MODEL_MAP:
            v2_model_ids.add(VIDEO_MODEL_MAP[m])

    if model_name in v2_model_ids:
        fallback_per_sec = 60
    elif resolution == "720p":
        fallback_per_sec = 6
    elif resolution == "480p":
        fallback_per_sec = 2

    fallback_time = (int(duration) * fallback_per_sec) + DEFAULT_FALLBACK_BASE
    try:
        resp = await asyncio.to_thread(
            ark_client.content_generation.tasks.list,
            status="succeeded",
            model=model_name,
            page_size=HISTORY_PAGE_SIZE,
        )
        if not resp.items:
            return (fallback_time, "est_fallback")

        exact_timings = []
        recent_exact_timings = []
        all_data_points = []

        for item in resp.items:
            if not (
                item.status == "succeeded"
                and hasattr(item, "resolution")
                and item.resolution == resolution
            ):
                continue

            item_duration = getattr(item, "duration", 0)
            t_start = item.created_at
            t_end = item.updated_at
            if hasattr(t_start, "timestamp"):
                t_start = t_start.timestamp()
            if hasattr(t_end, "timestamp"):
                t_end = t_end.timestamp()

            raw_diff = float(t_end) - float(t_start)
            try:
                local_offset = (
                    datetime.datetime.now().astimezone().utcoffset().total_seconds()
                )
            except Exception:
                local_offset = 0

            fixed_diff = raw_diff - local_offset
            task_time = (
                fixed_diff
                if fixed_diff > 0 and abs(fixed_diff) < abs(raw_diff)
                else raw_diff
            )

            if task_time <= 0 or item_duration <= 0:
                continue

            all_data_points.append((float(item_duration), float(task_time)))
            if item_duration == int(duration):
                exact_timings.append(task_time)
                if len(recent_exact_timings) < RECENT_TASK_COUNT:
                    recent_exact_timings.append(task_time)

        if len(exact_timings) >= MIN_DATA_POINTS:
            mean = sum(exact_timings) / len(exact_timings)
            variance = sum([(x - mean) ** 2 for x in exact_timings]) / len(
                exact_timings
            )
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
        std_dev_t = math.sqrt(
            sum([(t - mean_t) ** 2 for t in all_times]) / len(all_times)
        )
        threshold_t = std_dev_t * OUTLIER_STD_DEV_FACTOR
        filtered_data_points = [
            (d, t) for d, t in all_data_points if abs(t - mean_t) < threshold_t
        ]

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

class JimengGenerationExecutor:
    """
    统一的 Jimeng 生成任务执行器。
    支持：
    1. 异步任务模式 (Async Task): 适用于视频生成 (提交 -> 轮询 -> 结果)
    2. 并行直接模式 (Parallel Direct): 适用于图像生成 (并发请求 -> 结果)
    """
    def __init__(self, client, node_id=None, ignore_errors=False):
        self.client = client
        self.ark_client = client.ark
        self.node_id = node_id
        self.ignore_errors = ignore_errors
        self.ps_instance = PromptServer.instance

    def _log_batch_task_failure(self, error_message, task_id=None):
        log_msg("err_task_fail_msg", tid=task_id or "N/A", msg=error_message)

    def _create_failure_json(self, error_message, task_id=None):
        """
        创建并抛出失败异常信息。
        """
        clean_msg = error_message
        prefix = "[JimengAI]"
        if clean_msg.strip().startswith(prefix):
            clean_msg = clean_msg.strip()[len(prefix) :].strip()
        if clean_msg.startswith("Error:"):
            clean_msg = clean_msg[6:].strip()
        
        if task_id:
            display_msg = get_text("popup_task_failed").format(
                task_id=task_id, msg=clean_msg
            )
        else:
            display_msg = get_text("popup_req_failed").format(msg=clean_msg)
        
        if self.ignore_errors:
            clean_display_msg = display_msg
            if clean_display_msg.startswith("[JimengAI] "):
                clean_display_msg = clean_display_msg[11:].strip()
            
            if clean_display_msg.startswith("[JimengAI] "):
                clean_display_msg = clean_display_msg[11:].strip()

            log_msg("err_task_fail_ignored", node_id=self.node_id or "N/A", msg=clean_display_msg)
            return
            
        raise JimengException(display_msg)

    def _create_pending_json(self, status, task_id=None, task_count=0):
        """
        创建并抛出等待中状态异常，用于非阻塞模式下的 UI 提示。
        """
        if task_count > 0:
            msg = get_text("popup_batch_pending").format(count=task_count)
        else:
            msg = get_text("popup_task_pending").format(task_id=task_id, status=status)
        raise JimengException(msg)

    async def run_batch_tasks(
        self,
        model_name,
        content,
        estimation_duration,
        resolution,
        generation_count,
        non_blocking,
        non_blocking_cache_dict,
        poll_interval=2,
        service_tier="default",
        execution_expires_after=None,
        extra_api_params=None,
        return_last_frame=True,
        on_tasks_created=None,
    ):
        """
        执行批量生成任务 (Video 模式)。
        包含任务创建、状态轮询、进度估算和异常处理。
        """
        ark_client = self.ark_client
        ps_instance = self.ps_instance
        node_id = self.node_id
        
        cached_data = non_blocking_cache_dict.get(node_id)

        if non_blocking and cached_data:
            task_ids = cached_data["task_ids"]
            log_msg("check_status", count=len(task_ids))
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                get_coroutines = [
                    asyncio.to_thread(
                        ark_client.content_generation.tasks.get, task_id=tid
                    )
                    for tid in task_ids
                ]
                results = await asyncio.gather(*get_coroutines, return_exceptions=True)

                successful_tasks, failed_tasks_info, pending_tasks = [], [], []
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        failed_tasks_info.append((task_ids[i], str(res)))
                    elif res.status == "succeeded":
                        successful_tasks.append(res)
                    elif res.status in ["failed", "cancelled", "expired"]:
                        fail_reason = "Failed"
                        if hasattr(res, "error") and res.error:
                            if hasattr(res.error, "message"):
                                fail_reason = res.error.message
                            elif isinstance(res.error, dict) and "message" in res.error:
                                fail_reason = res.error["message"]
                            else:
                                fail_reason = str(res.error)
                        elif res.status == "cancelled":
                            fail_reason = "Cancelled"
                        elif res.status == "expired":
                            fail_reason = "Expired"
                        failed_tasks_info.append(
                            (res.id, format_api_error(fail_reason))
                        )
                    else:
                        pending_tasks.append(res)

                for tid, error_msg in failed_tasks_info:
                    self._log_batch_task_failure(error_msg, tid)

                if pending_tasks:
                    self._create_pending_json(
                        pending_tasks[0].status, pending_tasks[0].id, len(pending_tasks)
                    )
                else:
                    del non_blocking_cache_dict[node_id]
                    if not successful_tasks:
                        if failed_tasks_info:
                            first_tid, first_msg = failed_tasks_info[0]
                            self._create_failure_json(first_msg, task_id=first_tid)
                        else:
                            self._create_failure_json(
                                get_text("err_batch_fail_all")
                            )
                    
                    return successful_tasks

            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                if str(e).startswith("[JimengAI]"):
                    raise e
                del non_blocking_cache_dict[node_id]
                log_msg("err_check_status_batch", e=e)
                self._create_failure_json(format_api_error(e))

        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_name)

        request_kwargs = {
            "model": model_name,
            "content": content,
            "return_last_frame": return_last_frame,
        }
        if service_tier:
            request_kwargs["service_tier"] = service_tier
        if execution_expires_after is not None:
            request_kwargs["execution_expires_after"] = execution_expires_after
        
        if extra_api_params:
            request_kwargs.update(extra_api_params)

        is_multi_content = isinstance(content, list) and len(content) > 0 and isinstance(content[0], list)

        create_coroutines = []
        for i in range(generation_count):
            task_kwargs = request_kwargs.copy()
            if is_multi_content:
                task_kwargs["content"] = content[i % len(content)]
            
            create_coroutines.append(
                asyncio.to_thread(
                    ark_client.content_generation.tasks.create, **task_kwargs
                )
            )

        comfy.model_management.throw_exception_if_processing_interrupted()
        results = await asyncio.gather(*create_coroutines, return_exceptions=True)

        tasks_to_poll = []
        creation_errors = []
        creation_error_counts = {}

        for res in results:
            if isinstance(res, Exception):
                creation_errors.append(res)
                err_text = format_api_error(res)
                if err_text.startswith("[JimengAI] "):
                    err_text = err_text[11:]

                creation_error_counts[err_text] = (
                    creation_error_counts.get(err_text, 0) + 1
                )
            else:
                tasks_to_poll.append(res)

        failed_count = len(creation_errors)
        created_count = len(tasks_to_poll)

        if generation_count > 1:
            log_msg(
                "batch_submit_result",
                created=created_count,
                failed=failed_count,
            )
            if failed_count > 0 and created_count > 0:
                log_msg("batch_failed_summary", count=failed_count)
                for err_msg, count in creation_error_counts.items():
                    log_msg("batch_failed_reason", msg=err_msg, count=count)

        if not tasks_to_poll:
            final_error_msg = get_text("err_batch_fail_all")
            if creation_errors:
                final_error_msg = format_api_error(creation_errors[0])
            self._create_failure_json(final_error_msg)
            return []

        if on_tasks_created:
            try:
                on_tasks_created(tasks_to_poll)
            except Exception as e:
                log_msg("err_on_tasks_created", e=e)

        estimated_single_task_time, method_key = await _get_api_estimated_time_async(
            ark_client, model_name, estimation_duration, resolution
        )
        if estimated_single_task_time <= 0:
            estimated_single_task_time = 1

        if non_blocking:
            task_ids = [t.id for t in tasks_to_poll]
            non_blocking_cache_dict[node_id] = {"task_ids": task_ids}
            if node_id and ps_instance:
                ps_instance.send_sync(
                    "jimeng_fake_progress",
                    {
                        "value": 0,
                        "max": estimated_single_task_time,
                        "node": node_id,
                    },
                )
            self._create_pending_json("submitted", task_ids[0], len(task_ids))

        method_name = get_text(method_key)
        if tasks_to_poll:
            log_msg(
                "task_submitted_est", time=estimated_single_task_time, method=method_name
            )

        if generation_count == 1 and tasks_to_poll:
            log_msg("task_info_simple", task_id=tasks_to_poll[0].id, model=model_name)

        accumulated_running_time = 0.0
        max_concurrency_seen = 0
        running_task_start_times = {}
        last_loop_time = time.time()
        extra_tail_seconds = poll_interval
        extended_est_total = estimated_single_task_time + extra_tail_seconds
        extended_est_applied = False

        successful_tasks = []
        failed_tasks_info = []
        tasks_to_poll_ids = [t.id for t in tasks_to_poll]
        total_tasks_count = len(tasks_to_poll_ids)

        try:
            while tasks_to_poll_ids:
                now = time.time()
                loop_delta = now - last_loop_time
                last_loop_time = now

                comfy.model_management.throw_exception_if_processing_interrupted()

                get_coroutines = [
                    asyncio.to_thread(
                        ark_client.content_generation.tasks.get, task_id=tid
                    )
                    for tid in tasks_to_poll_ids
                ]
                results = await asyncio.gather(*get_coroutines, return_exceptions=True)

                next_poll_ids = []
                current_running_ids = []
                current_queued_count = 0
                single_task_status_for_display = "queued"

                for i, res in enumerate(results):
                    current_task_id = tasks_to_poll_ids[i]
                    if isinstance(res, Exception):
                        next_poll_ids.append(current_task_id)
                        current_queued_count += 1
                        single_task_status_for_display = "unknown"
                    else:
                        if res.status == "succeeded":
                            successful_tasks.append(res)
                            if current_task_id in running_task_start_times:
                                del running_task_start_times[current_task_id]
                        elif res.status in ["failed", "cancelled", "expired"]:
                            if current_task_id in running_task_start_times:
                                del running_task_start_times[current_task_id]

                            fail_reason = "Failed"
                            if hasattr(res, "error") and res.error:
                                if hasattr(res.error, "message"):
                                    fail_reason = res.error.message
                                elif (
                                    isinstance(res.error, dict)
                                    and "message" in res.error
                                ):
                                    fail_reason = res.error["message"]
                                else:
                                    fail_reason = str(res.error)
                            failed_tasks_info.append(
                                (current_task_id, format_api_error(fail_reason))
                            )
                        else:
                            next_poll_ids.append(current_task_id)
                            if res.status == "running":
                                current_running_ids.append(current_task_id)
                                if current_task_id not in running_task_start_times:
                                    running_task_start_times[current_task_id] = now
                            else:
                                current_queued_count += 1
                                single_task_status_for_display = res.status

                tasks_to_poll_ids = next_poll_ids

                if not tasks_to_poll_ids:
                    break

                running_count = len(current_running_ids)

                if running_count > max_concurrency_seen:
                    max_concurrency_seen = running_count

                if running_count > 0:
                    accumulated_running_time += loop_delta

                    if generation_count == 1:
                        if not extended_est_applied:
                            current_max = int(estimated_single_task_time)
                            if accumulated_running_time >= 0.95 * estimated_single_task_time:
                                extended_est_applied = True
                        if extended_est_applied:
                            if accumulated_running_time > extended_est_total:
                                extended_est_total = accumulated_running_time + extra_tail_seconds
                            current_max = int(extended_est_total)
                        else:
                            if accumulated_running_time > current_max:
                                current_max = int(accumulated_running_time)
                    else:
                        running_remainings = []
                        for tid in current_running_ids:
                            start_ts = running_task_start_times.get(tid, now)
                            elapsed_for_task = now - start_ts
                            rem = max(
                                1.0, estimated_single_task_time - elapsed_for_task
                            )
                            running_remainings.append(rem)

                        max_running_rem = (
                            max(running_remainings)
                            if running_remainings
                            else estimated_single_task_time
                        )
                        effective_concurrency = max(max_concurrency_seen, 1)
                        queue_est_time = (
                            current_queued_count * estimated_single_task_time
                        ) / effective_concurrency

                        future_est = max_running_rem + queue_est_time
                        current_max = int(accumulated_running_time + future_est)

                    if node_id and ps_instance:
                        ps_instance.send_sync(
                            "progress",
                            {
                                "value": int(accumulated_running_time),
                                "max": current_max,
                                "node": node_id,
                            },
                        )

                    if generation_count == 1:
                        print(
                            get_text("polling_single").format(
                                task_id=tasks_to_poll_ids[0],
                                elapsed=int(accumulated_running_time),
                                max=current_max,
                            ),
                            end="\r",
                        )
                    else:
                        done_count = len(successful_tasks) + len(failed_tasks_info)
                        print(
                            get_text("polling_batch_stats").format(
                                done=done_count,
                                total=total_tasks_count,
                                pending=len(tasks_to_poll_ids),
                                elapsed=int(accumulated_running_time),
                                max=current_max,
                                running=running_count,
                                queued=current_queued_count,
                            ),
                            end="\r",
                        )
                else:
                    if generation_count == 1:
                        print(
                            get_text("polling_single_waiting").format(
                                task_id=tasks_to_poll_ids[0],
                                status=single_task_status_for_display,
                                queued=current_queued_count,
                            ),
                            end="\r",
                        )
                    else:
                        done_count = len(successful_tasks) + len(failed_tasks_info)
                        print(
                            get_text("polling_batch_stats").format(
                                done=done_count,
                                total=total_tasks_count,
                                pending=len(tasks_to_poll_ids),
                                elapsed=int(accumulated_running_time),
                                max=int(accumulated_running_time + 10),
                                running=0,
                                queued=current_queued_count,
                            ),
                            end="\r",
                        )

                await asyncio.sleep(poll_interval)

        except comfy.model_management.InterruptProcessingException as e:
            log_msg("interrupted")

            cancel_stats = {"success": 0, "failed_counts": {}}

            async def _cancel_task_safe(tid):
                try:
                    await asyncio.to_thread(
                        ark_client.content_generation.tasks.delete, task_id=tid
                    )
                    return True, None
                except Exception as ex:
                    err_msg = format_api_error(ex)

                    clean_msg = err_msg.replace("[JimengAI] ", "").strip()
                    return False, clean_msg

            cancel_coroutines = [_cancel_task_safe(tid) for tid in tasks_to_poll_ids]
            results = await asyncio.gather(*cancel_coroutines)

            for success, msg in results:
                if success:
                    cancel_stats["success"] += 1
                else:
                    cancel_stats["failed_counts"][msg] = (
                        cancel_stats["failed_counts"].get(msg, 0) + 1
                    )

            if generation_count == 1:
                if cancel_stats["success"] > 0:
                    log_msg("cancel_task_success", task_id=tasks_to_poll_ids[0])
                else:
                    first_fail_msg = next(iter(cancel_stats["failed_counts"]))
                    log_msg(
                        "cancel_task_failed",
                        task_id=tasks_to_poll_ids[0],
                        msg=first_fail_msg,
                    )
            else:
                failed_total = sum(cancel_stats["failed_counts"].values())
                log_msg(
                    "cancel_batch_summary",
                    success=cancel_stats["success"],
                    failed=failed_total,
                )
                if failed_total > 0:
                    for msg, count in cancel_stats["failed_counts"].items():
                        log_msg("cancel_batch_reason", msg=msg, count=count)

            raise e

        finally:
            print()
            if node_id and ps_instance:
                ps_instance.send_sync(
                    "progress", {"value": 0, "max": 100, "node": node_id}
                )

        for tid, error_msg in failed_tasks_info:
            self._log_batch_task_failure(error_msg, tid)

        if generation_count == 1:
            if successful_tasks:
                log_msg("task_finished_single")
        else:
            log_msg(
                "batch_finished_stats",
                success=len(successful_tasks),
                failed=len(failed_tasks_info),
            )

        if not successful_tasks:
            if failed_tasks_info:
                first_tid, first_msg = failed_tasks_info[0]
                self._create_failure_json(first_msg, task_id=first_tid)
            else:
                self._create_failure_json(get_text("err_batch_fail_all"))
        
        return successful_tasks

    async def run_parallel_requests(
        self,
        generation_count,
        request_func,
        **kwargs
    ):
        """
        执行并发请求 (Image 模式)。
        并发调用 request_func，并处理结果。
        
        request_func: async function(index, session) -> (result_tensor, result_metadata)
        """
        async with aiohttp.ClientSession() as session:
            tasks = [request_func(i, session) for i in range(generation_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        error_counts = {}
        first_exception = None

        for res in results:
            if isinstance(res, Exception):
                if isinstance(res, comfy.model_management.InterruptProcessingException):
                    raise res

                msg = str(res)
                prefix = "[JimengAI] "
                if msg.startswith(prefix):
                    msg = msg[len(prefix) :]

                error_counts[msg] = error_counts.get(msg, 0) + 1
                if first_exception is None:
                    first_exception = res
            else:
                valid_results.append(res)

        if generation_count > 1:
            log_msg(
                "batch_finished_stats",
                success=len(valid_results),
                failed=sum(error_counts.values()),
            )
            if error_counts:
                log_msg("batch_failed_summary", count=sum(error_counts.values()))
                for msg, count in error_counts.items():
                    log_msg("batch_failed_reason", msg=msg, count=count)

        if not valid_results:
            if self.ignore_errors:
                 return [], []

            if generation_count > 1:
                log_msg("err_batch_fail_all")
            if first_exception:
                raise first_exception
            raise JimengException(get_text("err_batch_fail_all"))

        valid_results.sort(key=lambda x: x[1].get("batch_index", 0))
        
        tensors = [r[0] for r in valid_results]
        metadata = [r[1] for r in valid_results]
        
        return tensors, metadata

    async def stream_generation_helper(
        self,
        session,
        ark_client,
        kwargs,
        idx,
        enable_group_generation,
        generation_count,
    ):
        """
        处理流式 API 请求和响应 (Image 模式)。
        """
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer_thread():
            try:
                kwargs["stream"] = True
                stream = ark_client.images.generate(**kwargs)

                for event in stream:
                    if event is None:
                        continue

                    if event.type == "image_generation.partial_succeeded":
                        if event.error is None and event.url:
                            data = {
                                "type": "url",
                                "url": event.url,
                                "index": event.image_index + 1,
                                "size": getattr(event, "size", None),
                            }
                            loop.call_soon_threadsafe(queue.put_nowait, data)

                    elif event.type == "image_generation.partial_failed":
                        error_msg = (
                            event.error.message if event.error else "Unknown Error"
                        )
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type": "log",
                                "key": "stream_partial_fail",
                                "kwargs": {
                                    "index": event.image_index + 1,
                                    "msg": error_msg,
                                },
                            },
                        )
                        if (
                            event.error
                            and hasattr(event.error, "code")
                            and event.error.code == "InternalServiceError"
                        ):
                            raise JimengException(
                                f"Critical API Error: {event.error.message}"
                            )

                    elif event.type == "image_generation.completed":
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type": "completed",
                                "usage": event.usage,
                                "model": event.model,
                                "created": event.created,
                            },
                        )

            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "error": e}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        asyncio.create_task(asyncio.to_thread(_producer_thread))

        download_tasks = []
        final_metadata = {
            "batch_index": idx,
            "created": int(time.time()),
            "usage": {},
            "images": [],
        }

        try:
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()

                item = await queue.get()

                if item["type"] == "done":
                    break
                elif item["type"] == "error":
                    raise JimengException(format_api_error(item["error"]))
                elif item["type"] == "log":
                    if enable_group_generation and generation_count == 1:
                        key = item.get("key")
                        log_kwargs = item.get("kwargs", {})
                        log_msg(key, **log_kwargs)
                elif item["type"] == "completed":
                    final_metadata["usage"] = item["usage"]
                    if "model" in item:
                        final_metadata["model"] = item["model"]
                    if "created" in item:
                        final_metadata["created"] = item["created"]
                elif item["type"] == "url":
                    url = item["url"]
                    idx_in_group = item["index"]

                    if enable_group_generation and generation_count == 1:
                        log_msg("stream_recv_image", index=idx_in_group, url=url)

                    async def _download_wrapper(d_url, d_idx):
                        tensor = await download_url_to_image_tensor_async(
                            session, d_url
                        )
                        return (d_idx, tensor, d_url)

                    task = asyncio.create_task(_download_wrapper(url, idx_in_group))
                    download_tasks.append(task)

            if not download_tasks:
                raise JimengException(get_text("err_batch_fail_all"))

            results = await asyncio.gather(*download_tasks)
            valid_results = [r for r in results if r[1] is not None]

            if not valid_results:
                raise JimengException(get_text("err_download_img"))

            valid_results.sort(key=lambda x: x[0])

            output_tensors = []
            for v_idx, tensor, url in valid_results:
                output_tensors.append(tensor)
                final_metadata["images"].append({"url": url, "index": v_idx})

            return safe_cat_tensors(output_tensors), final_metadata

        except Exception as e:
            if isinstance(e, comfy.model_management.InterruptProcessingException):
                raise e
            raise JimengException(str(e))
