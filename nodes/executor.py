import asyncio
import time
import math
import datetime
import logging
import json
import aiohttp
import torch
import random
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.responses.response_completed_event import ResponseCompletedEvent
from volcenginesdkarkruntime.types.responses.response_reasoning_summary_text_delta_event import ResponseReasoningSummaryTextDeltaEvent
from volcenginesdkarkruntime.types.responses.response_output_item_added_event import ResponseOutputItemAddedEvent
from volcenginesdkarkruntime.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from volcenginesdkarkruntime.types.responses.response_text_done_event import ResponseTextDoneEvent
try:
    from volcenginesdkarkruntime.types.responses.response_reasoning_text_delta_event import ResponseReasoningTextDeltaEvent
except ImportError:
    ResponseReasoningTextDeltaEvent = None

import comfy.model_management
from server import PromptServer
from .nodes_shared import log_msg, format_api_error, get_text, JimengException, create_white_image_tensor, create_white_video_file, safe_cat_tensors
from .constants import JIMENG_API_BASE_URL, SEEDANCE_REQUEST_MAX_BYTES
from .models_config import VIDEO_MODEL_MAP, VIDEO_2_UI_OPTIONS
from .utils_download import b64_image_to_tensor_async

DEFAULT_FALLBACK_PER_SEC = 12
DEFAULT_FALLBACK_BASE = 20
HISTORY_PAGE_SIZE = 50
MIN_DATA_POINTS = 3
OUTLIER_STD_DEV_FACTOR = 2.0
RECENT_TASK_COUNT = 5
RECENT_SPIKE_FACTOR = 1.1


def _json_request_default(value):
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compact_json_size_bytes(payload) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_request_default,
        ).encode("utf-8")
    )


def validate_seedance_request_size(payload, max_bytes=SEEDANCE_REQUEST_MAX_BYTES):
    request_bytes = compact_json_size_bytes(payload)
    if request_bytes > max_bytes:
        raise JimengException(
            get_text("err_request_body_too_large").format(
                max_bytes=max_bytes, current_bytes=request_bytes
            )
        )
    return request_bytes


def _update_node_progress(node_id, value, max_value, ps_instance=None):
    if not node_id:
        return
    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().update_progress(str(node_id), float(value), float(max_value))
    except Exception:
        pass
    if ps_instance:
        ps_instance.send_sync(
            "progress",
            {"value": int(value), "max": int(max_value), "node": node_id},
        )


def _finish_node_progress(node_id, ps_instance=None):
    if not node_id:
        return
    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().finish_progress(str(node_id))
    except Exception:
        pass
    if ps_instance:
        ps_instance.send_sync("progress", {"value": 0, "max": 100, "node": node_id})

def _contains_reference_video(content) -> bool:
    if content is None:
        return False
    if isinstance(content, list):
        return any(_contains_reference_video(item) for item in content)
    if isinstance(content, dict):
        item_type = str(content.get("type", "")).strip().lower()
        role = str(content.get("role", "")).strip().lower()
        if item_type in {"video_url", "reference_video"}:
            return True
        if role == "reference_video":
            return True
        if "video_url" in content:
            return True
        return any(_contains_reference_video(v) for v in content.values())
    return False


async def _get_api_estimated_time_async(
    ark_client, model_name: str, duration: int, resolution: str, content=None
) -> (int, str):
    """
    异步获取 API 预估耗时。
    通过分析历史任务数据，使用均值、线性回归或近期负载调整来估算任务完成时间。
    """
    fallback_per_sec = DEFAULT_FALLBACK_PER_SEC
    model_name_lc = str(model_name).lower()

    v2_model_ids = set(VIDEO_2_UI_OPTIONS)
    for m in VIDEO_2_UI_OPTIONS:
        if m in VIDEO_MODEL_MAP:
            v2_model_ids.add(VIDEO_MODEL_MAP[m])

    is_seedance2 = model_name in v2_model_ids or "doubao-seedance-2-0" in model_name_lc
    request_has_ref_video = _contains_reference_video(content) if is_seedance2 else False
    seedance2_expected_with_video_per_sec = None
    seedance2_expected_without_video_per_sec = None

    if is_seedance2:
        seedance2_with_video_per_sec = {
            "4k": 90,
            "1080p": 60,
            "720p": 40,
            "480p": 20,
        }
        fallback_per_sec = seedance2_with_video_per_sec.get(str(resolution).lower(), 40)
        if "fast" in model_name_lc:
            fallback_per_sec = fallback_per_sec / 2
        seedance2_expected_with_video_per_sec = float(fallback_per_sec)
        seedance2_expected_without_video_per_sec = float(fallback_per_sec / 2)
        if not request_has_ref_video:
            fallback_per_sec = fallback_per_sec / 2
    elif resolution == "720p":
        fallback_per_sec = 6
    elif resolution == "480p":
        fallback_per_sec = 3

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

            # API 历史任务不区分是否带参考视频，2.0 系列按耗时区间推断后再筛样本。
            if is_seedance2:
                observed_per_sec = float(task_time) / float(item_duration)
                delta_with_video = abs(observed_per_sec - seedance2_expected_with_video_per_sec)
                delta_without_video = abs(observed_per_sec - seedance2_expected_without_video_per_sec)
                inferred_has_ref_video = delta_with_video <= delta_without_video
                if inferred_has_ref_video != request_has_ref_video:
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

    def _log_batch_task_failure(self, error_message, task_id=None, raw_api_response=None):
        log_msg(
            "err_task_fail_msg",
            tid=task_id or "N/A",
            msg=error_message,
            raw_api_response=raw_api_response,
        )

    def _should_skip_failure_log_before_raise(self, successful_tasks, failed_tasks_info):
        """
        单个最终失败会继续抛给前端弹窗，避免同一条信息再打印一次控制台日志。
        """
        return (
            (not self.ignore_errors)
            and (not successful_tasks)
            and len(failed_tasks_info) == 1
        )

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
        
        if self.ignore_errors:
            print(f"[JimengAI] Pending (Ignored for multi-node): {msg}")
            return

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
                        raw_error = str(res)
                        failed_tasks_info.append((task_ids[i], raw_error, raw_error))
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
                            (res.id, format_api_error(fail_reason), fail_reason)
                        )
                    else:
                        pending_tasks.append(res)

                if pending_tasks:
                    for tid, error_msg, raw_api_response in failed_tasks_info:
                        self._log_batch_task_failure(error_msg, tid, raw_api_response)
                    pending_ids = [task.id for task in pending_tasks]
                    if node_id and ps_instance:
                        ps_instance.send_progress_text(
                            get_text("progress_non_blocking_pending").format(
                                task_ids=", ".join(pending_ids)
                            ),
                            node_id,
                        )
                    return {
                        "non_blocking": True,
                        "status": pending_tasks[0].status,
                        "task_ids": pending_ids,
                        "pending_count": len(pending_ids),
                    }
                else:
                    del non_blocking_cache_dict[node_id]
                    if not self._should_skip_failure_log_before_raise(
                        successful_tasks, failed_tasks_info
                    ):
                        for tid, error_msg, raw_api_response in failed_tasks_info:
                            self._log_batch_task_failure(
                                error_msg, tid, raw_api_response
                            )
                    if not successful_tasks:
                        if failed_tasks_info:
                            first_tid, first_msg, _ = failed_tasks_info[0]
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

        if "seedance-2-0" in str(model_name):
            request_kwargs.pop("service_tier", None)
            request_kwargs.pop("execution_expires_after", None)

        is_multi_content = isinstance(content, list) and len(content) > 0 and isinstance(content[0], list)

        def _compact_debug_payload(value):
            if isinstance(value, dict):
                return {k: _compact_debug_payload(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_compact_debug_payload(v) for v in value]
            if isinstance(value, str) and value.startswith("data:") and len(value) > 200:
                return value[:120] + f"...<truncated:{len(value)}>"
            return value

        create_coroutines = []
        submitted_task_kwargs = []
        for i in range(generation_count):
            task_kwargs = request_kwargs.copy()
            if is_multi_content:
                task_kwargs["content"] = content[i % len(content)]
            if "seedance-2-0" in str(model_name).lower():
                validate_seedance_request_size(task_kwargs)
            submitted_task_kwargs.append(task_kwargs)
            
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

        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                creation_errors.append(res)
                err_text = format_api_error(res)
                if err_text.startswith("[JimengAI] "):
                    err_text = err_text[11:]

                raw_err_text = str(res)
                is_unlocalized_error = err_text.startswith("Error:")
                has_param_error_in_raw = (
                    ("InvalidParameter" in raw_err_text)
                    or ("MissingParameter" in raw_err_text)
                )
                should_print_param_debug = is_unlocalized_error and has_param_error_in_raw

                if should_print_param_debug:
                    debug_req = _compact_debug_payload(submitted_task_kwargs[idx])
                    log_msg(
                        "debug_create_request_params",
                        params=json.dumps(debug_req, ensure_ascii=False, default=str),
                    )
                    log_msg("log_raw_api_response", raw=raw_err_text)

                if err_text not in creation_error_counts:
                    creation_error_counts[err_text] = {"count": 0, "raw": str(res)}
                creation_error_counts[err_text]["count"] += 1
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
                for err_msg, info in creation_error_counts.items():
                    log_msg(
                        "batch_failed_reason",
                        msg=err_msg,
                        count=info["count"],
                        raw_api_response=info["raw"],
                    )

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

        est_resolution = resolution
        if extra_api_params and extra_api_params.get("draft"):
            est_resolution = "480p"

        estimated_single_task_time, method_key = await _get_api_estimated_time_async(
            ark_client, model_name, estimation_duration, est_resolution, content=content
        )
        if estimated_single_task_time <= 0:
            estimated_single_task_time = 1

        if non_blocking:
            task_ids = [t.id for t in tasks_to_poll]
            non_blocking_cache_dict[node_id] = {"task_ids": task_ids}
            if node_id and ps_instance:
                ps_instance.send_progress_text(
                    get_text("progress_non_blocking_submitted").format(
                        task_ids=", ".join(task_ids)
                    ),
                    node_id,
                )
            return {
                "non_blocking": True,
                "status": "submitted",
                "task_ids": task_ids,
                "pending_count": len(task_ids),
            }

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
                                (
                                    current_task_id,
                                    format_api_error(fail_reason),
                                    fail_reason,
                                )
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

                    _update_node_progress(
                        node_id,
                        int(accumulated_running_time),
                        current_max,
                        ps_instance,
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
            _finish_node_progress(node_id, ps_instance)

        if not self._should_skip_failure_log_before_raise(
            successful_tasks, failed_tasks_info
        ):
            for tid, error_msg, raw_api_response in failed_tasks_info:
                self._log_batch_task_failure(error_msg, tid, raw_api_response)

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
                first_tid, first_msg, _ = failed_tasks_info[0]
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
                msg = format_api_error(first_exception) if first_exception else get_text("err_batch_fail_all")
                self._create_failure_json(msg)
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
                        b64_json = getattr(event, "b64_json", None)
                        if event.error is None and b64_json:
                            data = {
                                "type": "b64_image",
                                "b64_json": b64_json,
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

        decode_tasks = []
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
                elif item["type"] == "b64_image":
                    idx_in_group = item["index"]
                    b64_json = item.get("b64_json")
                    if b64_json:
                        if enable_group_generation and generation_count == 1:
                            log_msg("stream_recv_image", index=idx_in_group)

                        async def _decode_b64_wrapper(d_idx, d_b64, d_size):
                            tensor = await b64_image_to_tensor_async(d_b64)
                            return (d_idx, tensor, d_size)

                        decode_tasks.append(
                            asyncio.create_task(
                                _decode_b64_wrapper(
                                    idx_in_group, b64_json, item.get("size")
                                )
                            )
                        )
                elif item["type"] == "completed":
                    final_metadata["usage"] = item["usage"]
                    if "model" in item:
                        final_metadata["model"] = item["model"]
                    if "created" in item:
                        final_metadata["created"] = item["created"]

            if not decode_tasks:
                raise JimengException(get_text("err_batch_fail_all"))

            results = await asyncio.gather(*decode_tasks)
            valid_results = [r for r in results if r[1] is not None]

            if not valid_results:
                raise JimengException(get_text("err_download_img"))

            valid_results.sort(key=lambda x: x[0])

            output_tensors = []
            for v_idx, tensor, size in valid_results:
                output_tensors.append(tensor)
                final_metadata["images"].append(
                    {"index": v_idx, "size": size, "source": "b64_json"}
                )

            return safe_cat_tensors(output_tensors), final_metadata

        except Exception as e:
            if isinstance(e, comfy.model_management.InterruptProcessingException):
                raise e
            raise JimengException(str(e))


class JimengVisualExecutor:
    """
    视觉理解任务执行器.
    """
    def __init__(self, client):
        self.client = client
        self.ark_client = client.ark
        self.api_key = client.api_key
        self._results_cache = {}

    async def create_response_task(self, payload):
        """
        调用 SDK create 方法 (非流式) 以创建任务。
        """
        payload["stream"] = False
        
        log_msg("visual_task_created")
        task_id = "N/A"
        
        try:
            def _call():
                return self.ark_client.responses.create(**payload)
            
            response = await asyncio.to_thread(_call)
            
            task_id = getattr(response, "id", "N/A")
            self._results_cache[task_id] = response
            
            log_msg("visual_task_complete", id=task_id)
            return task_id
            
        except Exception as e:
            if isinstance(e, comfy.model_management.InterruptProcessingException):
                raise e
            formatted_error = format_api_error(e)
            log_msg(
                "visual_task_failed",
                id=task_id,
                msg=formatted_error,
                raw_api_response=str(e),
            )
            raise JimengException(formatted_error)

    async def poll_response_result(self, task_id):
        """
        获取任务结果。
        """
        try:
            if task_id in self._results_cache:
                response = self._results_cache.pop(task_id)
                if hasattr(response, "model_dump"):
                    return response.model_dump()
                else:
                    return response.dict()
            
            raise JimengException(f"Task result not found for ID: {task_id}")
        except Exception as e:
            if isinstance(e, comfy.model_management.InterruptProcessingException):
                raise e
            formatted_error = format_api_error(e)
            log_msg(
                "visual_task_failed",
                id=task_id,
                msg=formatted_error,
                raw_api_response=str(e),
            )
            raise JimengException(formatted_error)

    async def stream_response_task(self, payload, is_single_node=True):
        """
        调用 SDK create 方法 (流式) 并处理事件。
        """
        payload["stream"] = True
        
        log_msg("visual_stream_start")
        
        full_content = ""
        final_json = {}
        
        # 简单的任务标识符，用于控制台区分
        task_short_id = "Main"
        if "previous_response_id" in payload:
             task_short_id = str(payload["previous_response_id"])[-4:]
        else:
             task_short_id = f"{random.randint(1000, 9999)}"

        line_buffer = ""

        def _print_chunk(text):
            if is_single_node:
                print(text, end="", flush=True)
                return

            nonlocal line_buffer
            line_buffer += text
            if "\n" in line_buffer:
                lines = line_buffer.split("\n")
                # 打印除最后一部分外的所有行
                for line in lines[:-1]:
                    print(f"\033[94m[Visual-{task_short_id}]\033[0m {line}")
                line_buffer = lines[-1]

        def _run_stream():
            nonlocal full_content, final_json
            try:
                stream = self.ark_client.responses.create(**payload)
                
                for event in stream:
                    if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                        delta = event.delta
                        if delta:
                            _print_chunk(delta)
                    
                    if ResponseReasoningTextDeltaEvent and isinstance(event, ResponseReasoningTextDeltaEvent):
                         delta = event.delta
                         if delta:
                             _print_chunk(delta)

                    if isinstance(event, ResponseOutputItemAddedEvent):
                         pass

                    if isinstance(event, ResponseTextDeltaEvent):
                        delta = event.delta
                        if delta:
                            _print_chunk(delta)
                            full_content += delta
                            
                    if isinstance(event, ResponseTextDoneEvent):
                         if not is_single_node and line_buffer:
                            print(f"\033[94m[Visual-{task_short_id}]\033[0m {line_buffer}")
                         elif is_single_node:
                            print("")
                        
                    if isinstance(event, ResponseCompletedEvent):
                        # print(f"\nResponse Completed. Usage: {event.response.usage}")
                        if hasattr(event.response, "model_dump"):
                            final_json = event.response.model_dump()
                        else:
                             final_json = event.response.dict()
                             
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                formatted_error = format_api_error(e)
                log_msg(
                    "visual_task_failed",
                    id=task_short_id,
                    msg=formatted_error,
                    raw_api_response=str(e),
                )
                raise JimengException(formatted_error)
                
        await asyncio.to_thread(_run_stream)
        
        if not final_json:
             final_json = {"status": "completed (stream)", "output": full_content}
        
        return full_content, json.dumps(final_json, indent=2, ensure_ascii=False)
