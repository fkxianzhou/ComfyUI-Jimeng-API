import os
import time
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
import torch
import PIL.Image
import numpy

from comfy_api.latest import io as comfy_io

from .nodes_shared import (
    GLOBAL_CATEGORY,
    _image_to_base64,
    log_msg,
    get_text,
    format_api_error,
    JimengClientType,
)
from .utils_download import (
    download_video_to_temp,
    download_image_to_temp,
    save_to_output,
)

from .models_config import (
    VIDEO_MODEL_MAP,
    VIDEO_UI_OPTIONS,
    VIDEO_1_5_UI_OPTIONS,
    QUERY_TASKS_MODEL_LIST,
    REF_IMG_2_VIDEO_MODEL_ID,
)

logging.getLogger("volcenginesdkarkruntime").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:

    class VideoFromFile:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Failed to import 'comfy_api'. Video output nodes may not work."
            )


DEFAULT_FALLBACK_PER_SEC = 12
DEFAULT_FALLBACK_BASE = 20
HISTORY_PAGE_SIZE = 50
MIN_DATA_POINTS = 3
OUTLIER_STD_DEV_FACTOR = 2.0
RECENT_TASK_COUNT = 5
RECENT_SPIKE_FACTOR = 1.1

NON_BLOCKING_TASK_CACHE = {}
LAST_SEEDANCE_1_5_DRAFT_TASK_ID = {}



async def _get_api_estimated_time_async(
    ark_client, model_name: str, duration: int, resolution: str
) -> (int, str):
    fallback_time = (int(duration) * DEFAULT_FALLBACK_PER_SEC) + DEFAULT_FALLBACK_BASE
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


def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    for i in text_params:
        if f"--{i}" in prompt:
            raise ValueError(get_text("popup_param_not_allowed").format(param=i))


def _calculate_duration_and_frames_args(duration: float) -> (str, int):
    if duration == int(duration):
        return (f"--dur {int(duration)}", int(duration))
    else:
        target_frames = duration * 24.0
        n = round((target_frames - 25.0) / 4.0)
        final_frames = int(max(29, min(289, 25 + 4 * n)))
        return (f"--frames {final_frames}", int(round(final_frames / 24.0)))


class JimengVideoBase:
    NON_BLOCKING_TASK_CACHE = NON_BLOCKING_TASK_CACHE

    def _log_batch_task_failure(self, error_message, task_id=None):
        log_msg("err_task_fail_msg", tid=task_id or "N/A", msg=error_message)

    def _create_failure_json(self, error_message, task_id=None):
        clean_msg = error_message
        prefix = "[JimengAI]"
        if clean_msg.strip().startswith(prefix):
            clean_msg = clean_msg.strip()[len(prefix) :].strip()
        if clean_msg.startswith("Error:"):
            clean_msg = clean_msg[6:].strip()
        print(f"[JimengAI] {clean_msg}")
        if task_id:
            display_msg = get_text("popup_task_failed").format(
                task_id=task_id, msg=clean_msg
            )
        else:
            display_msg = get_text("popup_req_failed").format(msg=clean_msg)
        raise RuntimeError(display_msg)

    def _create_pending_json(self, status, task_id=None, task_count=0):
        if task_count > 0:
            msg = get_text("popup_batch_pending").format(count=task_count)
        else:
            msg = get_text("popup_task_pending").format(task_id=task_id, status=status)
        raise RuntimeError(msg)

    def _get_service_options(self, enable_offline, timeout_seconds):
        service_tier = "flex" if enable_offline else "default"
        execution_expires_after = timeout_seconds
        return service_tier, execution_expires_after

    def _append_image_content(self, content_list, image, role):
        if image is not None:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_image_to_base64(image)}"
                    },
                    "role": role,
                }
            )

    async def _handle_batch_success_async(
        self,
        successful_tasks,
        filename_prefix,
        generation_count,
        save_last_frame_batch,
        session,
    ):
        if generation_count > 1:
            log_msg("batch_handling", count=len(successful_tasks))

        temp_save_path = "Jimeng"
        video_prefix = "Jimeng_Vid_Temp"
        frame_prefix = "Jimeng_Frame_Temp"

        async def _process_task(task):
            video_url = task.content.video_url
            last_frame_url = getattr(task.content, "last_frame_url", None)
            seed = getattr(task, "seed", random.randint(0, 4294967295))

            v_coro = download_video_to_temp(
                session, video_url, video_prefix, seed, temp_save_path
            )
            
            f_coro = None
            if last_frame_url:
                f_coro = download_image_to_temp(
                    session, last_frame_url, frame_prefix, seed, temp_save_path
                )

            if f_coro:
                v_path, (f_tensor, f_path) = await asyncio.gather(v_coro, f_coro)
            else:
                v_path = await v_coro
                f_tensor, f_path = None, None

            resp = task.model_dump()
            for k in ["created_at", "updated_at"]:
                if k in resp and isinstance(resp[k], (int, float)):
                    resp[k] = datetime.datetime.fromtimestamp(resp[k]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            return {
                "seed": seed,
                "video_path": v_path,
                "frame_tensor": f_tensor,
                "frame_path": f_path,
                "response": resp,
            }

        results = await asyncio.gather(
            *[_process_task(t) for t in successful_tasks], return_exceptions=True
        )
        valid_results = []
        for res in results:
            if isinstance(res, Exception):
                log_msg("err_download_url", url="batch_task", e=res)
                continue
            valid_results.append(res)

        valid_results.sort(key=lambda x: x["seed"])

        all_responses = []
        first_video = None
        first_frame = None

        for res in valid_results:
            if res["frame_tensor"] is None and res["video_path"]:
                try:
                    import cv2
                    cap = cv2.VideoCapture(res["video_path"])
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                image = frame.astype(numpy.float32) / 255.0
                                res["frame_tensor"] = torch.from_numpy(image)[None,]
                    cap.release()
                except Exception as e:
                    print(f"[JimengAI] Warning: Failed to extract last frame locally: {e}")

            all_responses.append(res["response"])
            v_path = res["video_path"]
            f_tensor = res["frame_tensor"]
            f_path = res["frame_path"]

            if first_video is None and v_path:
                first_video = VideoFromFile(v_path)
            if first_frame is None and f_tensor is not None:
                first_frame = f_tensor

            if generation_count > 1:
                save_to_output(v_path, filename_prefix)
                if save_last_frame_batch and f_path:
                    save_to_output(f_path, filename_prefix)

        return comfy_io.NodeOutput(
            first_video, first_frame, json.dumps(all_responses, indent=2)
        )

    async def _execute_batch_generation(
        self,
        client,
        model_name,
        content,
        estimation_duration,
        resolution,
        generation_count,
        filename_prefix,
        save_last_frame_batch,
        non_blocking,
        node_id,
        poll_interval=2,
        service_tier="default",
        execution_expires_after=None,
        extra_api_params=None,
        return_last_frame=True,
        on_tasks_created=None,
    ):
        ark_client = client.ark
        ps_instance = PromptServer.instance
        cached_data = self.NON_BLOCKING_TASK_CACHE.get(node_id)

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
                    del self.NON_BLOCKING_TASK_CACHE[node_id]
                    if not successful_tasks:
                        if failed_tasks_info:
                            first_tid, first_msg = failed_tasks_info[0]
                            self._create_failure_json(first_msg, task_id=first_tid)
                        else:
                            self._create_failure_json(
                                "Batch failed: No tasks succeeded."
                            )

                    ret_results = None
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(force_close=True)
                    ) as session:
                        ret_results = await self._handle_batch_success_async(
                            successful_tasks,
                            filename_prefix,
                            generation_count,
                            save_last_frame_batch,
                            session,
                        )
                        await asyncio.sleep(0.25)
                    return ret_results

            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                if str(e).startswith("[JimengAI]"):
                    raise e
                del self.NON_BLOCKING_TASK_CACHE[node_id]
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
            if failed_count > 0:
                log_msg("batch_failed_summary", count=failed_count)
                for err_msg, count in creation_error_counts.items():
                    log_msg("batch_failed_reason", msg=err_msg, count=count)

        if not tasks_to_poll:
            log_msg("err_batch_fail_all")
            final_error_msg = "All tasks failed on creation."
            if creation_errors:
                final_error_msg = format_api_error(creation_errors[0])
            self._create_failure_json(final_error_msg)

        if on_tasks_created:
            try:
                on_tasks_created(tasks_to_poll)
            except Exception as e:
                print(f"[JimengAI] Warning: on_tasks_created callback failed: {e}")

        if non_blocking:
            task_ids = [t.id for t in tasks_to_poll]
            self.NON_BLOCKING_TASK_CACHE[node_id] = {"task_ids": task_ids}
            self._create_pending_json("submitted", task_ids[0], len(task_ids))

        estimated_single_task_time, method_key = await _get_api_estimated_time_async(
            ark_client, model_name, estimation_duration, resolution
        )
        if estimated_single_task_time <= 0:
            estimated_single_task_time = 1

        method_name = get_text(method_key)
        log_msg(
            "task_submitted_est", time=estimated_single_task_time, method=method_name
        )

        if generation_count == 1 and tasks_to_poll:
            log_msg("task_info_simple", task_id=tasks_to_poll[0].id, model=model_name)

        accumulated_running_time = 0.0
        max_concurrency_seen = 0
        running_task_start_times = {}
        last_loop_time = time.time()

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

                    running_remainings = []
                    for tid in current_running_ids:
                        start_ts = running_task_start_times.get(tid, now)
                        elapsed_for_task = now - start_ts
                        rem = max(1.0, estimated_single_task_time - elapsed_for_task)
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
                self._create_failure_json("Batch failed: No tasks succeeded.")

        ret_results = None
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(force_close=True)
        ) as session:
            ret_results = await self._handle_batch_success_async(
                successful_tasks,
                filename_prefix,
                generation_count,
                save_last_frame_batch,
                session,
            )
            await asyncio.sleep(0.25)

        await asyncio.sleep(0.1)
        return ret_results

    async def _common_generation_logic(
        self,
        client,
        prompt,
        duration,
        resolution,
        aspect_ratio,
        seed,
        generation_count,
        filename_prefix,
        save_last_frame_batch,
        non_blocking,
        node_id,
        model_name,
        content,
        forbidden_params,
        prompt_extra_args="",
        service_tier="default",
        execution_expires_after=None,
        enable_random_seed=False,
        is_auto_duration=False,
        extra_api_params=None,
        return_last_frame=True,
        on_tasks_created=None,
    ):
        try:
            _raise_if_text_params(prompt, forbidden_params)

            api_seed = seed
            if enable_random_seed:
                api_seed = -1

            (duration_or_frames_arg, estimation_duration) = (
                _calculate_duration_and_frames_args(duration)
            )

            if is_auto_duration:
                estimation_duration = 5

            prompt_string = f"{prompt} --resolution {resolution} --ratio {aspect_ratio} {duration_or_frames_arg} --seed {api_seed} {prompt_extra_args.strip()}"
            content.insert(0, {"type": "text", "text": prompt_string})
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            return await self._execute_batch_generation(
                client=client,
                model_name=model_name,
                content=content,
                estimation_duration=estimation_duration,
                resolution=resolution,
                generation_count=generation_count,
                filename_prefix=filename_prefix,
                save_last_frame_batch=save_last_frame_batch,
                non_blocking=non_blocking,
                node_id=node_id,
                service_tier=service_tier,
                execution_expires_after=execution_expires_after,
                extra_api_params=extra_api_params,
                return_last_frame=return_last_frame,
                on_tasks_created=on_tasks_created,
            )
        except Exception as e:
            if isinstance(e, comfy.model_management.InterruptProcessingException):
                raise e
            s_e = str(e)
            if s_e.startswith("[JimengAI]"):
                raise e
            raise RuntimeError(format_api_error(e))


class JimengSeedance1(JimengVideoBase, comfy_io.ComfyNode):
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedance1",
            display_name="Jimeng Seedance 1.0",
            category=GLOBAL_CATEGORY,
            is_output_node=True,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Combo.Input(
                    "model_version",
                    options=VIDEO_UI_OPTIONS,
                    default=VIDEO_UI_OPTIONS[0],
                ),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Float.Input(
                    "duration",
                    default=5.0,
                    min=1.2,
                    max=12.0,
                    step=0.2,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "resolution", options=["480p", "720p", "1080p"], default="720p"
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio", options=cls.ASPECT_RATIOS, default="adaptive"
                ),
                comfy_io.Boolean.Input("camerafixed", default=True),
                comfy_io.Boolean.Input(
                    "enable_random_seed",
                    default=True,
                    tooltip="On=Enabled, Off=Disabled",
                ),
                comfy_io.Int.Input("seed", default=0, min=0, max=4294967295),
                comfy_io.Int.Input("generation_count", default=1, min=1),
                comfy_io.String.Input("filename_prefix", default="Jimeng/Video/Batch/Seedance"),
                comfy_io.Boolean.Input("save_last_frame_batch", default=False),
                comfy_io.Int.Input(
                    "timeout_seconds", default=172800, min=3600, max=259200
                ),
                comfy_io.Boolean.Input("enable_offline_inference", default=False),
                comfy_io.Boolean.Input("non_blocking", default=False),
                comfy_io.Image.Input("image", optional=True),
                comfy_io.Image.Input("last_frame_image", optional=True),
            ],
            hidden=[comfy_io.Hidden.unique_id],
            outputs=[
                comfy_io.Video.Output(display_name="video"),
                comfy_io.Image.Output(display_name="last_frame"),
                comfy_io.String.Output(display_name="response"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        model_version,
        prompt,
        duration,
        resolution,
        aspect_ratio,
        camerafixed,
        enable_random_seed,
        seed,
        generation_count,
        filename_prefix,
        save_last_frame_batch,
        timeout_seconds,
        enable_offline_inference,
        non_blocking,
        image=None,
        last_frame_image=None,
    ) -> comfy_io.NodeOutput:

        node_id = cls.hidden.unique_id

        final_model_name = ""
        if model_version in VIDEO_MODEL_MAP:
            final_model_name = VIDEO_MODEL_MAP[model_version]
        else:
            suffix = "-i2v" if image is not None else "-t2v"
            try_key = f"{model_version}{suffix}"
            final_model_name = VIDEO_MODEL_MAP.get(try_key)

        if not final_model_name:
            raise ValueError(f"Model ID not found for selection: {model_version}")

        helper = JimengVideoBase()
        helper.NON_BLOCKING_TASK_CACHE = cls.NON_BLOCKING_TASK_CACHE

        content = []
        helper._append_image_content(content, image, "first_frame")

        if last_frame_image is not None:
            if image is None:
                raise ValueError(get_text("popup_first_frame_missing"))
            helper._append_image_content(content, last_frame_image, "last_frame")

        service_tier, execution_expires_after = helper._get_service_options(
            enable_offline_inference, timeout_seconds
        )

        return await helper._common_generation_logic(
            client,
            prompt,
            duration,
            resolution,
            aspect_ratio,
            seed,
            generation_count,
            filename_prefix,
            save_last_frame_batch,
            non_blocking,
            node_id,
            model_name=final_model_name,
            content=content,
            forbidden_params=[
                "resolution",
                "ratio",
                "dur",
                "frames",
                "camerafixed",
                "seed",
            ],
            prompt_extra_args=f"--camerafixed {'true' if camerafixed else 'false'}",
            service_tier=service_tier,
            execution_expires_after=execution_expires_after,
            enable_random_seed=enable_random_seed,
        )


class JimengSeedance1_5(JimengVideoBase, comfy_io.ComfyNode):
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedance1_5",
            display_name="Jimeng Seedance 1.5 Pro",
            category=GLOBAL_CATEGORY,
            is_output_node=True,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Combo.Input(
                    "model_version",
                    options=VIDEO_1_5_UI_OPTIONS,
                    default=VIDEO_1_5_UI_OPTIONS[0],
                ),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip="Generate synchronized audio with video",
                ),
                comfy_io.Boolean.Input(
                    "auto_duration",
                    default=False,
                    tooltip="If enabled, overrides duration to -1 (Auto)",
                ),
                comfy_io.Int.Input(
                    "duration",
                    default=5,
                    min=4,
                    max=12,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "resolution", options=["480p", "720p", "1080p"], default="720p"
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio", options=cls.ASPECT_RATIOS, default="adaptive"
                ),
                comfy_io.Boolean.Input("camerafixed", default=True),
                comfy_io.Boolean.Input(
                    "enable_random_seed",
                    default=True,
                    tooltip="On=Enabled, Off=Disabled",
                ),
                comfy_io.Int.Input("seed", default=0, min=0, max=4294967295),
                comfy_io.Boolean.Input("draft_mode", default=False),
                comfy_io.Boolean.Input(
                    "reuse_last_draft_task",
                    default=False,
                    tooltip="Reuse the last task ID generated in Draft mode",
                ),
                comfy_io.String.Input("draft_task_id", default=""),
                comfy_io.Int.Input("generation_count", default=1, min=1),
                comfy_io.String.Input("filename_prefix", default="Jimeng/Video/Batch/Seedance"),
                comfy_io.Boolean.Input("save_last_frame_batch", default=False),
                comfy_io.Int.Input(
                    "timeout_seconds", default=172800, min=3600, max=259200
                ),
                comfy_io.Boolean.Input("enable_offline_inference", default=False),
                comfy_io.Boolean.Input("non_blocking", default=False),
                comfy_io.Image.Input("image", optional=True),
                comfy_io.Image.Input("last_frame_image", optional=True),
            ],
            hidden=[comfy_io.Hidden.unique_id],
            outputs=[
                comfy_io.Video.Output(display_name="video"),
                comfy_io.Image.Output(display_name="last_frame"),
                comfy_io.String.Output(display_name="response"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        model_version,
        prompt,
        generate_audio,
        auto_duration,
        duration,
        resolution,
        aspect_ratio,
        camerafixed,
        enable_random_seed,
        seed,
        generation_count,
        filename_prefix,
        save_last_frame_batch,
        timeout_seconds,
        enable_offline_inference,
        non_blocking,
        draft_mode,
        reuse_last_draft_task,
        draft_task_id,
        image=None,
        last_frame_image=None,
    ) -> comfy_io.NodeOutput:

        node_id = cls.hidden.unique_id

        global LAST_SEEDANCE_1_5_DRAFT_TASK_ID

        content_for_reuse = None

        if draft_task_id and draft_task_id.strip():
            content_for_reuse = [
                {
                    "type": "draft_task", 
                    "draft_task": {"id": draft_task_id.strip()}
                }
            ]
        
        elif reuse_last_draft_task:
            cached = LAST_SEEDANCE_1_5_DRAFT_TASK_ID.get(node_id)
            if cached:
                if generation_count == 1:
                    tid = None
                    if isinstance(cached, list) and len(cached) > 0:
                        tid = cached[0]
                    elif isinstance(cached, str):
                        tid = cached
                    
                    if tid:
                        content_for_reuse = [
                            {
                                "type": "draft_task", 
                                "draft_task": {"id": tid}
                            }
                        ]
                else:
                    ids_to_use = []
                    if isinstance(cached, list):
                        ids_to_use = cached
                    elif isinstance(cached, str):
                        ids_to_use = [cached]
                    
                    if ids_to_use:
                        content_for_reuse = []
                        for tid in ids_to_use:
                            content_for_reuse.append([
                                {
                                    "type": "draft_task", 
                                    "draft_task": {"id": tid}
                                }
                            ])

        final_model_name = VIDEO_MODEL_MAP.get(model_version)
        if not final_model_name:
            raise ValueError(f"Model ID not found for selection: {model_version}")

        helper = JimengVideoBase()
        helper.NON_BLOCKING_TASK_CACHE = cls.NON_BLOCKING_TASK_CACHE

        service_tier, execution_expires_after = helper._get_service_options(
            enable_offline_inference, timeout_seconds
        )
        
        if content_for_reuse:
            extra_params = {
                "resolution": resolution,
            }

            estimation_duration = 5 if auto_duration else float(duration)
            
            return await helper._execute_batch_generation(
                client=client,
                model_name=final_model_name,
                content=content_for_reuse,
                estimation_duration=estimation_duration,
                resolution=resolution,
                generation_count=generation_count,
                filename_prefix=filename_prefix,
                save_last_frame_batch=save_last_frame_batch,
                non_blocking=non_blocking,
                node_id=node_id,
                service_tier=service_tier,
                execution_expires_after=execution_expires_after,
                extra_api_params=extra_params,
                return_last_frame=True
            )

        content = []
        helper._append_image_content(content, image, "first_frame")

        if last_frame_image is not None:
            if image is None:
                raise ValueError(get_text("popup_first_frame_missing"))
            helper._append_image_content(content, last_frame_image, "last_frame")

        final_duration = -1.0 if auto_duration else float(duration)
        
        extra_api_params = {}
        should_return_last_frame = True
        final_resolution = resolution
        
        if draft_mode:
            extra_api_params["draft"] = True
            final_resolution = "480p"
            should_return_last_frame = False
            service_tier = "default"

        extra_args = f"--camerafixed {'true' if camerafixed else 'false'}"
        extra_args += f" --generate_audio {'true' if generate_audio else 'false'}"

        def _on_tasks_created(tasks):
            if draft_mode:
                try:
                    global LAST_SEEDANCE_1_5_DRAFT_TASK_ID
                    if tasks and len(tasks) > 0:
                        if generation_count == 1:
                            LAST_SEEDANCE_1_5_DRAFT_TASK_ID[node_id] = tasks[0].id
                        else:
                            LAST_SEEDANCE_1_5_DRAFT_TASK_ID[node_id] = [t.id for t in tasks]
                except Exception as e:
                    print(f"[JimengAI] Failed to record draft task ID: {e}")

        result = await helper._common_generation_logic(
            client,
            prompt,
            final_duration,
            final_resolution,
            aspect_ratio,
            seed,
            generation_count,
            filename_prefix,
            save_last_frame_batch if not draft_mode else False,
            non_blocking,
            node_id,
            model_name=final_model_name,
            content=content,
            forbidden_params=[
                "resolution",
                "ratio",
                "dur",
                "frames",
                "camerafixed",
                "seed",
                "generate_audio",
            ],
            prompt_extra_args=extra_args,
            service_tier=service_tier,
            execution_expires_after=execution_expires_after,
            enable_random_seed=enable_random_seed,
            is_auto_duration=auto_duration,
            extra_api_params=extra_api_params,
            return_last_frame=should_return_last_frame,
            on_tasks_created=_on_tasks_created,
        )

        return result


class JimengReferenceImage2Video(JimengVideoBase, comfy_io.ComfyNode):
    ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengReferenceImage2Video",
            display_name="Jimeng Reference to Video",
            category=GLOBAL_CATEGORY,
            is_output_node=True,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Float.Input(
                    "duration",
                    default=5.0,
                    min=1.2,
                    max=12.0,
                    step=0.2,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "resolution", options=["480p", "720p"], default="720p"
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio", options=cls.ASPECT_RATIOS, default="adaptive"
                ),
                comfy_io.Boolean.Input(
                    "enable_random_seed",
                    default=True,
                    tooltip="On=Enabled, Off=Disabled",
                ),
                comfy_io.Int.Input("seed", default=0, min=0, max=4294967295),
                comfy_io.Int.Input("generation_count", default=1, min=1),
                comfy_io.String.Input("filename_prefix", default="Jimeng/Video/Batch/Seedance"),
                comfy_io.Boolean.Input("save_last_frame_batch", default=False),
                comfy_io.Int.Input(
                    "timeout_seconds", default=172800, min=3600, max=259200
                ),
                comfy_io.Boolean.Input("enable_offline_inference", default=False),
                comfy_io.Boolean.Input("non_blocking", default=False),
                comfy_io.Image.Input("ref_image_1", optional=True),
                comfy_io.Image.Input("ref_image_2", optional=True),
                comfy_io.Image.Input("ref_image_3", optional=True),
                comfy_io.Image.Input("ref_image_4", optional=True),
            ],
            hidden=[comfy_io.Hidden.unique_id],
            outputs=[
                comfy_io.Video.Output(display_name="video"),
                comfy_io.Image.Output(display_name="last_frame"),
                comfy_io.String.Output(display_name="response"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        prompt,
        duration,
        resolution,
        aspect_ratio,
        enable_random_seed,
        seed,
        generation_count,
        filename_prefix,
        save_last_frame_batch,
        timeout_seconds,
        enable_offline_inference,
        non_blocking,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
    ) -> comfy_io.NodeOutput:

        node_id = cls.hidden.unique_id

        helper = JimengVideoBase()
        helper.NON_BLOCKING_TASK_CACHE = cls.NON_BLOCKING_TASK_CACHE

        content = []
        for img in [ref_image_1, ref_image_2, ref_image_3, ref_image_4]:
            helper._append_image_content(content, img, "reference_image")

        if not content:
            raise ValueError(get_text("popup_ref_missing"))

        service_tier, execution_expires_after = helper._get_service_options(
            enable_offline_inference, timeout_seconds
        )

        return await helper._common_generation_logic(
            client,
            prompt,
            duration,
            resolution,
            aspect_ratio,
            seed,
            generation_count,
            filename_prefix,
            save_last_frame_batch,
            non_blocking,
            node_id,
            model_name=REF_IMG_2_VIDEO_MODEL_ID,
            content=content,
            forbidden_params=["resolution", "ratio", "dur", "frames", "seed"],
            prompt_extra_args="",
            service_tier=service_tier,
            execution_expires_after=execution_expires_after,
            enable_random_seed=enable_random_seed,
        )


class JimengVideoQueryTasks(comfy_io.ComfyNode):
    MODELS = QUERY_TASKS_MODEL_LIST
    STATUSES = [
        "all",
        "succeeded",
        "failed",
        "running",
        "queued",
        "cancelled",
        "expired",
    ]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengVideoQueryTasks",
            display_name="Jimeng Video Query Tasks",
            category=GLOBAL_CATEGORY,
            is_output_node=True,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Int.Input("page_num", default=1),
                comfy_io.Int.Input("page_size", default=10),
                comfy_io.Combo.Input("status", options=cls.STATUSES, default="all"),
                comfy_io.Combo.Input(
                    "service_tier", options=["default", "flex"], default="default"
                ),
                comfy_io.String.Input("task_ids", default=""),
                comfy_io.Combo.Input(
                    "model_version", options=cls.MODELS, default="all"
                ),
                comfy_io.Int.Input("seed", default=0, min=0, max=4294967295),
            ],
            outputs=[
                comfy_io.String.Output(display_name="task_list_json"),
                comfy_io.Int.Output(display_name="total_tasks"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        page_num,
        page_size,
        status,
        service_tier,
        task_ids,
        model_version,
        seed,
    ) -> comfy_io.NodeOutput:
        ark_client = client.ark
        base_kwargs = {"page_num": page_num, "page_size": page_size}

        if status != "all":
            base_kwargs["status"] = status

        if service_tier:
            base_kwargs["service_tier"] = service_tier

        if task_ids and task_ids.strip():
            base_kwargs["task_ids"] = [
                tid.strip() for tid in task_ids.split("\n") if tid.strip()
            ]

        target_models = []
        if model_version == "all":
            target_models = [None]
        elif model_version == "doubao-seedance-1-0-lite":
            target_models.append(VIDEO_MODEL_MAP["doubao-seedance-1-0-lite-t2v"])
            target_models.append(VIDEO_MODEL_MAP["doubao-seedance-1-0-lite-i2v"])
        elif model_version in VIDEO_MODEL_MAP:
            target_models.append(VIDEO_MODEL_MAP[model_version])
        else:
            target_models.append(model_version)

        try:
            tasks = []
            for mid in target_models:
                kw = base_kwargs.copy()
                if mid is not None:
                    kw["model"] = mid
                tasks.append(
                    asyncio.to_thread(ark_client.content_generation.tasks.list, **kw)
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_items = []
            total_count = 0

            for res in results:
                if isinstance(res, Exception):
                    print(f"[JimengAI] Query Partial Error: {res}")
                    continue
                total_count += getattr(res, "total", 0)
                if hasattr(res, "items") and res.items:
                    for item in res.items:
                        item_dict = item.model_dump()
                        if "created_at" in item_dict and isinstance(
                            item_dict["created_at"], (int, float)
                        ):
                            item_dict["created_at_ts"] = item_dict["created_at"]
                            item_dict["created_at"] = datetime.datetime.fromtimestamp(
                                item_dict["created_at"]
                            ).strftime("%Y-%m-%d %H:%M:%S")
                        if "updated_at" in item_dict and isinstance(
                            item_dict["updated_at"], (int, float)
                        ):
                            item_dict["updated_at"] = datetime.datetime.fromtimestamp(
                                item_dict["updated_at"]
                            ).strftime("%Y-%m-%d %H:%M:%S")
                        all_items.append(item_dict)

            if not all_items and any(isinstance(r, Exception) for r in results):
                first_err = next(r for r in results if isinstance(r, Exception))
                return comfy_io.NodeOutput(
                    json.dumps(
                        {"error": format_api_error(first_err)}, ensure_ascii=False
                    ),
                    0,
                )

            all_items.sort(key=lambda x: x.get("created_at_ts", 0), reverse=True)
            for item in all_items:
                if "created_at_ts" in item:
                    del item["created_at_ts"]

            if len(target_models) > 1:
                all_items = all_items[:page_size]

            return comfy_io.NodeOutput(
                json.dumps(all_items, indent=2, ensure_ascii=False), total_count
            )
        except Exception as e:
            return comfy_io.NodeOutput(
                json.dumps({"error": format_api_error(e)}, ensure_ascii=False), 0
            )