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
from comfy_api.input_impl import VideoFromFile

from .nodes_shared import (
    GLOBAL_CATEGORY,
    _image_to_base64,
    log_msg,
    get_text,
    format_api_error,
    JimengClientType,
    JimengException,
)
from .nodes_video_schema import (
    get_common_video_inputs,
    get_duration_input,
    get_resolution_input,
    get_aspect_ratio_input,
    _calculate_duration_and_frames_args,
    ASPECT_RATIOS,
    resolve_model_id,
    resolve_query_models,
    VIDEO_UI_OPTIONS,
    VIDEO_1_5_UI_OPTIONS,
    QUERY_TASKS_MODEL_LIST,
    REF_IMG_2_VIDEO_MODEL_ID,
)
from .utils_download import (
    download_video_to_temp,
    download_image_to_temp,
    save_to_output,
)

from .task_runner import JimengBatchTaskRunner

logging.getLogger("volcenginesdkarkruntime").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


NON_BLOCKING_TASK_CACHE = {}
LAST_SEEDANCE_1_5_DRAFT_TASK_ID = {}


def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    for i in text_params:
        if f"--{i}" in prompt:
            raise JimengException(get_text("popup_param_not_allowed").format(param=i))


from .constants import VIDEO_MAX_SEED

class JimengVideoBase:
    """
    Jimeng 视频生成基类。
    提供通用的任务提交、结果处理和辅助方法。
    """
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
        # print(f"[JimengAI] {clean_msg}")
        if task_id:
            display_msg = get_text("popup_task_failed").format(
                task_id=task_id, msg=clean_msg
            )
        else:
            display_msg = get_text("popup_req_failed").format(msg=clean_msg)
        raise JimengException(display_msg)

    def _create_pending_json(self, status, task_id=None, task_count=0):
        if task_count > 0:
            msg = get_text("popup_batch_pending").format(count=task_count)
        else:
            msg = get_text("popup_task_pending").format(task_id=task_id, status=status)
        raise JimengException(msg)

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
        """
        异步处理批量任务成功的结果。
        下载视频和首尾帧，并整理输出。
        """
        # t_start = time.time()
        if generation_count > 1:
            log_msg("batch_handling", count=len(successful_tasks))

        temp_save_path = "Jimeng"
        video_prefix = "Jimeng_Vid_Temp"
        frame_prefix = "Jimeng_Frame_Temp"

        async def _process_task(task):
            video_url = task.content.video_url
            last_frame_url = getattr(task.content, "last_frame_url", None)
            seed = getattr(task, "seed", random.randint(0, VIDEO_MAX_SEED))

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
                    print(
                        f"[JimengAI] Warning: Failed to extract last frame locally: {e}"
                    )

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

        # t_end = time.time()
        # print(f"[JimengAI Debug] Batch handling finished in {t_end - t_start:.2f}s")
        
        return comfy_io.NodeOutput(
            first_video, first_frame, json.dumps(all_responses, indent=2)
        )

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
        service_tier="default",
        execution_expires_after=None,
        enable_random_seed=False,
        is_auto_duration=False,
        extra_api_params=None,
        return_last_frame=True,
        on_tasks_created=None,
    ):
        """
        通用的视频生成逻辑。
        处理参数准备、任务提交、轮询和结果处理。
        """
        try:
            _raise_if_text_params(prompt, forbidden_params)

            api_seed = seed
            if enable_random_seed:
                api_seed = -1

            if extra_api_params is None:
                extra_api_params = {}

            extra_api_params["resolution"] = resolution
            extra_api_params["ratio"] = aspect_ratio
            extra_api_params["seed"] = api_seed

            estimation_duration = 5
            if is_auto_duration:
                extra_api_params["duration"] = -1
            else:
                key, val, est = _calculate_duration_and_frames_args(duration)
                extra_api_params[key] = val
                estimation_duration = est

            content.insert(0, {"type": "text", "text": prompt})
            comfy.model_management.throw_exception_if_processing_interrupted()

            runner = JimengBatchTaskRunner(client, node_id)
            successful_tasks = await runner.run_batch(
                model_name=model_name,
                content=content,
                estimation_duration=estimation_duration,
                resolution=resolution,
                generation_count=generation_count,
                non_blocking=non_blocking,
                non_blocking_cache_dict=self.NON_BLOCKING_TASK_CACHE,
                service_tier=service_tier,
                execution_expires_after=execution_expires_after,
                extra_api_params=extra_api_params,
                return_last_frame=return_last_frame,
                on_tasks_created=on_tasks_created,
            )

            ret_results = None
            async with aiohttp.ClientSession() as session:
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
            s_e = str(e)
            if s_e.startswith("[JimengAI]"):
                raise e
            raise JimengException(format_api_error(e))


class JimengSeedance1(JimengVideoBase, comfy_io.ComfyNode):
    """
    Jimeng Seedance 1.0 视频生成节点。
    支持文生视频和图生视频。
    """
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
                get_duration_input(
                    default=5.0, min_val=1.2, max_val=12.0, step=0.2, is_int=False
                ),
                get_resolution_input(default="720p", support_1080p=True),
                get_aspect_ratio_input(default="adaptive", include_adaptive=True),
                comfy_io.Boolean.Input("camerafixed", default=True),
            ]
            + get_common_video_inputs()
            + [
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

        if image is None and aspect_ratio == "adaptive":
            aspect_ratio = "16:9"

        final_model_name = resolve_model_id(model_version, image)

        helper = JimengVideoBase()
        helper.NON_BLOCKING_TASK_CACHE = cls.NON_BLOCKING_TASK_CACHE

        content = []
        helper._append_image_content(content, image, "first_frame")

        if last_frame_image is not None:
            if image is None:
                raise JimengException(get_text("popup_first_frame_missing"))
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
            extra_api_params={"camera_fixed": camerafixed},
            service_tier=service_tier,
            execution_expires_after=execution_expires_after,
            enable_random_seed=enable_random_seed,
        )


class JimengSeedance1_5(JimengVideoBase, comfy_io.ComfyNode):
    """
    Jimeng Seedance 1.5 Pro 视频生成节点。
    支持文生视频、图生视频，以及草稿模式和草稿复用。
    """
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
                comfy_io.Boolean.Input("draft_mode", default=False),
                comfy_io.Boolean.Input("reuse_last_draft_task", default=False),
                comfy_io.String.Input("draft_task_id", default=""),
                comfy_io.Boolean.Input("generate_audio", default=True),
                comfy_io.Boolean.Input("auto_duration", default=False),
                get_duration_input(default=5, min_val=4, max_val=12, is_int=True),
                get_resolution_input(default="720p", support_1080p=True),
                get_aspect_ratio_input(default="adaptive", include_adaptive=True),
                comfy_io.Boolean.Input("camerafixed", default=True),
            ]
            + get_common_video_inputs()
            + [
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
                {"type": "draft_task", "draft_task": {"id": draft_task_id.strip()}}
            ]

        elif reuse_last_draft_task and draft_mode:
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
                            {"type": "draft_task", "draft_task": {"id": tid}}
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
                            content_for_reuse.append(
                                [{"type": "draft_task", "draft_task": {"id": tid}}]
                            )

        final_model_name = resolve_model_id(model_version, image)

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

            runner = JimengBatchTaskRunner(client, node_id)
            successful_tasks = await runner.run_batch(
                model_name=final_model_name,
                content=content_for_reuse,
                estimation_duration=estimation_duration,
                resolution=resolution,
                generation_count=generation_count,
                non_blocking=non_blocking,
                non_blocking_cache_dict=cls.NON_BLOCKING_TASK_CACHE,
                service_tier=service_tier,
                execution_expires_after=execution_expires_after,
                extra_api_params=extra_params,
                return_last_frame=True,
            )

            ret_results = None
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)
            ) as session:
                ret_results = await helper._handle_batch_success_async(
                    successful_tasks,
                    filename_prefix,
                    generation_count,
                    save_last_frame_batch,
                    session,
                )
                await asyncio.sleep(0.25)
            return ret_results

        content = []
        helper._append_image_content(content, image, "first_frame")

        if last_frame_image is not None:
            if image is None:
                raise JimengException(get_text("popup_first_frame_missing"))
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

        extra_api_params["camera_fixed"] = camerafixed
        extra_api_params["generate_audio"] = generate_audio

        def _on_tasks_created(tasks):
            if draft_mode:
                try:
                    global LAST_SEEDANCE_1_5_DRAFT_TASK_ID
                    if tasks and len(tasks) > 0:
                        if generation_count == 1:
                            LAST_SEEDANCE_1_5_DRAFT_TASK_ID[node_id] = tasks[0].id
                        else:
                            LAST_SEEDANCE_1_5_DRAFT_TASK_ID[node_id] = [
                                t.id for t in tasks
                            ]
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
    """
    Jimeng 参考图生视频节点。
    支持使用 1-4 张参考图生成视频。
    """
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
                get_duration_input(
                    default=5.0, min_val=1.2, max_val=12.0, step=0.2, is_int=False
                ),
                get_resolution_input(default="720p", support_1080p=False),
                get_aspect_ratio_input(default="16:9", include_adaptive=False),
            ]
            + get_common_video_inputs()
            + [
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
            raise JimengException(get_text("popup_ref_missing"))

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
            service_tier=service_tier,
            execution_expires_after=execution_expires_after,
            enable_random_seed=enable_random_seed,
        )


class JimengVideoQueryTasks(comfy_io.ComfyNode):
    """
    Jimeng 任务查询节点。
    用于查询历史任务状态和列表。
    """
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
                comfy_io.Int.Input("seed", default=0, min=0, max=VIDEO_MAX_SEED),
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

        target_models = resolve_query_models(model_version)

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
