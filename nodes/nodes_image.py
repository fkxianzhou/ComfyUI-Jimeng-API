import random
import asyncio
import aiohttp
import torch
import comfy.model_management
import json
import time

from comfy_api.latest import io as comfy_io

from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions

from .nodes_shared import (
    GLOBAL_CATEGORY,
    _image_to_base64,
    get_text,
    log_msg,
    format_api_error,
    JimengClientType,
    JimengException,
)
from .utils_download import download_url_to_image_tensor_async

from .models_config import (
    SEEDREAM_4_MODEL_MAP,
    SEEDREAM_3_MODELS,
)
from .constants import (
    MAX_SEED,
    MIN_SEED,
    MAX_GENERATION_COUNT,
    MIN_IMAGE_PIXELS_DEFAULT,
    MAX_IMAGE_PIXELS_DEFAULT,
    MIN_IMAGE_PIXELS_V4_5,
    MAX_IMAGE_PIXELS_V4,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
)
from .nodes_image_schema import (
    RECOMMENDED_SIZES_V3,
    RECOMMENDED_SIZES_V4,
    get_image_size_inputs,
    get_common_generation_inputs,
)


class JimengImageHelper:
    """
    Jimeng 图像生成辅助类。
    处理尺寸验证、批量生成执行和结果处理。
    """
    def validate_custom_size(
        self, width, height, min_pixels, max_pixels, min_desc, max_desc
    ):
        """
        验证自定义宽高是否符合模型的像素限制和宽高比限制。
        """
        total_pixels = width * height
        if not (min_pixels <= total_pixels <= max_pixels):
            raise JimengException(
                get_text("err_pixels_range").format(
                    min=min_pixels,
                    min_desc=min_desc,
                    max=max_pixels,
                    max_desc=max_desc,
                    current=total_pixels,
                )
            )

        aspect_ratio = width / height
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            raise JimengException(
                get_text("err_aspect_ratio").format(
                    min="1/16", max="16", current=aspect_ratio
                )
            )
        return f"{width}x{height}"

    async def execute_generation(self, generation_count, generate_single_func):
        """
        执行批量生成任务。
        并发调用 generate_single_func 来生成图片。
        """
        async with aiohttp.ClientSession() as session:
            tasks = [generate_single_func(i, session) for i in range(generation_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return self.process_batch_results(results, generation_count)

    def process_batch_results(self, results, generation_count):
        """
        处理批量生成的结果。
        收集成功的结果，统计失败原因，并合并输出。
        """
        valid_tensors = []
        valid_responses = []
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
                valid_tensors.append(res[0])
                valid_responses.append(res[1])

        if generation_count > 1:
            log_msg(
                "batch_finished_stats",
                success=len(valid_tensors),
                failed=sum(error_counts.values()),
            )
            if error_counts:
                log_msg("batch_failed_summary", count=sum(error_counts.values()))
                for msg, count in error_counts.items():
                    log_msg("batch_failed_reason", msg=msg, count=count)

        if not valid_tensors:
            if generation_count > 1:
                log_msg("err_batch_fail_all")
            if first_exception:
                raise first_exception
            raise JimengException(get_text("err_batch_fail_all"))

        combined_response = sorted(valid_responses, key=lambda x: x["batch_index"])
        output_tensor = torch.cat(valid_tensors, dim=0)

        return comfy_io.NodeOutput(
            output_tensor, json.dumps(combined_response, indent=2)
        )


class JimengSeedream3(comfy_io.ComfyNode):
    """
    Jimeng Seedream 3 图像生成节点。
    支持文生图和图生图。
    """
    RECOMMENDED_SIZES = RECOMMENDED_SIZES_V3

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedream3",
            display_name="Jimeng Seedream 3",
            category=GLOBAL_CATEGORY,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Float.Input(
                    "guidance_scale", default=5.0, min=1.0, max=10.0, step=0.1
                ),
            ]
            + get_image_size_inputs(cls.RECOMMENDED_SIZES)
            + get_common_generation_inputs()
            + [
                comfy_io.Image.Input("image", optional=True),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="image"),
                comfy_io.String.Output(display_name="response"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        prompt,
        size,
        width,
        height,
        seed,
        generation_count,
        watermark,
        guidance_scale,
        image=None,
    ) -> comfy_io.NodeOutput:
        helper = JimengImageHelper()
        ark_client = client.ark

        if size == "Custom":
            size_param = helper.validate_custom_size(
                width,
                height,
                MIN_IMAGE_PIXELS_DEFAULT,
                MAX_IMAGE_PIXELS_DEFAULT,
                "512x512",
                "2048x2048",
            )
        else:
            size_param = size.split(" ")[0]

        image_param = None
        if image is None:
            model_id = SEEDREAM_3_MODELS["t2i"]
        else:
            model_id = SEEDREAM_3_MODELS["i2i"]
            image_param = f"data:image/jpeg;base64,{_image_to_base64(image)}"
            size_param = "adaptive"

        client.check_quota(model_id, generation_count)

        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_id)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, MAX_SEED) if seed == -1 else seed + idx

            def _call_api():
                kwargs = {
                    "model": model_id,
                    "prompt": prompt,
                    "size": size_param,
                    "response_format": "url",
                    "watermark": watermark,
                    "seed": current_seed,
                    "guidance_scale": guidance_scale,
                }
                if image_param:
                    kwargs["image"] = image_param

                return ark_client.images.generate(**kwargs)

            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                resp = await asyncio.to_thread(_call_api)
                comfy.model_management.throw_exception_if_processing_interrupted()

                image_tensor = await download_url_to_image_tensor_async(
                    session, resp.data[0].url
                )

                if image_tensor is None:
                    raise JimengException(get_text("err_download_img"))

                output_response = {
                    "batch_index": idx,
                    "model": resp.model,
                    "created": resp.created,
                    "url": resp.data[0].url,
                    "revised_prompt": getattr(resp.data[0], "revised_prompt", None),
                }
                return image_tensor, output_response
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise JimengException(format_api_error(e))

        result = await helper.execute_generation(generation_count, _generate_single)
        
        if result and result[0] is not None:
            try:
                count = result[0].shape[0]
                client.update_usage(model_id, count)
            except:
                pass
        
        return result


class JimengSeedream4(comfy_io.ComfyNode):
    """
    Jimeng Seedream 4 图像生成节点。
    支持文生图、组图生成，并使用流式 API 接收结果。
    """
    RECOMMENDED_SIZES = RECOMMENDED_SIZES_V4

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedream4",
            display_name="Jimeng Seedream 4",
            category=GLOBAL_CATEGORY,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Combo.Input(
                    "model_version", options=list(SEEDREAM_4_MODEL_MAP.keys())
                ),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Boolean.Input(
                    "enable_group_generation",
                    default=False,
                    tooltip="On=Group, Off=Single",
                ),
                comfy_io.Int.Input("max_images", default=1, min=1, max=15),
            ]
            + get_image_size_inputs(cls.RECOMMENDED_SIZES, default_width=2048, default_height=2048)
            + get_common_generation_inputs()
            + [
                comfy_io.Image.Input("images", optional=True),
            ],
            outputs=[
                comfy_io.Image.Output(display_name="images"),
                comfy_io.String.Output(display_name="response"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        model_version,
        prompt,
        enable_group_generation,
        max_images,
        size,
        width,
        height,
        seed,
        generation_count,
        watermark,
        images=None,
    ) -> comfy_io.NodeOutput:
        helper = JimengImageHelper()
        ark_client = client.ark

        model_id = SEEDREAM_4_MODEL_MAP.get(model_version)
        if not model_id:
            model_id = list(SEEDREAM_4_MODEL_MAP.values())[0]

        sequential_param = "auto" if enable_group_generation else "disabled"

        n_input_images = 0
        if images is not None:
            n_input_images = images.shape[0]

        if sequential_param == "auto":
            total_count = n_input_images + max_images
            if total_count > 15:
                raise JimengException(
                    get_text("err_img_limit_group_15").format(
                        n=n_input_images, max=max_images, total=total_count
                    )
                )

        if size == "Custom":
            min_pixels = 1280 * 720
            min_desc = "1280x720"

            if "4.5" in model_version:
                min_pixels = MIN_IMAGE_PIXELS_V4_5
                min_desc = "2560x1440"

            size_str = helper.validate_custom_size(
                width,
                height,
                min_pixels,
                MAX_IMAGE_PIXELS_V4,
                min_desc,
                "4096x4096",
            )
        else:
            size_str = size.split(" ")[0]

        image_param = None
        if images is not None:
            image_b64_list = [
                _image_to_base64(images[i : i + 1]) for i in range(n_input_images)
            ]
            if n_input_images == 1:
                image_param = f"data:image/jpeg;base64,{image_b64_list[0]}"
            else:
                image_param = [
                    f"data:image/jpeg;base64,{b64}" for b64 in image_b64_list
                ]

        seq_options = None
        if sequential_param == "auto":
            seq_options = SequentialImageGenerationOptions(max_images=max_images)

        client.check_quota(model_id, generation_count * max_images if enable_group_generation else generation_count)

        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_id)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, MAX_SEED) if seed == -1 else seed + idx

            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _producer_thread():
                """
                在独立线程中处理流式 API 响应，并将事件放入队列。
                """
                try:
                    kwargs = {
                        "model": model_id,
                        "prompt": prompt,
                        "size": size_str,
                        "response_format": "url",
                        "watermark": watermark,
                        "seed": current_seed,
                        "sequential_image_generation": sequential_param,
                        "stream": True,
                    }
                    if image_param:
                        kwargs["image"] = image_param
                    if seq_options:
                        kwargs["sequential_image_generation_options"] = seq_options

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
                "model": model_id,
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
                            kwargs = item.get("kwargs", {})
                            log_msg(key, **kwargs)
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

                return torch.cat(output_tensors, dim=0), final_metadata

            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise JimengException(str(e))

        result = await helper.execute_generation(generation_count, _generate_single)

        if result and result[0] is not None:
            try:
                count = result[0].shape[0]
                client.update_usage(model_id, count)
            except:
                pass
        
        return result
