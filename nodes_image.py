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
)
from .utils_download import download_url_to_image_tensor_async

from .models_config import (
    SEEDREAM_4_MODEL_MAP,
    SEEDREAM_3_MODELS,
)


class JimengSeedream3(comfy_io.ComfyNode):
    RECOMMENDED_SIZES = [
        "Custom",
        "1024x1024 (1:1)",
        "864x1152 (3:4)",
        "1152x864 (4:3)",
        "1280x720 (16:9)",
        "720x1280 (9:16)",
        "832x1248 (2:3)",
        "1248x832 (3:2)",
        "1512x648 (21:9)",
    ]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedream3",
            display_name="Jimeng Seedream 3",
            category=GLOBAL_CATEGORY,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Combo.Input("size", options=cls.RECOMMENDED_SIZES),
                comfy_io.Int.Input("width", default=1024, min=1, max=8192),
                comfy_io.Int.Input("height", default=1024, min=1, max=8192),
                comfy_io.Int.Input("seed", default=0, min=-1, max=2147483647),
                comfy_io.Float.Input(
                    "guidance_scale", default=5.0, min=1.0, max=10.0, step=0.1
                ),
                comfy_io.Int.Input("generation_count", default=1, min=1, max=2048),
                comfy_io.Boolean.Input("watermark", default=False),
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
        guidance_scale,
        watermark,
        image=None,
    ) -> comfy_io.NodeOutput:
        ark_client = client.ark

        if size == "Custom":
            total_pixels = width * height
            min_pixels = 512 * 512
            max_pixels = 2048 * 2048
            if not (min_pixels <= total_pixels <= max_pixels):
                raise ValueError(
                    get_text("err_pixels_range").format(
                        min=min_pixels,
                        min_desc="512x512",
                        max=max_pixels,
                        max_desc="2048x2048",
                        current=total_pixels,
                    )
                )

            aspect_ratio = width / height
            if not (1 / 16 <= aspect_ratio <= 16):
                raise ValueError(
                    get_text("err_aspect_ratio").format(
                        min="1/16", max="16", current=aspect_ratio
                    )
                )

            size_param = f"{width}x{height}"
        else:
            size_param = size.split(" ")[0]

        image_param = None
        if image is None:
            model_id = SEEDREAM_3_MODELS["t2i"]
        else:
            model_id = SEEDREAM_3_MODELS["i2i"]
            image_param = f"data:image/jpeg;base64,{_image_to_base64(image)}"
            size_param = "adaptive"

        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_id)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, 2147483647) if seed == -1 else seed + idx

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
                    raise RuntimeError(get_text("err_download_img"))

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
                raise RuntimeError(format_api_error(e))

        async with aiohttp.ClientSession() as session:
            tasks = [_generate_single(i, session) for i in range(generation_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

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
            log_msg("err_batch_fail_all")
            if first_exception:
                raise first_exception
            raise RuntimeError(get_text("err_batch_fail_all"))

        combined_response = sorted(valid_responses, key=lambda x: x["batch_index"])
        output_tensor = torch.cat(valid_tensors, dim=0)

        return comfy_io.NodeOutput(
            output_tensor, json.dumps(combined_response, indent=2)
        )


class JimengSeedream4(comfy_io.ComfyNode):
    RECOMMENDED_SIZES = [
        "2K (adaptive)",
        "4K (adaptive)",
        "2048x2048 (1:1)",
        "2304x1728 (4:3)",
        "1728x2304 (3:4)",
        "2560x1440 (16:9)",
        "1440x2560 (9:16)",
        "2496x1664 (3:2)",
        "1664x2496 (2:3)",
        "3024x1296 (21:9)",
        "4096x4096 (1:1)",
        "Custom",
    ]

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
                comfy_io.Combo.Input("size", options=cls.RECOMMENDED_SIZES),
                comfy_io.Int.Input("width", default=2048, min=1, max=8192),
                comfy_io.Int.Input("height", default=2048, min=1, max=8192),
                comfy_io.Int.Input("seed", default=0, min=-1, max=2147483647),
                comfy_io.Int.Input("generation_count", default=1, min=1, max=2048),
                comfy_io.Boolean.Input("watermark", default=False),
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
                raise ValueError(
                    get_text("err_img_limit_group_15").format(
                        n=n_input_images, max=max_images, total=total_count
                    )
                )

        if size == "Custom":
            total_pixels = width * height

            min_pixels = 1280 * 720
            min_desc = "1280x720"

            if "4.5" in model_version:
                min_pixels = 3686400
                min_desc = "2560x1440"

            max_pixels = 4096 * 4096

            if not (min_pixels <= total_pixels <= max_pixels):
                raise ValueError(
                    get_text("err_pixels_range").format(
                        min=min_pixels,
                        min_desc=min_desc,
                        max=max_pixels,
                        max_desc="4096x4096",
                        current=total_pixels,
                    )
                )

            aspect_ratio = width / height

            if not (1 / 16 <= aspect_ratio <= 16):
                raise ValueError(
                    get_text("err_aspect_ratio").format(
                        min="1/16", max="16", current=aspect_ratio
                    )
                )
            size_str = f"{width}x{height}"
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

        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_id)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, 2147483647) if seed == -1 else seed + idx

            queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _producer_thread():
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
                                raise RuntimeError(
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
                        raise RuntimeError(format_api_error(item["error"]))
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
                    raise RuntimeError(get_text("err_batch_fail_all"))

                results = await asyncio.gather(*download_tasks)
                valid_results = [r for r in results if r[1] is not None]

                if not valid_results:
                    raise RuntimeError(get_text("err_download_img"))

                valid_results.sort(key=lambda x: x[0])

                output_tensors = []
                for v_idx, tensor, url in valid_results:
                    output_tensors.append(tensor)
                    final_metadata["images"].append({"url": url, "index": v_idx})

                return torch.cat(output_tensors, dim=0), final_metadata

            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(str(e))

        async with aiohttp.ClientSession() as session:
            tasks = [_generate_single(i, session) for i in range(generation_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

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
            log_msg("err_batch_fail_all")
            if first_exception:
                raise first_exception
            raise RuntimeError(get_text("err_batch_fail_all"))

        combined_response = sorted(valid_responses, key=lambda x: x["batch_index"])
        output_tensor = torch.cat(valid_tensors, dim=0)

        return comfy_io.NodeOutput(
            output_tensor,
            json.dumps(combined_response, indent=2),
        )
