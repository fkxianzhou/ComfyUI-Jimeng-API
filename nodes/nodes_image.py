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
    get_node_count_in_workflow,
    create_white_image_tensor,
)
from .utils_download import download_url_to_image_tensor_async
from .executor import JimengGenerationExecutor

from .models_config import (
    SEEDREAM_4_MODEL_MAP,
    SEEDREAM_5_MODEL_MAP,
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
    MIN_IMAGE_PIXELS_V5,
    MAX_IMAGE_PIXELS_V5,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
)
from .nodes_image_schema import (
    RECOMMENDED_SIZES_V3,
    RECOMMENDED_SIZES_V4,
    RECOMMENDED_SIZES_V5,
    get_image_size_inputs,
    get_common_generation_inputs,
)

def validate_custom_size(width, height, min_pixels, max_pixels):
    """
    验证自定义宽高是否符合模型的像素限制和宽高比限制。
    """
    total_pixels = width * height
    if not (min_pixels <= total_pixels <= max_pixels):
        raise JimengException(
            get_text("err_pixels_range").format(
                min=min_pixels,
                max=max_pixels,
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
            hidden=[comfy_io.Hidden.unique_id, comfy_io.Hidden.prompt],
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
        node_id = cls.hidden.unique_id
        ark_client = client.ark

        if size == "Custom":
            size_param = validate_custom_size(
                width,
                height,
                MIN_IMAGE_PIXELS_DEFAULT,
                MAX_IMAGE_PIXELS_DEFAULT,
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
        
        node_count = get_node_count_in_workflow("JimengSeedream3", prompt=cls.hidden.prompt)
        # log_msg("debug_node_count", count=node_count, type="JimengSeedream3")
        ignore_errors = node_count > 1
        
        executor = JimengGenerationExecutor(client, node_id, ignore_errors=ignore_errors)
        tensors, metadata = await executor.run_parallel_requests(generation_count, _generate_single)
        
        if tensors:
            try:
                count = tensors[0].shape[0] if isinstance(tensors, list) else tensors.shape[0]
                client.update_usage(model_id, count)
            except:
                pass
        
        if not tensors:
            # Should have been handled by executor, but just in case
             return comfy_io.NodeOutput(create_white_image_tensor(), "[]")

        output_tensor = torch.cat(tensors, dim=0) if isinstance(tensors, list) else tensors
        
        return comfy_io.NodeOutput(
            output_tensor, json.dumps(metadata, indent=2)
        )


class JimengSeedream5(comfy_io.ComfyNode):
    """
    Jimeng Seedream 5 图像生成节点。
    支持文生图、组图生成，并使用流式 API 接收结果。
    """
    RECOMMENDED_SIZES = RECOMMENDED_SIZES_V5

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedream5",
            display_name="Jimeng Seedream 5",
            category=GLOBAL_CATEGORY,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Combo.Input(
                    "model_version", options=list(SEEDREAM_5_MODEL_MAP.keys())
                ),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Boolean.Input(
                    "enable_group_generation",
                    default=False,
                    tooltip="On=Group, Off=Single",
                ),
                comfy_io.Int.Input("max_images", default=1, min=1, max=15),
                                comfy_io.Boolean.Input(
                    "enable_web_search",
                    default=False,
                    tooltip="Enable internet search capabilities",
                ),
            ]
            + get_image_size_inputs(cls.RECOMMENDED_SIZES, default_width=2048, default_height=2048)
            + get_common_generation_inputs()
            + [
                comfy_io.Image.Input("images", optional=True),
            ],
            hidden=[comfy_io.Hidden.unique_id, comfy_io.Hidden.prompt],
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
        enable_web_search,
        max_images,
        size,
        width,
        height,
        seed,
        generation_count,
        watermark,
        images=None,
    ) -> comfy_io.NodeOutput:
        node_id = cls.hidden.unique_id
        ark_client = client.ark

        model_id = SEEDREAM_5_MODEL_MAP.get(model_version)
        if not model_id:
            model_id = list(SEEDREAM_5_MODEL_MAP.values())[0]

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
            min_pixels = MIN_IMAGE_PIXELS_V5

            size_str = validate_custom_size(
                width,
                height,
                min_pixels,
                MAX_IMAGE_PIXELS_V5,
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
        
        node_count = get_node_count_in_workflow("JimengSeedream5", prompt=cls.hidden.prompt)
        # log_msg("debug_node_count", count=node_count, type="JimengSeedream5")
        ignore_errors = node_count > 1

        executor = JimengGenerationExecutor(client, node_id, ignore_errors=ignore_errors)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, MAX_SEED) if seed == -1 else seed + idx
            
            kwargs = {
                "model": model_id,
                "prompt": prompt,
                "size": size_str,
                "response_format": "url",
                "watermark": watermark,
                "seed": current_seed,
                "sequential_image_generation": sequential_param,
            }
            if enable_web_search:
                kwargs["extra_body"] = {"tools": [{"type": "web_search"}]}

            if image_param:
                kwargs["image"] = image_param
            if seq_options:
                kwargs["sequential_image_generation_options"] = seq_options
                
            return await executor.stream_generation_helper(
                session, ark_client, kwargs, idx, enable_group_generation, generation_count
            )

        tensors, metadata = await executor.run_parallel_requests(generation_count, _generate_single)

        if tensors:
            try:
                # Calculate total images generated (might be > generation_count if grouped)
                total_imgs = sum([t.shape[0] for t in tensors]) if isinstance(tensors, list) else tensors.shape[0]
                client.update_usage(model_id, total_imgs)
            except:
                pass
        
        if not tensors:
             return comfy_io.NodeOutput(create_white_image_tensor(), "[]")
             
        output_tensor = torch.cat(tensors, dim=0) if isinstance(tensors, list) else tensors
        
        return comfy_io.NodeOutput(
            output_tensor, json.dumps(metadata, indent=2)
        )


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
            hidden=[comfy_io.Hidden.unique_id, comfy_io.Hidden.prompt],
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
        node_id = cls.hidden.unique_id
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

            if "4.5" in model_version:
                min_pixels = MIN_IMAGE_PIXELS_V4_5

            size_str = validate_custom_size(
                width,
                height,
                min_pixels,
                MAX_IMAGE_PIXELS_V4,
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

        node_count = get_node_count_in_workflow("JimengSeedream4", prompt=cls.hidden.prompt)
        # log_msg("debug_node_count", count=node_count, type="JimengSeedream4")
        ignore_errors = node_count > 1

        executor = JimengGenerationExecutor(client, node_id, ignore_errors=ignore_errors)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, MAX_SEED) if seed == -1 else seed + idx
            
            kwargs = {
                "model": model_id,
                "prompt": prompt,
                "size": size_str,
                "response_format": "url",
                "watermark": watermark,
                "seed": current_seed,
                "sequential_image_generation": sequential_param,
            }
            if image_param:
                kwargs["image"] = image_param
            if seq_options:
                kwargs["sequential_image_generation_options"] = seq_options
                
            return await executor.stream_generation_helper(
                session, ark_client, kwargs, idx, enable_group_generation, generation_count
            )

        tensors, metadata = await executor.run_parallel_requests(generation_count, _generate_single)

        if tensors:
            try:
                total_imgs = sum([t.shape[0] for t in tensors]) if isinstance(tensors, list) else tensors.shape[0]
                client.update_usage(model_id, total_imgs)
            except:
                pass
        
        if not tensors:
             return comfy_io.NodeOutput(create_white_image_tensor(), "[]")

        output_tensor = torch.cat(tensors, dim=0) if isinstance(tensors, list) else tensors
        
        return comfy_io.NodeOutput(
            output_tensor, json.dumps(metadata, indent=2)
        )
