import random
import asyncio
import aiohttp
import torch
import comfy.model_management
import json
import time
import re

from comfy_api.latest import io as comfy_io

from volcenginesdkarkruntime.types.images.images import (
    SequentialImageGenerationOptions,
    ContentGenerationTool,
    OptimizePromptOptions,
)

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
    safe_cat_tensors,
)
from .utils_download import download_url_to_image_tensor_async
from .executor import JimengGenerationExecutor

from .models_config import (
    SEEDREAM_4_MODEL_MAP,
    SEEDREAM_5_MODEL_MAP,
    SEEDREAM_5_PRO_UI_MODEL,
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
    MIN_IMAGE_PIXELS_V5_PRO,
    MAX_IMAGE_PIXELS_V5_PRO,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
)
from .nodes_image_schema import (
    RECOMMENDED_SIZES_V3,
    RECOMMENDED_SIZES_V4,
    RECOMMENDED_SIZES_V5,
    RECOMMENDED_SIZES_V5_PRO,
    get_image_generation_inputs,
)

SEEDREAM_5_PRO_SIZE_OVERRIDES = {
    "2848x1600": ("2K", "16:9"),
    "1600x2848": ("2K", "9:16"),
    "3136x1344": ("2K", "21:9"),
}


def _prompt_declares_aspect_ratio(prompt: str) -> bool:
    text = prompt or ""
    return bool(
        re.search(r"(?<!\d)\d+\s*[:：]\s*\d+(?!\d)", text, re.IGNORECASE)
    )


def prepare_seedream5_pro_size(prompt: str, size: str, width: int, height: int):
    if size == "Custom":
        if width % 16 != 0 or height % 16 != 0:
            raise JimengException(get_text("err_size_multiple_16"))
        size_value = validate_custom_size(
            width,
            height,
            MIN_IMAGE_PIXELS_V5_PRO,
            MAX_IMAGE_PIXELS_V5_PRO,
        )
        return prompt, size_value

    size_value = (size or "").split(" ")[0]
    override = SEEDREAM_5_PRO_SIZE_OVERRIDES.get(size_value)
    if not override:
        return prompt, size_value

    api_size, aspect_ratio = override
    final_prompt = prompt or ""
    if not _prompt_declares_aspect_ratio(final_prompt):
        final_prompt = f"{final_prompt.rstrip()}\nAspect Ratio: {aspect_ratio}".strip()
    return final_prompt, api_size

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


def _get_dynamic_input_order(name):
    match = None
    try:
        import re
        match = re.search(r"(\d+)$", str(name))
    except Exception:
        match = None
    if match:
        return int(match.group(1))
    return 999


def _create_seedream_autogrow_input():
    return comfy_io.Autogrow.Input(
        "images",
        template=comfy_io.Autogrow.TemplateNames(
            input=comfy_io.Image.Input("image", optional=True),
            names=[f"image_{idx}" for idx in range(1, 15)],
            min=1,
        ),
    )


def _prepare_multi_image_inputs(images=None, **kwargs):
    input_image_tensors = []

    if isinstance(images, dict):
        sorted_items = sorted(images.items(), key=lambda item: _get_dynamic_input_order(item[0]))
        for _, image_tensor in sorted_items:
            if isinstance(image_tensor, torch.Tensor):
                input_image_tensors.append(image_tensor)
    elif isinstance(images, torch.Tensor):
        input_image_tensors.append(images)

    for key in sorted([k for k in kwargs.keys() if k.startswith("image_")], key=_get_dynamic_input_order):
        image_tensor = kwargs[key]
        if isinstance(image_tensor, torch.Tensor):
            input_image_tensors.append(image_tensor)

    n_input_images = 0
    image_b64_list = []
    for tensor in input_image_tensors:
        batch_size = tensor.shape[0]
        n_input_images += batch_size
        for i in range(batch_size):
            image_b64_list.append(_image_to_base64(tensor[i : i + 1]))

    image_param = None
    if n_input_images > 0:
        if n_input_images == 1:
            image_param = f"data:image/jpeg;base64,{image_b64_list[0]}"
        else:
            image_param = [
                f"data:image/jpeg;base64,{image_b64}" for image_b64 in image_b64_list
            ]

    return n_input_images, image_param


class JimengSeedream3(comfy_io.ComfyNode):
    """
    Jimeng Seedream 3 图像生成节点。
    支持文生图。
    """
    RECOMMENDED_SIZES = RECOMMENDED_SIZES_V3

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedream3",
            display_name="Jimeng Seedream 3",
            category=GLOBAL_CATEGORY,
            is_deprecated=True,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.String.Input("prompt", multiline=True, default=""),
                comfy_io.Float.Input(
                    "guidance_scale", default=5.0, min=1.0, max=10.0, step=0.1
                ),
            ] + get_image_generation_inputs(cls.RECOMMENDED_SIZES),
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

        model_id = SEEDREAM_3_MODELS["t2i"]

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
             return comfy_io.NodeOutput(create_white_image_tensor(), "[]")

        output_tensor = safe_cat_tensors(tensors)
        
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
        generation_inputs = get_image_generation_inputs(
            cls.RECOMMENDED_SIZES,
            default_width=2048,
            default_height=2048,
            enable_group_generation=True,
        )
        generation_inputs.insert(
            len(generation_inputs) - 1,
            comfy_io.Boolean.Input(
                "thinking",
                default=True,
                tooltip=(
                    "Optimize the prompt before generation. Supported by "
                    "Seedream 4.0 only."
                ),
            ),
        )
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
            ]
            + generation_inputs
            + [_create_seedream_autogrow_input()],
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
        thinking=True,
        images=None,
        **kwargs,
    ) -> comfy_io.NodeOutput:
        node_id = cls.hidden.unique_id
        ark_client = client.ark

        model_id = SEEDREAM_4_MODEL_MAP.get(model_version)
        if not model_id:
            model_id = list(SEEDREAM_4_MODEL_MAP.values())[0]

        sequential_param = "auto" if enable_group_generation else "disabled"
        n_input_images, image_param = _prepare_multi_image_inputs(images, **kwargs)

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
                "response_format": "b64_json",
                "watermark": watermark,
                "seed": current_seed,
                "sequential_image_generation": sequential_param,
            }
            if image_param:
                kwargs["image"] = image_param
            if seq_options:
                kwargs["sequential_image_generation_options"] = seq_options
            if (
                model_version == "doubao-seedream-4.0"
                and thinking
            ):
                kwargs["optimize_prompt_options"] = OptimizePromptOptions(
                    mode="standard"
                )
                
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

        output_tensor = safe_cat_tensors(tensors)
        
        return comfy_io.NodeOutput(
            output_tensor, json.dumps(metadata, indent=2)
        )

class JimengSeedream5(comfy_io.ComfyNode):
    """
    Jimeng Seedream 5 图像生成节点。
    支持文生图、组图生成，并使用流式 API 接收结果。
    """
    RECOMMENDED_SIZES = RECOMMENDED_SIZES_V5

    @staticmethod
    def _model_inputs(model_version):
        if model_version == SEEDREAM_5_PRO_UI_MODEL:
            return [
                comfy_io.String.Input("prompt", multiline=True, default=""),
                *get_image_generation_inputs(
                    RECOMMENDED_SIZES_V5_PRO,
                    default_width=2048,
                    default_height=2048,
                )[:-2],
                comfy_io.Int.Input(
                    "generation_count", default=1, min=1, max=MAX_GENERATION_COUNT
                ),
                comfy_io.Boolean.Input(
                    "thinking",
                    default=True,
                    tooltip=(
                        "Optimize the prompt before generation. Required when "
                        "Seedream 5 Pro uses reference images."
                    ),
                ),
                comfy_io.Boolean.Input("watermark", default=False),
            ]

        return [
            comfy_io.String.Input("prompt", multiline=True, default=""),
            *get_image_generation_inputs(
                RECOMMENDED_SIZES_V5,
                default_width=2048,
                default_height=2048,
                enable_group_generation=True,
                enable_web_search=True,
            ),
        ]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengSeedream5",
            display_name="Jimeng Seedream 5",
            category=GLOBAL_CATEGORY,
            description=(
                "Generate images with Seedream 5 Pro or Lite. Pro supports prompt "
                "optimization; Lite supports grouped generation and web search."
            ),
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.DynamicCombo.Input(
                    "model_version",
                    options=[
                        comfy_io.DynamicCombo.Option(
                            model_version, cls._model_inputs(model_version)
                        )
                        for model_version in SEEDREAM_5_MODEL_MAP
                    ],
                ),
                _create_seedream_autogrow_input(),
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
        prompt="",
        enable_group_generation=False,
        enable_web_search=False,
        max_images=1,
        size="2K (Adaptive)",
        width=2048,
        height=2048,
        seed=0,
        generation_count=1,
        watermark=False,
        thinking=True,
        images=None,
        **kwargs,
    ) -> comfy_io.NodeOutput:
        node_id = cls.hidden.unique_id
        ark_client = client.ark

        if isinstance(model_version, dict):
            model_config = model_version
            model_version = model_config.get("model_version", SEEDREAM_5_PRO_UI_MODEL)
            prompt = model_config.get("prompt", prompt)
            enable_group_generation = model_config.get(
                "enable_group_generation", enable_group_generation
            )
            enable_web_search = model_config.get("enable_web_search", enable_web_search)
            max_images = model_config.get("max_images", max_images)
            size = model_config.get("size", size)
            width = model_config.get("width", width)
            height = model_config.get("height", height)
            seed = model_config.get("seed", seed)
            generation_count = model_config.get("generation_count", generation_count)
            watermark = model_config.get("watermark", watermark)
            thinking = model_config.get("thinking", thinking)

        model_id = SEEDREAM_5_MODEL_MAP.get(model_version)
        if not model_id:
            raise JimengException(get_text("err_model_not_supported").format(model=model_version))

        is_pro = model_version == SEEDREAM_5_PRO_UI_MODEL

        sequential_param = "auto" if enable_group_generation and not is_pro else "disabled"
        n_input_images, image_param = _prepare_multi_image_inputs(images, **kwargs)

        if is_pro and n_input_images > 10:
            raise JimengException(get_text("err_img_limit_10"))
        if is_pro and n_input_images and not thinking:
            raise JimengException(get_text("err_seedream5_pro_thinking_required"))

        if sequential_param == "auto":
            total_count = n_input_images + max_images
            if total_count > 15:
                raise JimengException(
                    get_text("err_img_limit_group_15").format(
                        n=n_input_images, max=max_images, total=total_count
                    )
                )

        if is_pro:
            prompt, size_str = prepare_seedream5_pro_size(prompt, size, width, height)
        elif size == "Custom":
            min_pixels = MIN_IMAGE_PIXELS_V5

            size_str = validate_custom_size(
                width,
                height,
                min_pixels,
                MAX_IMAGE_PIXELS_V5,
            )
        else:
            size_str = size.split(" ")[0]

        seq_options = None
        if sequential_param == "auto":
            seq_options = SequentialImageGenerationOptions(max_images=max_images)

        client.check_quota(
            model_id,
            generation_count * max_images
            if enable_group_generation and not is_pro
            else generation_count,
        )

        if generation_count > 1:
            log_msg("batch_submit_start", count=generation_count, model=model_id)
        
        node_count = get_node_count_in_workflow("JimengSeedream5", prompt=cls.hidden.prompt)
        # log_msg("debug_node_count", count=node_count, type="JimengSeedream5")
        ignore_errors = node_count > 1

        executor = JimengGenerationExecutor(client, node_id, ignore_errors=ignore_errors)

        async def _generate_single(idx, session):
            current_seed = random.randint(0, MAX_SEED) if seed == -1 else seed + idx

            if is_pro:
                request_kwargs = {
                    "model": model_id,
                    "prompt": prompt,
                    "size": size_str,
                    "response_format": "url",
                    "watermark": watermark,
                    "seed": current_seed,
                    "optimize_prompt_options": OptimizePromptOptions(
                        thinking="enabled" if thinking else "disabled"
                    ),
                }
                if image_param:
                    request_kwargs["image"] = image_param

                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    response = await asyncio.to_thread(
                        ark_client.images.generate, **request_kwargs
                    )
                    comfy.model_management.throw_exception_if_processing_interrupted()

                    urls = [
                        getattr(item, "url", None)
                        for item in (getattr(response, "data", None) or [])
                    ]
                    urls = [url for url in urls if url]
                    if not urls:
                        raise JimengException(get_text("err_download_img"))

                    downloaded = await asyncio.gather(
                        *[
                            download_url_to_image_tensor_async(session, url)
                            for url in urls
                        ]
                    )
                    downloaded = [tensor for tensor in downloaded if tensor is not None]
                    if not downloaded:
                        raise JimengException(get_text("err_download_img"))

                    metadata = {
                        "batch_index": idx,
                        "model": getattr(response, "model", model_id),
                        "created": getattr(response, "created", None),
                        "urls": urls,
                    }
                    return safe_cat_tensors(downloaded), metadata
                except comfy.model_management.InterruptProcessingException:
                    raise
                except Exception as exc:
                    if isinstance(exc, JimengException):
                        raise
                    raise JimengException(format_api_error(exc))
            
            request_kwargs = {
                "model": model_id,
                "prompt": prompt,
                "size": size_str,
                "response_format": "b64_json",
                "watermark": watermark,
                "seed": current_seed,
                "sequential_image_generation": sequential_param,
            }
            if enable_web_search:
                request_kwargs["tools"] = [ContentGenerationTool(type="web_search")]

            if image_param:
                request_kwargs["image"] = image_param
            if seq_options:
                request_kwargs["sequential_image_generation_options"] = seq_options
                
            return await executor.stream_generation_helper(
                session,
                ark_client,
                request_kwargs,
                idx,
                enable_group_generation,
                generation_count,
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
             
        output_tensor = safe_cat_tensors(tensors)
        
        return comfy_io.NodeOutput(
            output_tensor, json.dumps(metadata, indent=2)
        )
