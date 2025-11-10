import random
import asyncio
import aiohttp
import torch
import comfy.model_management
import json

from .nodes_shared import (
    GLOBAL_CATEGORY, 
    _image_to_base64, 
    _download_url_to_image_tensor_async
)

class JimengSeedream3:
    """Seedream 3 & Seededit 3 图像生成。"""
    RECOMMENDED_SIZES = ["Custom", "1024x1024 (1:1)", "864x1152 (3:4)", "1152x864 (4:3)", "1280x720 (16:9)", "720x1280 (9:16)", "832x1248 (2:3)", "1248x832 (3:2)", "1512x648 (21:9)"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (s.RECOMMENDED_SIZES,),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, size, width, height, seed, guidance_scale, watermark, image=None):
        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        openai_client = client.openai

        model_id = ""
        extra_body = {
            "seed": actual_seed,
            "guidance_scale": guidance_scale,
            "watermark": watermark
        }
        
        if size == "Custom":
            total_pixels = width * height
            min_pixels = 512 * 512  
            max_pixels = 2048 * 2048 
            if not (min_pixels <= total_pixels <= max_pixels):
                raise ValueError(f"Total pixels must be between {min_pixels} (512x512) and {max_pixels} (2048x2048). Your current: {total_pixels}")

            aspect_ratio = width / height
            if not (1/16 <= aspect_ratio <= 16):
                raise ValueError(f"Aspect ratio must be between 1/16 and 16. Your current: {aspect_ratio}")
                
            size_param = f"{width}x{height}"
        else:
            size_param = size.split(" ")[0]

        if image is None:
            model_id = "doubao-seedream-3-0-t2i-250415"
        else:
            model_id = "doubao-seededit-3-0-i2i-250628"
            image_b64 = f"data:image/jpeg;base64,{_image_to_base64(image)}"
            extra_body["image"] = image_b64
            size_param = "adaptive"

        async with aiohttp.ClientSession() as session:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                resp = await openai_client.images.generate(
                    model=model_id,
                    prompt=prompt,
                    size=size_param,
                    response_format="url",
                    extra_body=extra_body
                )
                
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                image_tensor = await _download_url_to_image_tensor_async(session, resp.data[0].url)
                if image_tensor is None:
                    raise RuntimeError("Failed to download the generated image.")
                
                output_response = {
                    "model": resp.model,
                    "created": resp.created,
                    "url": resp.data[0].url,
                    "revised_prompt": getattr(resp.data[0], 'revised_prompt', None)
                }
                return (image_tensor, json.dumps(output_response, indent=2))
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to generate image with model {model_id}: {e}")

class JimengSeedream4:
    """Seedream4 文生图与图生图。"""
    RECOMMENDED_SIZES = [ "Custom", "2048x2048 (1:1)", "2304x1728 (4:3)", "1728x2304 (3:4)", "2560x1440 (16:9)", "1440x2560 (9:16)", "2496x1664 (3:2)", "1664x2496 (2:3)", "3024x1296 (21:9)", "4096x4096 (1:1)" ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "generation_mode": (["Single Image (disabled)", "Image Group (auto)"],),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "size": (s.RECOMMENDED_SIZES,),
                "width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    async def generate(self, client, prompt, generation_mode, max_images, size, width, height, seed, watermark, images=None):
        sequential_image_generation = "disabled" if "disabled" in generation_mode else "auto"
        n_input_images = 0
        if images is not None: n_input_images = images.shape[0]
        if n_input_images > 10: raise ValueError("The number of input images cannot exceed 10.")
        if sequential_image_generation == "auto" and n_input_images + max_images > 15:
            raise ValueError(f"The sum of input images ({n_input_images}) and max generated images ({max_images}) cannot exceed 15.")

        actual_seed = random.randint(0, 2147483647) if seed == -1 else seed
        
        if size == "Custom":
            total_pixels = width * height
            min_pixels = 1280 * 720
            max_pixels = 4096 * 4096
            if not (min_pixels <= total_pixels <= max_pixels):
                raise ValueError(f"Total pixels must be between {min_pixels} and {max_pixels}. Your current: {total_pixels}")

            aspect_ratio = width / height
            if not (1/16 <= aspect_ratio <= 16):
                raise ValueError(f"Aspect ratio must be between 1/16 and 16. Your current: {aspect_ratio}")

            size_str = f"{width}x{height}"
        else:
            size_str = size.split(" ")[0]
        
        image_param = None
        if images is not None:
            image_b64_list = [_image_to_base64(images[i:i+1]) for i in range(n_input_images)]
            if n_input_images == 1: image_param = f"data:image/jpeg;base64,{image_b64_list[0]}"
            else: image_param = [f"data:image/jpeg;base64,{b64}" for b64 in image_b64_list]

        extra_body = {"seed": actual_seed, "watermark": watermark, "sequential_image_generation": sequential_image_generation}
        if image_param: extra_body["image"] = image_param
        if sequential_image_generation == "auto":
            extra_body['sequential_image_generation_options'] = {"max_images": max_images}

        openai_client = client.openai
        async with aiohttp.ClientSession() as session:
            try:
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                resp = await openai_client.images.generate(model="doubao-seedream-4-0-250828", prompt=prompt, size=size_str, response_format="url", extra_body=extra_body)
                
                comfy.model_management.throw_exception_if_processing_interrupted()
                
                download_tasks = [_download_url_to_image_tensor_async(session, item.url) for item in resp.data]
                output_tensors = await asyncio.gather(*download_tasks)
                valid_tensors = [t for t in output_tensors if t is not None]
                if not valid_tensors: raise RuntimeError("Failed to download any of the generated images.")
                
                image_data_list = []
                for item in resp.data:
                    image_data_list.append({
                        "url": item.url,
                        "revised_prompt": getattr(item, 'revised_prompt', None)
                    })
                
                output_response = {
                    "model": resp.model,
                    "created": resp.created,
                    "images": image_data_list
                }
                return (torch.cat(valid_tensors, dim=0), json.dumps(output_response, indent=2))
            except Exception as e:
                if isinstance(e, comfy.model_management.InterruptProcessingException):
                    raise e
                raise RuntimeError(f"Failed to generate with Seedream 4: {e}")

NODE_CLASS_MAPPINGS = {
    "JimengSeedream3": JimengSeedream3,
    "JimengSeedream4": JimengSeedream4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengSeedream3": "Jimeng Seedream 3",
    "JimengSeedream4": "Jimeng Seedream 4",
}