import os
import io
import re
import time
import base64
import requests
import shutil
import time
import random
import datetime

import numpy
import PIL
import torch

from volcenginesdkarkruntime import Ark

import folder_paths
from comfy_api.util import VideoContainer

GLOBAL_CATEGORY = "JimengAI"

# --- Helper Functions (Best Practices) ---

def _fetch_data_from_url(url, stream=True):
    return requests.get(url, stream=stream).content

def _tensor2images(tensor):
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0.0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]

def _image_to_base64(image):
    if image is None:
        return None
    with io.BytesIO() as bytes_io:
        _tensor2images(image)[0].save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return base64.b64encode(data_bytes).decode("utf-8")

def _download_url_to_image_tensor(url: str):
    """Downloads an image from a URL and converts it to a ComfyUI IMAGE tensor."""
    if not url:
        return None
    try:
        image_data = _fetch_data_from_url(url)
        i = PIL.Image.open(io.BytesIO(image_data))
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        return torch.from_numpy(image)[None,]
    except Exception as e:
        print(f"Error downloading or converting image from URL {url}: {e}")
        return None

def _raise_if_text_params(prompt: str, text_params: list[str]) -> None:
    """Checks for command-line style parameters in the prompt and raises an error."""
    for i in text_params:
        if f"--{i}" in prompt:
            raise ValueError(
                f"Parameter '--{i}' is not allowed in the prompt. "
                f"Please use the dedicated node input for this value."
            )

def _poll_task_until_completion(client, task_id, timeout=600, interval=5):
    """Polls the task status until it's completed or fails."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            get_result = client.content_generation.tasks.get(task_id=task_id)
            if get_result.status == "succeeded":
                return get_result
            elif get_result.status in ["failed", "cancelled"]:
                error = getattr(get_result, 'error', None)
                if error:
                    raise RuntimeError(f"Task failed with code {error.code}: {error.message}")
                else:
                    raise RuntimeError(f"Task failed with status: {get_result.status}")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            print(f"Get task status failed, retrying... Error: {e}")
        time.sleep(interval)
    raise TimeoutError(f"Task polling timed out after {timeout} seconds for task_id: {task_id}")


# --- Node Classes ---

class JimengAPIClient:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"api_key": ("STRING", {"multiline": False, "default": ""})}}
    RETURN_TYPES = ("JIMENG_API_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_client"
    CATEGORY = GLOBAL_CATEGORY
    def create_client(self, api_key):
        return (Ark(api_key=api_key),)

class JimengText2Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (["1024x1024", "864x1152", "1152x864", "1280x720", "720x1280", "832x1248", "1248x832", "1512x648"],),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "watermark": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY
    def generate(self, client, prompt, size, seed, guidance_scale, watermark):
        if seed == -1:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = seed
        try:
            resp = client.images.generate(
                model="doubao-seedream-3-0-t2i-250415",
                prompt=prompt,
                response_format="url",
                size=size,
                seed=actual_seed,
                guidance_scale=guidance_scale,
                watermark=watermark
            )
            image_tensor = _download_url_to_image_tensor(resp.data[0].url)
            if image_tensor is None:
                raise RuntimeError("Failed to download the generated image.")
            return (image_tensor, actual_seed)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {e}")

class JimengImageEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "watermark": ("BOOLEAN", {"default": True}),
            },
        }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY
    def generate(self, client, image, prompt, seed, guidance_scale, watermark):
        if seed == -1:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = seed
        
        image_b64 = f"data:image/jpeg;base64,{_image_to_base64(image)}"

        try:
            resp = client.images.generate(
                model="doubao-seededit-3-0-i2i-250628",
                prompt=prompt,
                image=image_b64,
                response_format="url",
                size="adaptive",
                seed=actual_seed,
                guidance_scale=guidance_scale,
                watermark=watermark
            )
            image_tensor = _download_url_to_image_tensor(resp.data[0].url)
            if image_tensor is None:
                raise RuntimeError("Failed to download the edited image.")
            return (image_tensor, actual_seed)
        except Exception as e:
            raise RuntimeError(f"Failed to edit image: {e}")

class JimengSeedream4:
    RECOMMENDED_SIZES = [
        "2048x2048 (1:1)", "2304x1728 (4:3)", "1728x2304 (3:4)", 
        "2560x1440 (16:9)", "1440x2560 (9:16)", "2496x1664 (3:2)", 
        "1664x2496 (2:3)", "3024x1296 (21:9)", "4096x4096 (1:1)"
    ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (s.RECOMMENDED_SIZES,),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "seed")
    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY
    def generate(self, client, prompt, size, seed, watermark, images=None):
        if seed == -1:
            actual_seed = random.randint(0, 2147483647)
        else:
            actual_seed = seed
        
        size_str = size.split(" ")[0]
        
        image_param = None
        if images is not None:
            image_b64_list = []
            for i in range(images.shape[0]):
                single_image_tensor = images[i:i+1]
                image_b64_list.append(f"data:image/jpeg;base64,{_image_to_base64(single_image_tensor)}")
            
            if len(image_b64_list) == 1:
                image_param = image_b64_list[0]
            elif len(image_b64_list) > 1:
                image_param = image_b64_list

        try:
            resp = client.images.generate(
                model="doubao-seedream-4-0-250828",
                prompt=prompt,
                image=image_param,
                response_format="url",
                size=size_str,
                seed=actual_seed,
                watermark=watermark
            )
            
            output_tensors = []
            # --- FIX: Changed item["url"] to item.url ---
            for item in resp.data:
                tensor = _download_url_to_image_tensor(item.url)
                if tensor is not None:
                    output_tensors.append(tensor)
            # --- END FIX ---
            
            if not output_tensors:
                raise RuntimeError("Failed to download any of the generated images.")

            return (torch.cat(output_tensors, dim=0), actual_seed)
        except Exception as e:
            raise RuntimeError(f"Failed to generate with Seedream 4: {e}")

class JimengImage2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "image": ("IMAGE",),
                "model": (["doubao-seedance-1-0-lite-i2v-250428", "doubao-seedance-1-0-pro-250528"], {"default": "doubao-seedance-1-0-lite-i2v-250428"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("url", "task_id", "seed")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    def generate(self, client, image, model, prompt, duration, resolution, camerafixed, seed):
        _raise_if_text_params(prompt, ["resolution", "dur", "camerafixed", "seed"])
        
        if seed == -1:
            actual_seed = random.randint(0, 4294967295)
        else:
            actual_seed = seed
            
        base64_str = _image_to_base64(image)
        data_url = f"data:image/jpeg;base64,{base64_str}"

        prompt_string = f"{prompt} --resolution {resolution} --dur {duration} --camerafixed {'true' if camerafixed else 'false'} --seed {actual_seed}"

        try:
            create_result = client.content_generation.tasks.create(
                model=model,
                content=[
                    {"type": "text", "text": prompt_string},
                    {"type": "image_url", "image_url": {"url": data_url}, "role": "first_frame"}
                ]
            )
            task_id = create_result.id
        except Exception as e:
            raise RuntimeError(f"Create task failed: {e}")

        try:
            final_result = _poll_task_until_completion(client, task_id)
            return (final_result.content.video_url, task_id, actual_seed)
        except (RuntimeError, TimeoutError) as e:
            print(e)
            return ("", task_id, actual_seed)

class JimengFirstLastFrame2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "model": (["doubao-seedance-1-0-lite-i2v-250428"],),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "camerafixed": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("url", "task_id", "seed", "last_frame_url")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    def generate(self, client, first_frame_image, last_frame_image, model, prompt, duration, resolution, camerafixed, seed, return_last_frame):
        _raise_if_text_params(prompt, ["resolution", "dur", "camerafixed", "seed"])
        
        if seed == -1:
            actual_seed = random.randint(0, 4294967295)
        else:
            actual_seed = seed

        first_frame_data_url = f"data:image/jpeg;base64,{_image_to_base64(first_frame_image)}"
        last_frame_data_url = f"data:image/jpeg;base64,{_image_to_base64(last_frame_image)}"
        prompt_string = f"{prompt} --resolution {resolution} --dur {duration} --camerafixed {'true' if camerafixed else 'false'} --seed {actual_seed}"
        
        content = [
            {"type": "text", "text": prompt_string},
            {"type": "image_url", "image_url": {"url": first_frame_data_url}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": last_frame_data_url}, "role": "last_frame"}
        ]

        try:
            create_result = client.content_generation.tasks.create(
                model=model,
                content=content,
                return_last_frame=return_last_frame
            )
            task_id = create_result.id
        except Exception as e:
            raise RuntimeError(f"Create task failed: {e}")
        
        try:
            final_result = _poll_task_until_completion(client, task_id)
            last_frame_url = getattr(final_result.content, 'last_frame_url', "")
            return (final_result.content.video_url, task_id, actual_seed, last_frame_url)
        except (RuntimeError, TimeoutError) as e:
            print(e)
            return ("", task_id, actual_seed, "")

class JimengReferenceImage2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
            },
            "optional": {
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("url", "task_id", "seed")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    def generate(self, client, prompt, duration, resolution, seed, 
                 ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_image_4=None):
        _raise_if_text_params(prompt, ["resolution", "dur", "seed"])

        if seed == -1:
            actual_seed = random.randint(0, 4294967295)
        else:
            actual_seed = seed
        
        model = "doubao-seedance-1-0-lite-i2v-250428"
        prompt_string = f"{prompt} --resolution {resolution} --dur {duration} --seed {actual_seed}"

        content = [{"type": "text", "text": prompt_string}]
        ref_images = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        
        for img_tensor in ref_images:
            if img_tensor is not None:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img_tensor)}"},
                    "role": "reference_image"
                })
        
        if len(content) == 1:
            raise ValueError("At least one reference image must be provided.")

        try:
            create_result = client.content_generation.tasks.create(model=model, content=content)
            task_id = create_result.id
        except Exception as e:
            raise RuntimeError(f"Create task failed: {e}")

        try:
            final_result = _poll_task_until_completion(client, task_id)
            return (final_result.content.video_url, task_id, actual_seed)
        except (RuntimeError, TimeoutError) as e:
            print(e)
            return ("", task_id, actual_seed)

class JimengTaskStatusChecker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("JIMENG_API_CLIENT",),
                "task_id": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("video_url", "last_frame_url", "status", "error_message", "model", "created_at", "updated_at", "seed_out", "resolution_out", "ratio_out", "duration_out", "fps_out", "usage_info")
    FUNCTION = "check_status"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    def check_status(self, client, task_id):
        if not task_id:
            return ("", "", "no task_id provided", "Error: Task ID is empty.", "", "", "", -1, "", "", 0, 0, "")

        try:
            get_result = client.content_generation.tasks.get(task_id=task_id)
            
            status = get_result.status
            model = get_result.model
            created_at = datetime.datetime.fromtimestamp(get_result.created_at).strftime('%Y-%m-%d %H:%M:%S')
            updated_at = datetime.datetime.fromtimestamp(get_result.updated_at).strftime('%Y-%m-%d %H:%M:%S')
            
            video_url, last_frame_url, error_message, seed_out, res_out, ratio_out, dur_out, fps_out, usage_info = \
                "", "", "", -1, "", "", 0, 0, ""

            if status == "succeeded":
                video_url = get_result.content.video_url
                if hasattr(get_result.content, 'last_frame_url'):
                    last_frame_url = get_result.content.last_frame_url
                
                seed_out = getattr(get_result, 'seed', -1)
                res_out = getattr(get_result, 'resolution', '')
                ratio_out = getattr(get_result, 'ratio', '')
                dur_out = getattr(get_result, 'duration', 0)
                fps_out = getattr(get_result, 'framespersecond', 0)
                
                if hasattr(get_result, 'usage'):
                    usage_info = f"Total Tokens: {get_result.usage.total_tokens}"
                
            elif status == "failed":
                if get_result.error:
                    error_message = f"Code: {get_result.error.code}, Message: {get_result.error.message}"
            
            return (video_url, last_frame_url, status, error_message, model, created_at, updated_at, seed_out, res_out, ratio_out, dur_out, fps_out, usage_info)

        except Exception as e:
            return ("", "", "api_error", f"API Error: {e}", "", "", "", -1, "", "", 0, 0, "")

class PreviewImageFromUrl:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_url": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "Jimeng_Image"}),
             },
            "optional": {
                "add_seed_to_filename": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    OUTPUT_NODE = True
    CATEGORY = "JimengAI"

    def load_image(self, image_url, filename_prefix="Jimeng_Image", add_seed_to_filename=True, seed=None):
        if not image_url:
            raise ValueError("No image URL provided")

        image_data = _fetch_data_from_url(image_url)
        i = PIL.Image.open(io.BytesIO(image_data))
        
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        if 'A' in i.getbands():
            mask = numpy.array(i.getchannel('A')).astype(numpy.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        
        if add_seed_to_filename and seed is not None:
            filename_prefix = f"{filename_prefix}_seed_{seed}"
        
        (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{random.randint(1, 10000)}.png"
        
        with open(os.path.join(full_output_folder, file), "wb") as f:
            f.write(image_data)
            
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": self.type}]}, "result": (image, mask.unsqueeze(0))}

class PreviewVideoFromUrl:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "Jimeng_Video"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "format": (VideoContainer.as_input(), {"default": "mp4"}),
            },
            "optional": {
                "add_seed_to_filename": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"forceInput": True}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "JimengAI"
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    def run(self, video_url, filename_prefix, save_output, format, add_seed_to_filename=True, seed=None):
        if not video_url or not save_output:
            return {"ui": {"video": []}}

        if add_seed_to_filename and seed is not None:
            filename_prefix = f"{filename_prefix}_seed_{seed}"

        (full_output_folder, filename, _, subfolder, _) = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        counter = 0
        while True:
            file_ext = VideoContainer.get_extension(format)
            final_filename = f"{filename}_{counter:05}_.{file_ext}"
            final_path = os.path.join(full_output_folder, final_filename)
            if not os.path.exists(final_path):
                break
            counter += 1

        data = _fetch_data_from_url(video_url)
        with open(final_path, "wb") as f:
            f.write(data)

        results = [{
            "filename": final_filename,
            "subfolder": subfolder,
            "type": self.type
        }]
        
        return {
            "ui": {
                "images": results,
                "animated": (True,)
            }
        }

NODE_CLASS_MAPPINGS = {
    "JimengAPIClient": JimengAPIClient,
    "JimengText2Image": JimengText2Image,
    "JimengImageEdit": JimengImageEdit,
    "JimengSeedream4": JimengSeedream4,
    "JimengImage2Video": JimengImage2Video,
    "JimengFirstLastFrame2Video": JimengFirstLastFrame2Video,
    "JimengReferenceImage2Video": JimengReferenceImage2Video,
    "JimengTaskStatusChecker": JimengTaskStatusChecker,
    "PreviewImageFromUrl": PreviewImageFromUrl,
    "PreviewVideoFromUrl": PreviewVideoFromUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JimengAPIClient": "JimengAPIClient",
    "JimengText2Image": "JimengText2Image (Seedream 3)",
    "JimengImageEdit": "JimengImageEdit (Seededit 3)",
    "JimengSeedream4": "JimengSeedream4",
    "JimengImage2Video": "JimengImage2Video (I2V)",
    "JimengFirstLastFrame2Video": "JimengFirstLastFrame2Video (F&L-I2V)",
    "JimengReferenceImage2Video": "JimengReferenceImage2Video (Ref-I2V)",
    "JimengTaskStatusChecker": "JimengTaskStatusChecker",
    "PreviewImageFromUrl": "PreviewImageFromUrl",
    "PreviewVideoFromUrl": "PreviewVideoFromUrl",
}
