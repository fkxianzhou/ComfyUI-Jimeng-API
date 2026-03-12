import base64
import json
import os
import asyncio
import torch
import aiohttp
import tempfile
from comfy_api.latest import io as comfy_io
from PIL import Image

try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    VideoFromFile = None

from .nodes_shared import (
    GLOBAL_CATEGORY,
    JimengClientType,
    JimengException,
    _tensor2images,
    log_msg,
    format_api_error,
    upload_file_to_ark
)
from .executor import JimengVisualExecutor
from .models_config import VISUAL_MODEL_MAP, VISUAL_UI_OPTIONS

LAST_RESPONSE_ID = None

class JimengVisualUnderstanding(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengVisualUnderstanding",
            display_name="Jimeng Visual Understanding",
            category=GLOBAL_CATEGORY,
            is_experimental=True,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Combo.Input("model", options=VISUAL_UI_OPTIONS, default=VISUAL_UI_OPTIONS[0]),
                comfy_io.String.Input("system_prompt", multiline=True, default="你叫豆包，是由字节跳动开发的AI。"),
                comfy_io.String.Input("user_prompt", multiline=True, default="请描述这张图片或视频的内容。"),
                comfy_io.Combo.Input("detail", options=["low", "high"], default="high"),
                comfy_io.Float.Input("fps", default=1.0, min=0.2, max=5.0, step=0.1),
                comfy_io.Combo.Input("reasoning_mode", options=["auto", "enabled", "disabled"], default="auto"),
                comfy_io.Combo.Input("reasoning_effort", options=["minimal", "low", "medium", "high"], default="medium"),
                comfy_io.Int.Input("turns", default=1, min=1, max=10),
                comfy_io.Boolean.Input("stream", default=False, display_name="Enable Streaming"),
                comfy_io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff), 
                
                comfy_io.MultiType.Input(
                    comfy_io.Image.Input("visual_input_1", optional=True),
                    types=[comfy_io.Image, comfy_io.Video]
                ),
                comfy_io.MultiType.Input(
                    comfy_io.Image.Input("visual_input_2", optional=True),
                    types=[comfy_io.Image, comfy_io.Video]
                ),
                comfy_io.MultiType.Input(
                    comfy_io.Image.Input("visual_input_3", optional=True),
                    types=[comfy_io.Image, comfy_io.Video]
                ),
            ],
            outputs=[
                comfy_io.String.Output(display_name="content"),
                comfy_io.String.Output(display_name="raw_json"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        client,
        model,
        system_prompt,
        user_prompt,
        seed, 
        stream,
        detail,
        fps,
        reasoning_mode="auto",
        reasoning_effort="medium",
        turns=1,
        visual_input_1=None,
        visual_input_2=None,
        visual_input_3=None,
    ) -> comfy_io.NodeOutput:
        
        inputs_content = []
        
        if user_prompt:
            inputs_content.append({"type": "input_text", "text": user_prompt})

        visual_inputs = [visual_input_1, visual_input_2, visual_input_3]
        
        for i, v_input in enumerate(visual_inputs):
            if v_input is None:
                continue
            
            log_msg("visual_processing_input", i=i+1, type=type(v_input))

            file_id = None
            input_type = "input_image"

            file_path = None
            
            if isinstance(v_input, torch.Tensor):
                pil_imgs = _tensor2images(v_input)
                if len(pil_imgs) > 0:
                    img = pil_imgs[0]
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.save(tmp.name, format="JPEG", quality=95)
                        file_path = tmp.name
                    input_type = "input_image"
                    
            else:
                if isinstance(v_input, str):
                    file_path = v_input
                elif VideoFromFile and isinstance(v_input, VideoFromFile):
                    file_path = getattr(v_input, "path", None)
                    if not file_path and hasattr(v_input, "_VideoFromFile__file"):
                        potential_file = getattr(v_input, "_VideoFromFile__file")
                        if isinstance(potential_file, str):
                             file_path = potential_file

                elif hasattr(v_input, "path"):
                    file_path = v_input.path
                elif isinstance(v_input, dict):
                    if "video_path" in v_input:
                        file_path = v_input["video_path"]
                    elif "filenames" in v_input:
                        files = v_input["filenames"]
                        if isinstance(files, list) and len(files) > 0:
                            file_path = files[0]
                    elif "path" in v_input:
                        file_path = v_input["path"]

                if file_path and not os.path.exists(file_path):
                     import folder_paths
                     input_dir = folder_paths.get_input_directory()
                     potential_path = os.path.join(input_dir, file_path)
                     if os.path.exists(potential_path):
                         log_msg("visual_found_file", path=potential_path)
                         file_path = potential_path
            
            if file_path and os.path.exists(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"]:
                    input_type = "input_video"
            
            if file_path and os.path.exists(file_path):
                try:
                    file_id = await upload_file_to_ark(client, file_path, fps=fps if input_type == "input_video" else None)
                finally:
                    if isinstance(v_input, torch.Tensor) and file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            log_msg("visual_warn_del_temp", path=file_path, e=e)
            else:
                 log_msg("visual_warn_del_temp", path=file_path, e="File does not exist")

            if file_id:
                content_item = {
                    "type": input_type,
                    "file_id": file_id
                }
                if input_type == "input_image":
                    content_item["detail"] = detail
                
                inputs_content.append(content_item)
            else:
                log_msg("visual_warn_no_id", i=i+1)

        global LAST_RESPONSE_ID
        
        full_content = ""
        final_json_str = "{}"
        
        previous_response_id = None
        
        if turns > 1:
            if LAST_RESPONSE_ID:
                previous_response_id = LAST_RESPONSE_ID
                log_msg("visual_cont_conv", id=previous_response_id)
            else:
                log_msg("visual_warn_no_history")
        else:
            log_msg("visual_new_conv")
        
        model_id = VISUAL_MODEL_MAP.get(model, model)
        
        payload = {
            "model": model_id,
        }
        
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
            
            follow_up_content = []
            if user_prompt:
                follow_up_content.append({"type": "input_text", "text": user_prompt})
            else:
                follow_up_content.append({"type": "input_text", "text": "Continue."}) 
            
            payload["input"] = [
                {
                    "role": "user",
                    "content": follow_up_content
                }
            ]
            
        else:
            payload["input"] = [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": inputs_content
                }
            ]
        
        if reasoning_mode != "auto":
            if "thinking" not in payload:
                payload["thinking"] = {}
            payload["thinking"]["type"] = reasoning_mode
            
        if reasoning_effort != "medium":
            if "reasoning" not in payload:
                payload["reasoning"] = {}
            payload["reasoning"]["effort"] = reasoning_effort
            
        current_response_json = {}
        
        executor = JimengVisualExecutor(client)

        if stream:
            full_content, final_json_str = await executor.stream_response_task(payload)
            try:
                current_response_json = json.loads(final_json_str)
            except:
                pass
        else:
            task_id = await executor.create_response_task(payload)
            current_response_json = await executor.poll_response_result(task_id)
            
            if "output" in current_response_json:
                for item in current_response_json["output"]:
                    if item.get("role") == "assistant":
                        for content_item in item.get("content", []):
                            if content_item.get("type") == "output_text":
                                full_content += content_item.get("text", "")
            
            final_json_str = json.dumps(current_response_json, indent=2, ensure_ascii=False)
        
        if "id" in current_response_json:
            LAST_RESPONSE_ID = current_response_json["id"]
            log_msg("visual_cached_id", id=LAST_RESPONSE_ID)
        
        return comfy_io.NodeOutput(full_content, final_json_str)
