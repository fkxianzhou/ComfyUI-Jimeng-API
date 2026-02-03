from comfy_api.latest import io as comfy_io
from .nodes_shared import JimengClientType, get_text, JimengException
from .models_config import (
    VIDEO_MODEL_MAP,
    VIDEO_UI_OPTIONS,
    VIDEO_1_5_UI_OPTIONS,
    QUERY_TASKS_MODEL_LIST,
    REF_IMG_2_VIDEO_MODEL_ID,
)
from .constants import (
    VIDEO_MAX_SEED,
    VIDEO_DEFAULT_TIMEOUT,
    VIDEO_MIN_TIMEOUT,
    VIDEO_MAX_TIMEOUT,
    VIDEO_FRAME_RATE,
    VIDEO_MIN_FRAMES,
    VIDEO_MAX_FRAMES,
    VIDEO_FRAME_STEP,
    VIDEO_BASE_FRAMES,
    VIDEO_RESOLUTIONS,
    DEFAULT_FILENAME_PREFIX,
)

ASPECT_RATIOS = ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]

def resolve_model_id(model_version: str, image_input=None) -> str:
    """
    根据 UI 选择的模型版本和输入 (文生视频/图生视频) 解析实际的模型 ID。
    """
    if model_version in VIDEO_MODEL_MAP:
        return VIDEO_MODEL_MAP[model_version]

    suffix = "-i2v" if image_input is not None else "-t2v"
    try_key = f"{model_version}{suffix}"
    
    if try_key in VIDEO_MODEL_MAP:
        return VIDEO_MODEL_MAP[try_key]
        
    raise JimengException(f"Model ID not found for selection: {model_version}")

def resolve_query_models(model_version: str) -> list:
    """
    解析查询任务时使用的模型 ID 列表。
    """
    target_models = []
    if model_version == "all":
        target_models = [None]
    elif model_version == "doubao-seedance-1-0-lite":
        if "doubao-seedance-1-0-lite-t2v" in VIDEO_MODEL_MAP:
            target_models.append(VIDEO_MODEL_MAP["doubao-seedance-1-0-lite-t2v"])
        if "doubao-seedance-1-0-lite-i2v" in VIDEO_MODEL_MAP:
            target_models.append(VIDEO_MODEL_MAP["doubao-seedance-1-0-lite-i2v"])
    elif model_version in VIDEO_MODEL_MAP:
        target_models.append(VIDEO_MODEL_MAP[model_version])
    else:
        target_models.append(model_version)
    
    return target_models

def _calculate_duration_and_frames_args(duration: float):
    """
    根据持续时间计算 API 所需的 duration 或 frames 参数。
    如果是整数秒，直接使用 duration；否则根据帧率计算 frames。
    """
    if duration == int(duration):
        return ("duration", int(duration), int(duration))
    else:
        target_frames = duration * VIDEO_FRAME_RATE
        n = round((target_frames - VIDEO_BASE_FRAMES) / VIDEO_FRAME_STEP)
        final_frames = int(max(VIDEO_MIN_FRAMES, min(VIDEO_MAX_FRAMES, VIDEO_BASE_FRAMES + VIDEO_FRAME_STEP * n)))
        return ("frames", final_frames, int(round(final_frames / VIDEO_FRAME_RATE)))

def get_common_video_inputs():
    """
    获取通用的视频生成输入参数定义。
    包含随机种子、生成数量、文件前缀、超时设置等。
    """
    return [
        comfy_io.Boolean.Input(
            "enable_random_seed",
            default=True,
            tooltip="On=Enabled, Off=Disabled",
        ),
        comfy_io.Int.Input("seed", default=0, min=0, max=VIDEO_MAX_SEED),
        comfy_io.Int.Input("generation_count", default=1, min=1),
        comfy_io.String.Input("filename_prefix", default=DEFAULT_FILENAME_PREFIX),
        comfy_io.Boolean.Input("save_last_frame_batch", default=False),
        comfy_io.Int.Input(
            "timeout_seconds", default=VIDEO_DEFAULT_TIMEOUT, min=VIDEO_MIN_TIMEOUT, max=VIDEO_MAX_TIMEOUT
        ),
        comfy_io.Boolean.Input("enable_offline_inference", default=False),
        comfy_io.Boolean.Input("non_blocking", default=False),
    ]

def get_duration_input(default=5.0, min_val=1.2, max_val=12.0, step=0.2, is_int=False):
    """
    获取视频时长输入参数定义。
    """
    if is_int:
        return comfy_io.Int.Input(
            "duration",
            default=int(default),
            min=int(min_val),
            max=int(max_val),
            display_mode=comfy_io.NumberDisplay.number,
        )
    else:
        return comfy_io.Float.Input(
            "duration",
            default=float(default),
            min=float(min_val),
            max=float(max_val),
            step=step,
            display_mode=comfy_io.NumberDisplay.number,
        )

def get_resolution_input(default="720p", support_1080p=True):
    """
    获取分辨率输入参数定义。
    """
    options = ["480p", "720p"]
    if support_1080p:
        options.append("1080p")
    
    if default not in options:
        default = options[-1]
        
    return comfy_io.Combo.Input("resolution", options=options, default=default)

def get_aspect_ratio_input(default="adaptive", include_adaptive=True):
    """
    获取宽高比输入参数定义。
    """
    options = list(ASPECT_RATIOS)
    if not include_adaptive:
        if "adaptive" in options:
            options.remove("adaptive")
        if default == "adaptive":
            default = "16:9" # Fallback default
            
    return comfy_io.Combo.Input("aspect_ratio", options=options, default=default)
