from comfy_api.latest import io as comfy_io
from .nodes_shared import JimengClientType
from .models_config import SEEDREAM_4_MODEL_MAP
from .constants import (
    MAX_SEED,
    MIN_SEED,
    MAX_GENERATION_COUNT,
)

RECOMMENDED_SIZES_V3 = [
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

RECOMMENDED_SIZES_V4 = [
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


def get_image_size_inputs(recommended_sizes, default_width=1024, default_height=1024):
    """
    获取图片尺寸相关的输入定义。
    包含预设尺寸选择、自定义宽度和高度。
    """
    return [
        comfy_io.Combo.Input("size", options=recommended_sizes),
        comfy_io.Int.Input("width", default=default_width, min=1, max=8192),
        comfy_io.Int.Input("height", default=default_height, min=1, max=8192),
    ]


def get_common_generation_inputs():
    """
    获取通用的生成参数输入定义。
    包含种子、生成数量和水印开关。
    """
    return [
        comfy_io.Int.Input("seed", default=0, min=MIN_SEED, max=MAX_SEED),
        comfy_io.Int.Input(
            "generation_count", default=1, min=1, max=MAX_GENERATION_COUNT
        ),
        comfy_io.Boolean.Input("watermark", default=False),
    ]
