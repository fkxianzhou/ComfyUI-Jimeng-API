from comfy_api.latest import io as comfy_io
from .nodes_shared import JimengClientType
from .models_config import SEEDREAM_4_MODEL_MAP
from .constants import (
    MAX_SEED,
    MIN_SEED,
    MAX_GENERATION_COUNT,
)

RECOMMENDED_SIZES_V3 = [
    "1024x1024 (1:1)",
    "864x1152 (3:4)",
    "1152x864 (4:3)",
    "1280x720 (16:9)",
    "720x1280 (9:16)",
    "832x1248 (2:3)",
    "1248x832 (3:2)",
    "1512x648 (21:9)",
    "Custom",
]

RECOMMENDED_SIZES_V4 = [
    "2K (adaptive)",
    "4K (adaptive)",
    "2048x2048 (1:1)",
    "2304x1728 (4:3)",
    "1728x2304 (3:4)",
    "2848x1600 (16:9)",
    "1600x2848 (9:16)",
    "2496x1664 (3:2)",
    "1664x2496 (2:3)",
    "3136x1344 (21:9)",
    "4096x4096 (1:1)",
    "4704x3520 (4:3)",
    "3520x4704 (3:4)",
    "5504x3040 (16:9)",
    "3040x5504 (9:16)",
    "4992x3328 (3:2)",
    "3328x4992 (2:3)",
    "6240x2656 (21:9)",
    "Custom",
]

RECOMMENDED_SIZES_V5 = [
    "2K (adaptive)",
    "3K (adaptive)",
    "2048x2048 (1:1)",
    "2304x1728 (4:3)",
    "1728x2304 (3:4)",
    "2848x1600 (16:9)",
    "1600x2848 (9:16)",
    "2496x1664 (3:2)",
    "1664x2496 (2:3)",
    "3136x1344 (21:9)",
    "3072x3072 (1:1)",
    "3456x2592 (4:3)",
    "2592x3456 (3:4)",
    "4096x2304 (16:9)",
    "2304x4096 (9:16)",
    "2496x3744 (2:3)",
    "3744x2496 (3:2)",
    "4704x2016 (21:9)",
    "Custom",
]


def get_image_generation_inputs(
    recommended_sizes,
    default_width=1024,
    default_height=1024,
    enable_group_generation=False,
    enable_web_search=False,
):
    """
    获取图片生成相关的输入定义。
    包含预设尺寸选择、自定义宽度和高度、种子、生成数量和水印开关。
    """
    inputs = [
        comfy_io.Combo.Input("size", options=recommended_sizes),
        comfy_io.Int.Input("width", default=default_width, min=1, max=8192),
        comfy_io.Int.Input("height", default=default_height, min=1, max=8192),
        comfy_io.Int.Input("seed", default=0, min=MIN_SEED, max=MAX_SEED),
    ]

    if enable_group_generation:
        inputs.extend(
            [
                comfy_io.Boolean.Input(
                    "enable_group_generation",
                    default=False,
                    tooltip="On=Group, Off=Single",
                ),
                comfy_io.Int.Input("max_images", default=1, min=1, max=15),
            ]
        )

    if enable_web_search:
        inputs.append(
            comfy_io.Boolean.Input(
                "enable_web_search",
                default=False,
                tooltip="Enable internet search capabilities",
            )
        )

    inputs.extend(
        [
            comfy_io.Int.Input(
                "generation_count", default=1, min=1, max=MAX_GENERATION_COUNT
            ),
            comfy_io.Boolean.Input("watermark", default=False),
        ]
    )

    return inputs
