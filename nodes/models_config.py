# Jimeng API 模型统一配置文件

# Seedream 4 模型映射
SEEDREAM_4_MODEL_MAP = {
    "doubao-seedream-4.5": "doubao-seedream-4-5-251128",
    "doubao-seedream-4.0": "doubao-seedream-4-0-250828",
}

# Seedream 5 模型映射
SEEDREAM_5_MODEL_MAP = {
    "doubao-seedream-5.0-pro": "doubao-seedream-5-0-pro-260628",
    "doubao-seedream-5.0-lite": "doubao-seedream-5-0-260128",
}

SEEDREAM_5_PRO_UI_MODEL = "doubao-seedream-5.0-pro"
SEEDREAM_5_LITE_UI_MODEL = "doubao-seedream-5.0-lite"

# Seedream 3 模型 ID
SEEDREAM_3_MODELS = {
    "t2i": "doubao-seedream-3-0-t2i-250415",
}

# 视频模型映射
VIDEO_MODEL_MAP = {
    # 标准模型
    "doubao-seedance-1-0-pro": "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-pro-fast": "doubao-seedance-1-0-pro-fast-251015",
    "doubao-seedance-1.5-pro": "doubao-seedance-1-5-pro-251215",
    "doubao-seedance-2-0": "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast": "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-2-0-mini": "doubao-seedance-2-0-mini-260615",
    "doubao-seedance-2-5": "doubao-seedance-2-5-260628",
    # Lite 模型
    "doubao-seedance-1-0-lite-t2v": "doubao-seedance-1-0-lite-t2v-250428",
    "doubao-seedance-1-0-lite-i2v": "doubao-seedance-1-0-lite-i2v-250428",
}

# 视频节点 UI 选项 (1.0)
VIDEO_1_UI_OPTIONS = [
    "doubao-seedance-1-0-pro",
    "doubao-seedance-1-0-pro-fast",
    "doubao-seedance-1-0-lite",
]

# 视频节点 UI 选项 (1.5)
VIDEO_1_5_UI_OPTIONS = [
    "doubao-seedance-1.5-pro"
]

# 视频节点 UI 选项 (2.0)
VIDEO_2_UI_OPTIONS = [
    "doubao-seedance-2-0",
    "doubao-seedance-2-0-fast",
    "doubao-seedance-2-0-mini",
    "doubao-seedance-2-5",
]

VIDEO_2_MODEL_RESOLUTIONS = {
    "doubao-seedance-2-0": ["480p", "720p", "1080p", "4k"],
    "doubao-seedance-2-0-fast": ["480p", "720p"],
    "doubao-seedance-2-0-mini": ["480p", "720p"],
    "doubao-seedance-2-5": ["480p", "720p"],
}

VIDEO_2_MODEL_MAX_DURATIONS = {
    "doubao-seedance-2-0": 15,
    "doubao-seedance-2-0-fast": 15,
    "doubao-seedance-2-0-mini": 15,
    "doubao-seedance-2-5": 30,
}

VIDEO_2_MODEL_REFERENCE_LIMITS = {
    "doubao-seedance-2-0": {"images": 9, "videos": 3, "audios": 3},
    "doubao-seedance-2-0-fast": {"images": 9, "videos": 3, "audios": 3},
    "doubao-seedance-2-0-mini": {"images": 9, "videos": 3, "audios": 3},
    "doubao-seedance-2-5": {"images": 30, "videos": 10, "audios": 10},
}

# 任务查询节点专用的模型列表
QUERY_TASKS_MODEL_LIST = ["all"] + VIDEO_1_UI_OPTIONS + VIDEO_1_5_UI_OPTIONS + VIDEO_2_UI_OPTIONS

# 参考图生视频默认 ID
REF_IMG_2_VIDEO_MODEL_ID = VIDEO_MODEL_MAP["doubao-seedance-1-0-lite-i2v"]

# 视觉理解模型列表
VISUAL_MODEL_MAP = {
    "doubao-seed-2-1-pro": "doubao-seed-2-1-pro-260628",
    "doubao-seed-2-0-pro": "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-0-lite": "doubao-seed-2-0-lite-260215",
    "doubao-seed-2-0-mini": "doubao-seed-2-0-mini-260215",
}
VISUAL_UI_OPTIONS = list(VISUAL_MODEL_MAP.keys())
