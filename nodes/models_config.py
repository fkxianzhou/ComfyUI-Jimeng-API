# Jimeng API 模型统一配置文件

# Seedream 4 模型映射
SEEDREAM_4_MODEL_MAP = {
    "doubao-seedream-4.5": "doubao-seedream-4-5-251128",
    "doubao-seedream-4.0": "doubao-seedream-4-0-250828",
}

# Seedream 5 模型映射
SEEDREAM_5_MODEL_MAP = {
    "doubao-seedream-5.0-lite": "doubao-seedream-5-0-260128",
}

# Seedream 3 模型 ID
SEEDREAM_3_MODELS = {
    "t2i": "doubao-seedream-3-0-t2i-250415",
    "i2i": "doubao-seededit-3-0-i2i-250628",
}

# 视频模型映射
VIDEO_MODEL_MAP = {
    # 标准模型
    "doubao-seedance-1-0-pro": "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-pro-fast": "doubao-seedance-1-0-pro-fast-251015",
    "doubao-seedance-1.5-pro": "doubao-seedance-1-5-pro-251215",
    "doubao-seedance-2-0":"doubao-seedance-2-0-260128",
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
    "doubao-seedance-2-0"
]

# 任务查询节点专用的模型列表
QUERY_TASKS_MODEL_LIST = ["all"] + VIDEO_1_UI_OPTIONS + VIDEO_1_5_UI_OPTIONS + VIDEO_2_UI_OPTIONS

# 参考图生视频默认 ID
REF_IMG_2_VIDEO_MODEL_ID = VIDEO_MODEL_MAP["doubao-seedance-1-0-lite-i2v"]