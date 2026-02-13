import threading
import time
import logging
from .nodes_shared import JimengException, get_text, JimengClients
from .constants import VIDEO_FRAME_RATE, VIDEO_RESOLUTION_PIXELS

logger = logging.getLogger("JimengAI")

class QuotaManager:
    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        self._quotas = {}
        self._running_counts = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_quota(self, api_key: str, model: str, limit: int, quota_type: str):
        """
        设置配额。
        """
        from .nodes_shared import log_msg
        
        with self._lock:
            if api_key not in self._quotas:
                self._quotas[api_key] = {}
            
            if limit <= 0:
                if model in self._quotas[api_key]:
                    del self._quotas[api_key][model]
            else:
                self._quotas[api_key][model] = {
                    "limit": limit,
                    "used": 0,
                    "type": quota_type
                }
            
            log_msg("quota_set_log", model=model, limit=limit, type=quota_type)

    def get_status(self, api_key: str) -> str:
        """
        获取当前配额状态字符串。
        """
        with self._lock:
            if api_key not in self._quotas or not self._quotas[api_key]:
                return "No active quotas."
            
            lines = ["Updated Quota:"]
            for model, data in self._quotas[api_key].items():
                limit = data["limit"]
                used = data["used"]
                q_type = data["type"]
                unit = "images" if q_type == "image" else "tokens"
                lines.append(f"{model}: {used}/{limit} {unit}")
            
            return "\n".join(lines)

    def check_quota(self, api_key: str, model: str, estimated_cost: int):
        """
        检查配额是否足够。
        """
        with self._lock:
            if api_key not in self._quotas:
                return
            
            if model not in self._quotas[api_key]:
                return

            data = self._quotas[api_key][model]
            limit = data["limit"]
            used = data["used"]

            if used + estimated_cost > limit:
                del self._quotas[api_key][model]
                
                msg = get_text("quota_exceeded").format(
                    model=model,
                    limit=limit,
                    used=used,
                    estimated=estimated_cost
                )
                raise JimengException(msg)

    def update_usage(self, api_key: str, model: str, actual_cost: int):
        """
        更新实际用量。
        """
        from .nodes_shared import log_msg

        with self._lock:
            if api_key not in self._quotas:
                return
            
            if model not in self._quotas[api_key]:
                return

            self._quotas[api_key][model]["used"] += actual_cost
            log_msg("quota_update_log", model=model, cost=actual_cost, total=self._quotas[api_key][model]['used'])

    def estimate_video_tokens(self, model: str, width: int, height: int, duration: float, fps: float, has_audio: bool = False, is_draft: bool = False) -> int:
        """
        估算视频 Token 消耗。
        """
        # 基础公式: (宽 * 高 * 帧率 * 时长) / 1024
        base_tokens = (width * height * fps * duration) / 1024.0
        
        if is_draft:
            if "seedance-1.5-pro" in model or "seedance-1-5-pro" in model.replace(".", "-"):
                coeff = 0.6 if has_audio else 0.7
                return int(base_tokens * coeff)
            
            return int(base_tokens)
            
        return int(base_tokens)


from comfy_api.latest import io as comfy_io
from .nodes_shared import GLOBAL_CATEGORY, JimengClientType
from .models_config import SEEDREAM_4_MODEL_MAP, VIDEO_MODEL_MAP, SEEDREAM_3_MODELS

class JimengQuotaSettings(comfy_io.ComfyNode):
    """
    Jimeng 配额设置节点。
    用于设置图像和视频生成的配额限制。
    """
    
    IMAGE_MODELS = ["None"] + list(SEEDREAM_4_MODEL_MAP.keys()) + ["doubao-seedream-3.0-t2i", "doubao-seedream-3.0-i2i"]
    VIDEO_MODELS = ["None"] + list(VIDEO_MODEL_MAP.keys())

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="JimengQuotaSettings",
            display_name="Jimeng Quota Settings",
            category=GLOBAL_CATEGORY,
            inputs=[
                JimengClientType.Input("client"),
                comfy_io.Combo.Input("image_model", options=cls.IMAGE_MODELS, default="None"),
                comfy_io.Int.Input("image_limit", default=0, min=0, max=2147483647, tooltip="0 to disable"),
                comfy_io.Combo.Input("video_model", options=cls.VIDEO_MODELS, default="None"),
                comfy_io.Int.Input("video_limit", default=0, min=0, max=2147483647, tooltip="0 to disable"),
            ],
            outputs=[
                comfy_io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        client,
        image_model,
        image_limit,
        video_model,
        video_limit,
    ) -> comfy_io.NodeOutput:
        
        api_key = getattr(client, "api_key", None)
        
        if not api_key:
            return comfy_io.NodeOutput("Error: Client has no API Key bound.")

        manager = QuotaManager.instance()
        
        if image_model != "None":
            real_image_model = SEEDREAM_4_MODEL_MAP.get(image_model, image_model)
            
            if image_model == "doubao-seedream-3.0-t2i":
                real_image_model = SEEDREAM_3_MODELS["t2i"]
            elif image_model == "doubao-seedream-3.0-i2i":
                real_image_model = SEEDREAM_3_MODELS["i2i"]
                
            manager.set_quota(api_key, real_image_model, image_limit, "image")
            
        if video_model != "None":
            real_video_model = VIDEO_MODEL_MAP.get(video_model, video_model)
            manager.set_quota(api_key, real_video_model, video_limit, "video")

        status = manager.get_status(api_key)
        return comfy_io.NodeOutput(status)
