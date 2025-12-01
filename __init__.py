from .nodes_shared import NODE_CLASS_MAPPINGS as shared_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as shared_display_mappings
from .nodes_image import NODE_CLASS_MAPPINGS as image_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as image_display_mappings
from .nodes_video import NODE_CLASS_MAPPINGS as video_class_mappings, NODE_DISPLAY_NAME_MAPPINGS as video_display_mappings

NODE_CLASS_MAPPINGS = {
    **shared_class_mappings,
    **image_class_mappings,
    **video_class_mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **shared_display_mappings,
    **image_display_mappings,
    **video_display_mappings
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]