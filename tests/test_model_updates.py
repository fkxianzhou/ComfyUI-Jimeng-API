import asyncio
import importlib
import io
import json
import os
import sys
import types
import unittest
from types import SimpleNamespace


COMFY_ROOT = os.environ.get("COMFYUI_ROOT", r"F:\ComfyUI")
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

# ComfyUI owns the top-level ``utils`` package. Preloading it prevents the
# plugin's nodes/utils.py from shadowing ComfyUI when tests run from this repo.
import utils  # noqa: F401, E402

PACKAGE_NAME = "jimeng_plugin_test"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [PLUGIN_ROOT]
    sys.modules[PACKAGE_NAME] = package

models_config = importlib.import_module(f"{PACKAGE_NAME}.nodes.models_config")
constants = importlib.import_module(f"{PACKAGE_NAME}.nodes.constants")
executor = importlib.import_module(f"{PACKAGE_NAME}.nodes.executor")
nodes_image = importlib.import_module(f"{PACKAGE_NAME}.nodes.nodes_image")
nodes_video = importlib.import_module(f"{PACKAGE_NAME}.nodes.nodes_video")


class ModelConfigurationTests(unittest.TestCase):
    def test_new_model_ids_and_defaults(self):
        self.assertEqual(
            models_config.SEEDREAM_5_MODEL_MAP["doubao-seedream-5.0-pro"],
            "doubao-seedream-5-0-pro-260628",
        )
        self.assertEqual(
            models_config.VIDEO_MODEL_MAP["doubao-seedance-2-0-mini"],
            "doubao-seedance-2-0-mini-260615",
        )
        self.assertEqual(
            next(iter(models_config.VISUAL_MODEL_MAP.items())),
            ("doubao-seed-2-1-pro", "doubao-seed-2-1-pro-260628"),
        )

    def test_seedance_resolution_matrix(self):
        self.assertEqual(
            models_config.VIDEO_2_MODEL_RESOLUTIONS["doubao-seedance-2-0"],
            ["480p", "720p", "1080p", "4k"],
        )
        for model in ("doubao-seedance-2-0-fast", "doubao-seedance-2-0-mini"):
            self.assertEqual(
                models_config.VIDEO_2_MODEL_RESOLUTIONS[model], ["480p", "720p"]
            )
        self.assertEqual(
            nodes_video.validate_seedance2_resolution(
                "doubao-seedance-2-0", "4k"
            ),
            "4k",
        )
        for model in ("doubao-seedance-2-0-fast", "doubao-seedance-2-0-mini"):
            with self.assertRaises(Exception):
                nodes_video.validate_seedance2_resolution(model, "4k")

    def test_dynamic_combo_schema_and_nested_order(self):
        image_combo = nodes_image.JimengSeedream5.define_schema().inputs[1]
        video_combo = nodes_video.JimengSeedance2.define_schema().inputs[1]
        self.assertEqual(image_combo.io_type, "COMFY_DYNAMICCOMBO_V3")
        self.assertEqual(video_combo.io_type, "COMFY_DYNAMICCOMBO_V3")
        self.assertEqual(image_combo.options[0].key, "doubao-seedream-5.0-pro")
        self.assertEqual(video_combo.options[0].key, "doubao-seedance-2-0")

        lite = next(
            option
            for option in image_combo.options
            if option.key == "doubao-seedream-5.0-lite"
        )
        self.assertEqual(
            [item.id for item in lite.inputs],
            [
                "prompt",
                "size",
                "width",
                "height",
                "seed",
                "enable_group_generation",
                "max_images",
                "enable_web_search",
                "generation_count",
                "watermark",
            ],
        )

        pro = image_combo.options[0]
        size_input = next(item for item in pro.inputs if item.id == "size")
        self.assertEqual(size_input.options[:2], ["1K (adaptive)", "2K (adaptive)"])
        self.assertEqual(
            [item.id for item in pro.inputs][-3:],
            ["generation_count", "thinking", "watermark"],
        )

        seedream4_ids = [
            item.id for item in nodes_image.JimengSeedream4.define_schema().inputs
        ]
        self.assertEqual(
            seedream4_ids[1:12],
            [
                "model_version",
                "prompt",
                "size",
                "width",
                "height",
                "seed",
                "enable_group_generation",
                "max_images",
                "generation_count",
                "thinking",
                "watermark",
            ],
        )
        seedream4_thinking = next(
            item
            for item in nodes_image.JimengSeedream4.define_schema().inputs
            if item.id == "thinking"
        )
        self.assertEqual(seedream4_thinking.io_type, "BOOLEAN")
        self.assertTrue(seedream4_thinking.default)

    def test_dynamic_combo_zh_labels_cover_all_nested_inputs(self):
        node_defs_path = os.path.join(PLUGIN_ROOT, "locales", "zh", "nodeDefs.json")
        with open(node_defs_path, "r", encoding="utf-8") as file:
            node_defs = json.load(file)

        combos = {
            "JimengSeedream5": nodes_image.JimengSeedream5.define_schema().inputs[1],
            "JimengSeedance2": nodes_video.JimengSeedance2.define_schema().inputs[1],
        }
        for node_id, combo in combos.items():
            translated_inputs = node_defs[node_id]["inputs"]
            nested_ids = {
                item.id for option in combo.options for item in option.inputs
            }
            for nested_id in nested_ids:
                locale_key = f"model_version_{nested_id}"
                self.assertIn(locale_key, translated_inputs)
                label = translated_inputs[locale_key]["name"]
                self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in label))

        thinking_translation = node_defs["JimengSeedream5"]["inputs"][
            "model_version_thinking"
        ]
        self.assertTrue(
            any("\u4e00" <= char <= "\u9fff" for char in thinking_translation["tooltip"])
        )


class RequestAndEstimationTests(unittest.TestCase):
    def test_64_mib_exact_boundary(self):
        limit = constants.SEEDANCE_REQUEST_MAX_BYTES
        overhead = executor.compact_json_size_bytes({"x": ""})
        accepted = {"x": "a" * (limit - overhead)}
        self.assertEqual(executor.validate_seedance_request_size(accepted), limit)
        rejected = {"x": accepted["x"] + "a"}
        with self.assertRaises(Exception):
            executor.validate_seedance_request_size(rejected)

    def test_seedance_4k_estimation_rules(self):
        class Tasks:
            @staticmethod
            def list(**_kwargs):
                return SimpleNamespace(items=[])

        ark = SimpleNamespace(content_generation=SimpleNamespace(tasks=Tasks()))
        with_reference = asyncio.run(
            executor._get_api_estimated_time_async(
                ark,
                "doubao-seedance-2-0-260615",
                5,
                "4k",
                content=[{"type": "video_url", "video_url": {"url": "x.mp4"}}],
            )
        )
        without_reference = asyncio.run(
            executor._get_api_estimated_time_async(
                ark, "doubao-seedance-2-0-260615", 5, "4k", content=[]
            )
        )
        self.assertEqual(with_reference[0], 5 * 90 + executor.DEFAULT_FALLBACK_BASE)
        self.assertEqual(without_reference[0], 5 * 45 + executor.DEFAULT_FALLBACK_BASE)

    def test_mini_non_blocking_submission_returns_task_json_state(self):
        submitted = []

        class Tasks:
            @staticmethod
            def create(**kwargs):
                submitted.append(kwargs)
                return SimpleNamespace(id="task-mini-1")

            @staticmethod
            def list(**_kwargs):
                return SimpleNamespace(items=[])

        class ProgressServer:
            def send_progress_text(self, *_args, **_kwargs):
                return None

            def send_sync(self, *_args, **_kwargs):
                return None

        old_prompt_server = getattr(executor.PromptServer, "instance", None)
        executor.PromptServer.instance = ProgressServer()
        client = SimpleNamespace(
            ark=SimpleNamespace(
                content_generation=SimpleNamespace(tasks=Tasks())
            )
        )
        try:
            result = asyncio.run(
                executor.JimengGenerationExecutor(client, "test-node").run_batch_tasks(
                    model_name="doubao-seedance-2-0-mini-260615",
                    content=[{"type": "text", "text": "test"}],
                    estimation_duration=5,
                    resolution="720p",
                    generation_count=1,
                    non_blocking=True,
                    non_blocking_cache_dict={},
                )
            )
        finally:
            if old_prompt_server is None:
                delattr(executor.PromptServer, "instance")
            else:
                executor.PromptServer.instance = old_prompt_server

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["task_ids"], ["task-mini-1"])
        self.assertEqual(submitted[0]["model"], "doubao-seedance-2-0-mini-260615")
        self.assertNotIn("service_tier", submitted[0])


class Seedream4PromptOptimizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_seedream_4_0_sends_prompt_optimization(self):
        import torch

        requests = []

        class Client:
            ark = SimpleNamespace()

            @staticmethod
            def check_quota(*_args):
                return None

            @staticmethod
            def update_usage(*_args):
                return None

        async def fake_stream(
            _self,
            _session,
            _ark_client,
            kwargs,
            idx,
            _enable_group_generation,
            _generation_count,
        ):
            requests.append(dict(kwargs))
            return torch.zeros((1, 2, 2, 3)), {"batch_index": idx}

        old_stream = executor.JimengGenerationExecutor.stream_generation_helper
        old_count = nodes_image.get_node_count_in_workflow
        old_hidden = getattr(nodes_image.JimengSeedream4, "hidden", None)
        old_prompt_server = getattr(executor.PromptServer, "instance", None)
        executor.JimengGenerationExecutor.stream_generation_helper = fake_stream
        nodes_image.get_node_count_in_workflow = lambda *_args, **_kwargs: 1
        executor.PromptServer.instance = SimpleNamespace()
        nodes_image.JimengSeedream4.hidden = SimpleNamespace(
            unique_id="test-node", prompt={}
        )

        common = {
            "client": Client(),
            "prompt": "product photo",
            "enable_group_generation": False,
            "max_images": 1,
            "size": "2K (adaptive)",
            "width": 2048,
            "height": 2048,
            "seed": 7,
            "generation_count": 1,
            "watermark": False,
            "thinking": True,
        }
        try:
            await nodes_image.JimengSeedream4.execute(
                model_version="doubao-seedream-4.0", **common
            )
            await nodes_image.JimengSeedream4.execute(
                model_version="doubao-seedream-4.0", **{**common, "thinking": False}
            )
            await nodes_image.JimengSeedream4.execute(
                model_version="doubao-seedream-4.5", **common
            )
        finally:
            executor.JimengGenerationExecutor.stream_generation_helper = old_stream
            nodes_image.get_node_count_in_workflow = old_count
            if old_prompt_server is None:
                delattr(executor.PromptServer, "instance")
            else:
                executor.PromptServer.instance = old_prompt_server
            if old_hidden is None:
                delattr(nodes_image.JimengSeedream4, "hidden")
            else:
                nodes_image.JimengSeedream4.hidden = old_hidden

        self.assertEqual(requests[0]["optimize_prompt_options"].mode, "standard")
        self.assertNotIn("optimize_prompt_options", requests[1])
        self.assertNotIn("optimize_prompt_options", requests[2])


class Seedream5ProTests(unittest.IsolatedAsyncioTestCase):
    def test_size_validation_and_ratio_mapping(self):
        prompt, size = nodes_image.prepare_seedream5_pro_size(
            "product photo", "2848x1600 (16:9)", 2048, 2048
        )
        self.assertEqual(size, "2K")
        self.assertIn("Aspect Ratio: 16:9", prompt)

        declared, size = nodes_image.prepare_seedream5_pro_size(
            "product photo, aspect ratio 16:9", "2848x1600 (16:9)", 2048, 2048
        )
        self.assertEqual(declared.count("16:9"), 1)

        for preset, ratio in (
            ("1600x2848 (9:16)", "9:16"),
            ("3136x1344 (21:9)", "21:9"),
        ):
            mapped_prompt, mapped_size = nodes_image.prepare_seedream5_pro_size(
                "product photo", preset, 2048, 2048
            )
            self.assertEqual(mapped_size, "2K")
            self.assertIn(f"Aspect Ratio: {ratio}", mapped_prompt)

        self.assertEqual(
            nodes_image.prepare_seedream5_pro_size(
                "x", "Custom", 1280, 720
            )[1],
            "1280x720",
        )
        with self.assertRaises(Exception):
            nodes_image.prepare_seedream5_pro_size("x", "Custom", 1279, 720)
        with self.assertRaises(Exception):
            nodes_image.prepare_seedream5_pro_size("x", "Custom", 4112, 256)

    async def test_pro_url_request_and_forbidden_fields(self):
        calls = []

        class Images:
            @staticmethod
            def generate(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    model=kwargs["model"],
                    created=1,
                    data=[SimpleNamespace(url="https://example.invalid/image.png")],
                )

        class Client:
            ark = SimpleNamespace(images=Images())

            @staticmethod
            def check_quota(*_args):
                return None

            @staticmethod
            def update_usage(*_args):
                return None

        async def fake_download(_session, _url):
            import torch

            return torch.zeros((1, 2, 2, 3))

        old_download = nodes_image.download_url_to_image_tensor_async
        old_count = nodes_image.get_node_count_in_workflow
        old_hidden = getattr(nodes_image.JimengSeedream5, "hidden", None)
        old_prompt_server = getattr(executor.PromptServer, "instance", None)
        nodes_image.download_url_to_image_tensor_async = fake_download
        nodes_image.get_node_count_in_workflow = lambda *_args, **_kwargs: 1
        executor.PromptServer.instance = SimpleNamespace()
        nodes_image.JimengSeedream5.hidden = SimpleNamespace(
            unique_id="test-node", prompt={}
        )
        try:
            await nodes_image.JimengSeedream5.execute(
                Client(),
                {
                    "model_version": "doubao-seedream-5.0-pro",
                    "prompt": "product photo",
                    "size": "2K (Adaptive)",
                    "width": 2048,
                    "height": 2048,
                    "seed": 7,
                    "thinking": True,
                    "generation_count": 1,
                    "watermark": True,
                },
            )
        finally:
            nodes_image.download_url_to_image_tensor_async = old_download
            nodes_image.get_node_count_in_workflow = old_count
            if old_prompt_server is None:
                delattr(executor.PromptServer, "instance")
            else:
                executor.PromptServer.instance = old_prompt_server
            if old_hidden is None:
                delattr(nodes_image.JimengSeedream5, "hidden")
            else:
                nodes_image.JimengSeedream5.hidden = old_hidden

        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual(request["response_format"], "url")
        self.assertEqual(request["seed"], 7)
        self.assertTrue(request["watermark"])
        self.assertEqual(request["optimize_prompt_options"].thinking, "enabled")
        self.assertNotIn("tools", request)
        self.assertNotIn("sequential_image_generation", request)

    async def test_reference_requires_thinking(self):
        import torch

        fake_client = SimpleNamespace(ark=SimpleNamespace())
        old_hidden = getattr(nodes_image.JimengSeedream5, "hidden", None)
        nodes_image.JimengSeedream5.hidden = SimpleNamespace(
            unique_id="test-node", prompt={}
        )
        try:
            with self.assertRaises(Exception):
                await nodes_image.JimengSeedream5.execute(
                    fake_client,
                    {
                        "model_version": "doubao-seedream-5.0-pro",
                        "prompt": "edit",
                        "thinking": False,
                    },
                    images=torch.zeros((1, 8, 8, 3)),
                )
        finally:
            if old_hidden is None:
                delattr(nodes_image.JimengSeedream5, "hidden")
            else:
                nodes_image.JimengSeedream5.hidden = old_hidden

    async def test_pro_reference_limit_is_ten(self):
        import torch

        fake_client = SimpleNamespace(ark=SimpleNamespace())
        old_hidden = getattr(nodes_image.JimengSeedream5, "hidden", None)
        nodes_image.JimengSeedream5.hidden = SimpleNamespace(
            unique_id="test-node", prompt={}
        )
        try:
            with self.assertRaises(Exception):
                await nodes_image.JimengSeedream5.execute(
                    fake_client,
                    {
                        "model_version": "doubao-seedream-5.0-pro",
                        "prompt": "edit",
                        "thinking": True,
                    },
                    images=torch.zeros((11, 8, 8, 3)),
                )
        finally:
            if old_hidden is None:
                delattr(nodes_image.JimengSeedream5, "hidden")
            else:
                nodes_image.JimengSeedream5.hidden = old_hidden


class ReferenceVideoTests(unittest.TestCase):
    class FakeVideo:
        def __init__(self, fps=24, video_codec="h264", audio_codec="aac"):
            self.fps = fps
            self.video_codec = video_codec
            self.audio_codec = audio_codec

        def get_container_format(self):
            return "mp4"

        def get_dimensions(self):
            return 1280, 720

        def get_stream_source(self):
            return io.BytesIO(b"video")

        def get_duration(self):
            return 5

        def get_fps(self):
            return self.fps

        def get_video_codec(self):
            return self.video_codec

        def get_audio_codec(self):
            return self.audio_codec

    def test_media_limits_and_known_metadata(self):
        helper = nodes_video.JimengVideoBase()
        self.assertEqual(helper._validate_single_reference_video(self.FakeVideo()), 5)
        with self.assertRaises(Exception):
            helper._validate_single_reference_video(self.FakeVideo(fps=23.9))
        with self.assertRaises(Exception):
            helper._validate_single_reference_video(
                self.FakeVideo(video_codec="vp9")
            )
        with self.assertRaises(Exception):
            helper._validate_single_reference_video(
                self.FakeVideo(audio_codec="opus")
            )

    def test_unknown_codec_metadata_is_not_rejected(self):
        helper = nodes_video.JimengVideoBase()
        self.assertEqual(
            helper._validate_single_reference_video(
                self.FakeVideo(video_codec="", audio_codec="")
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
