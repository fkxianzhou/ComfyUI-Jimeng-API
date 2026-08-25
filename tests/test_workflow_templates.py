import json
import os
import unittest


PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_DIR = os.path.join(PLUGIN_ROOT, "example_workflows")
EXPECTED_WORKFLOWS = {
    "2.5 Model Updates.json",
    "QuotaSettings.json",
    "Seedance 1.json",
    "Seedance 2.json",
    "Seedream 4.json",
    "Seedream 5.json",
    "VisualUnderstanding.json",
}


def load_workflow(name):
    with open(os.path.join(WORKFLOW_DIR, name), "r", encoding="utf-8") as file:
        return json.load(file)


class WorkflowTemplateTests(unittest.TestCase):
    CURRENT_INPUT_ORDERS = {
        "JimengQuotaSettings": [
            "client", "image_model", "image_limit", "video_model", "video_limit"
        ],
        "JimengSeedance1": [
            "client", "model_version", "prompt", "enable_random_seed", "seed",
            "resolution", "aspect_ratio", "duration", "camerafixed",
            "enable_offline_inference", "generation_count", "filename_prefix",
            "save_last_frame_batch", "non_blocking", "image", "last_frame_image",
        ],
        "JimengSeedance1_5": [
            "client", "model_version", "prompt", "enable_random_seed", "seed",
            "resolution", "aspect_ratio", "auto_duration", "duration",
            "generate_audio", "draft_mode", "reuse_last_draft_task", "draft_task_id",
            "camerafixed", "enable_offline_inference", "generation_count",
            "filename_prefix", "save_last_frame_batch", "non_blocking", "image",
            "last_frame_image",
        ],
        "JimengSeedance2": [
            "client", "model_version", "model_version.prompt",
            "model_version.enable_random_seed", "model_version.seed",
            "model_version.resolution", "model_version.aspect_ratio",
            "model_version.auto_duration", "model_version.duration",
            "model_version.generate_audio", "model_version.enable_web_search",
            "model_version.generation_count", "model_version.filename_prefix",
            "model_version.save_last_frame_batch", "model_version.non_blocking",
            "first_frame_image", "last_frame_image", "ref_images.ref_image_1",
            "ref_videos.ref_video_1", "ref_audios.ref_audio_1",
        ],
        "JimengSeedream4": [
            "client", "model_version", "prompt", "size", "width", "height", "seed",
            "enable_group_generation", "max_images", "generation_count", "thinking",
            "watermark", "images.image_1",
        ],
        "JimengSeedream5": [
            "client", "model_version", "model_version.prompt", "model_version.size",
            "model_version.width", "model_version.height", "model_version.seed",
            "model_version.generation_count", "model_version.thinking",
            "model_version.watermark", "images.image_1",
        ],
        "JimengVisualUnderstanding": [
            "client", "model", "system_prompt", "user_prompt", "detail", "fps",
            "reasoning_mode", "reasoning_effort", "turns", "stream",
            "file_expire_seconds", "seed", "visual_input_1", "visual_input_2",
            "visual_input_3",
        ],
    }

    def test_template_set_is_current_and_has_no_third_party_nodes(self):
        actual = {
            name
            for name in os.listdir(WORKFLOW_DIR)
            if name.lower().endswith(".json")
        }
        self.assertEqual(actual, EXPECTED_WORKFLOWS)

        forbidden_types = {"Fast Groups Bypasser (rgthree)", "ShowText|pysssss", "Note"}
        for name in sorted(actual):
            with self.subTest(workflow=name):
                workflow = load_workflow(name)
                node_types = {node["type"] for node in workflow["nodes"]}
                self.assertTrue(node_types.isdisjoint(forbidden_types))
                self.assertEqual(workflow["version"], 0.4)

    def test_links_and_node_versions_are_consistent(self):
        for name in sorted(EXPECTED_WORKFLOWS):
            with self.subTest(workflow=name):
                workflow = load_workflow(name)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                self.assertEqual(len(nodes), len(workflow["nodes"]))
                self.assertGreaterEqual(workflow["last_node_id"], max(nodes))

                link_ids = set()
                for link in workflow["links"]:
                    link_id, origin_id, origin_slot, target_id, target_slot, link_type = link
                    self.assertNotIn(link_id, link_ids)
                    link_ids.add(link_id)
                    self.assertIn(origin_id, nodes)
                    self.assertIn(target_id, nodes)
                    origin = nodes[origin_id]
                    target = nodes[target_id]
                    self.assertLess(origin_slot, len(origin.get("outputs", [])))
                    self.assertLess(target_slot, len(target.get("inputs", [])))
                    self.assertEqual(target["inputs"][target_slot]["link"], link_id)
                    self.assertEqual(target["inputs"][target_slot]["type"].split(",")[0], link_type)
                    self.assertIn(link_id, origin["outputs"][origin_slot]["links"])

                self.assertGreaterEqual(
                    workflow["last_link_id"], max(link_ids, default=0)
                )
                for node in workflow["nodes"]:
                    if node["type"].startswith("Jimeng"):
                        self.assertEqual(node["properties"]["ver"], "2.5.0")

    def test_dynamic_combo_templates_use_v3_namespaced_inputs(self):
        for name in ("Seedance 2.json", "Seedream 5.json", "2.5 Model Updates.json"):
            workflow = load_workflow(name)
            for node in workflow["nodes"]:
                if node["type"] not in {"JimengSeedance2", "JimengSeedream5"}:
                    continue
                inputs = node["inputs"]
                model_input = next(item for item in inputs if item["name"] == "model_version")
                self.assertEqual(model_input["type"], "COMFY_DYNAMICCOMBO_V3")
                self.assertTrue(
                    any(item["name"].startswith("model_version.") for item in inputs)
                )
                self.assertFalse(any(item["name"] == "prompt" for item in inputs))

    def test_plugin_input_order_matches_current_schema(self):
        found_types = set()
        for name in sorted(EXPECTED_WORKFLOWS):
            workflow = load_workflow(name)
            for node in workflow["nodes"]:
                expected = self.CURRENT_INPUT_ORDERS.get(node["type"])
                if expected is None:
                    continue
                found_types.add(node["type"])
                self.assertEqual(
                    [item["name"] for item in node["inputs"]],
                    expected,
                    msg=f"{name}: {node['type']}",
                )
        self.assertEqual(found_types, set(self.CURRENT_INPUT_ORDERS))

    def test_seedance_templates_cover_2_0_and_2_5(self):
        seedance2 = load_workflow("Seedance 2.json")
        standard_node = next(
            node for node in seedance2["nodes"] if node["type"] == "JimengSeedance2"
        )
        self.assertEqual(standard_node["widgets_values"][0], "doubao-seedance-2-0")

        updates = load_workflow("2.5 Model Updates.json")
        seedance25_node = next(
            node for node in updates["nodes"] if node["type"] == "JimengSeedance2"
        )
        self.assertEqual(seedance25_node["widgets_values"][0], "doubao-seedance-2-5")
        self.assertEqual(seedance25_node["widgets_values"][5], "720p")
        self.assertEqual(seedance25_node["widgets_values"][8], 30)

    def test_templates_do_not_embed_api_keys(self):
        for name in sorted(EXPECTED_WORKFLOWS):
            workflow = load_workflow(name)
            for node in workflow["nodes"]:
                if node["type"] == "JimengAPIClient":
                    self.assertEqual(node["widgets_values"][0], "")


if __name__ == "__main__":
    unittest.main()
