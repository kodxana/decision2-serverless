"""Storage-selection checks with tiny files; no model inference or downloads."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app import CATALOG, DecisionWorker, selected_model_info
from bake_model import DEFAULT_MODEL
from model_storage import storage_paths

VEGA = "vllm-sr/Decision-2.0-Vega-27B"


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = self.root / "models"
        self.baked = self.root / "baked"
        self.baked.mkdir()
        (self.baked / "MODEL_MANIFEST.json").write_text("{}")
        self.metadata = self.root / "baked-model.json"
        self.metadata.write_text(json.dumps({"model_id": DEFAULT_MODEL,
            "revision": CATALOG[DEFAULT_MODEL]["revision"], "path": str(self.baked)}))
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        self.catalog = {key: dict(value) for key, value in CATALOG.items()}
        patch("app.CATALOG", self.catalog).start()

    def write_files(self, directory, files):
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

    def prepare(self, model_id, model_path=None, base_path=None):
        entry = self.catalog[model_id]
        default_model, default_base, metadata = storage_paths(self.storage, model_id, entry)
        model_path = model_path or default_model
        files = {"config.json": b"{}", "modeling_fixture.py": b"# trusted fixture\n", "weights.bin": b"weights"}
        manifest = {"model_name": model_id.split("/")[-1], "profile": "qwen-full",
                    "files_sha256": self.write_files(model_path, files)}
        if entry.get("base_model"):
            base_path = base_path or default_base
            base_files = {"config.json": b"{}", "base.bin": b"base weights"}
            manifest.update(profile="qwen-adapter", base={
                "repo_id": entry["base_model"], "revision": entry["base_revision"],
                "files_sha256": self.write_files(base_path, base_files)})
        raw = json.dumps(manifest).encode()
        (model_path / "MODEL_MANIFEST.json").write_bytes(raw)
        entry["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        info = {"model_id": model_id, "revision": entry["revision"]}
        if entry.get("base_model"):
            info["base"] = {"model_id": entry["base_model"], "revision": entry["base_revision"]}
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(info))
        return model_path, base_path

    def select(self, **env):
        with patch.dict(os.environ, env):
            return selected_model_info(self.metadata)

    def test_default_is_baked_lux_without_a_volume(self):
        info = self.select()
        self.assertEqual(DEFAULT_MODEL, "vllm-sr/Decision-2.0-Lux-9B")
        self.assertEqual(info["model_id"], DEFAULT_MODEL)
        self.assertEqual(info["source"], "baked")
        self.assertEqual(info["path"], str(self.baked))

    def test_model_id_selects_standard_volume_layout_and_pinned_base(self):
        model, base = self.prepare(VEGA)
        info = self.select(MODEL_ID=VEGA, MODEL_ROOT=str(self.storage))
        self.assertEqual(info["model_id"], VEGA)
        self.assertEqual(info["path"], str(model))
        self.assertEqual(info["base"]["path"], str(base))
        self.assertEqual(info["base"]["revision"], CATALOG[VEGA]["base_revision"])

    def test_custom_paths_override_standard_layout(self):
        model, base = self.prepare(VEGA, self.root / "custom-adapter", self.root / "custom-base")
        info = self.select(MODEL_ID=VEGA, MODEL_PATH=str(model), BASE_MODEL_PATH=str(base))
        self.assertEqual(info["path"], str(model))
        self.assertEqual(info["base"]["path"], str(base))

    def test_explicit_lux_path_overrides_baked_lux(self):
        model, _ = self.prepare(DEFAULT_MODEL)
        info = self.select(MODEL_PATH=str(model))
        self.assertEqual(info["model_id"], DEFAULT_MODEL)
        self.assertEqual(info["source"], "storage")
        self.assertNotIn("base", info)

    def test_missing_override_does_not_fall_back_to_lux(self):
        with self.assertRaisesRegex(RuntimeError, "Attach a network volume"):
            self.select(MODEL_ID=VEGA, MODEL_ROOT=str(self.root / "not-mounted" / "models"))

    def test_missing_base_does_not_download(self):
        model, base = self.prepare(VEGA)
        with self.assertRaisesRegex(RuntimeError, "BASE_MODEL_PATH directory is missing"):
            self.select(MODEL_ID=VEGA, MODEL_PATH=str(model), BASE_MODEL_PATH=str(self.root / "absent"))
        (base / "base.bin").unlink()
        with self.assertRaisesRegex(RuntimeError, "Model file is missing"):
            self.select(MODEL_ID=VEGA, MODEL_PATH=str(model), BASE_MODEL_PATH=str(base))

    def test_wrong_or_modified_manifest_is_rejected(self):
        model, _ = self.prepare(VEGA)
        manifest = model / "MODEL_MANIFEST.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest differs from the pinned"):
            self.select(MODEL_ID=VEGA, MODEL_PATH=str(model))

    def test_modified_code_is_rejected_before_execution(self):
        model, _ = self.prepare(VEGA)
        (model / "modeling_fixture.py").write_text("raise RuntimeError('untrusted')")
        with self.assertRaisesRegex(ValueError, "code/config checksum mismatch"):
            self.select(MODEL_ID=VEGA, MODEL_ROOT=str(self.storage))

    def test_missing_weight_is_rejected(self):
        model, _ = self.prepare(VEGA)
        (model / "weights.bin").unlink()
        with self.assertRaisesRegex(RuntimeError, "Model file is missing"):
            self.select(MODEL_ID=VEGA, MODEL_PATH=str(model))

    def test_invalid_selection_or_relative_paths_are_rejected(self):
        for env, message in (
            ({"MODEL_ID": "unrelated/chat-model"}, "MODEL_ID must name"),
            ({"MODEL_PATH": ""}, "absolute local directory"),
            ({"MODEL_PATH": "https://huggingface.co/model"}, "absolute local directory"),
            ({"MODEL_ID": VEGA, "MODEL_ROOT": "relative"}, "MODEL_ROOT must be"),
            ({"BASE_MODEL_PATH": str(self.root)}, "only used by models with an external base"),
        ):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, message):
                self.select(**env)

    def test_worker_loads_selected_storage_paths_offline(self):
        model, base = self.prepare(VEGA)
        loader = Mock()
        info = self.select(MODEL_ID=VEGA, MODEL_ROOT=str(self.storage))
        with patch("app.selected_model_info", return_value=info), \
                patch.dict(os.environ, {"DEVICE": "cpu"}), \
                patch.dict("sys.modules", {"torch": SimpleNamespace(),
                                          "transformers": SimpleNamespace(AutoModel=loader)}):
            worker = DecisionWorker()
        self.assertEqual(worker.model_id, VEGA)
        self.assertEqual(loader.from_pretrained.call_args.args, (str(model),))
        self.assertEqual(loader.from_pretrained.call_args.kwargs["base_path"], str(base))
        self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
