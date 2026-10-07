"""Verify Lux/Vega packaging and external model selection without downloads."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app import CATALOG, selected_model_info
from bake_model import BAKED_MODELS, DEFAULT_MODEL, bake_bundled, verify_baked_model

VEGA = "vllm-sr/Decision-2.0-Vega-27B"


class ModelCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "baked"
        self.metadata = self.root / "baked-model.json"
        self.catalog_path = self.root / "models.json"
        self.catalog = {key: dict(value) for key, value in CATALOG.items()}
        self.packages = {}
        self.base_files = {"config.json": b"{}", "weights.bin": b"Qwen base fixture"}
        for model_id, entry in self.catalog.items():
            files = {"config.json": b"{}", "weights.bin": model_id.encode()}
            manifest = {"model_name": model_id.split("/")[-1], "profile": "qwen-full",
                        "files_sha256": self.hashes(files)}
            if entry.get("base_model"):
                manifest.update(profile="qwen-adapter", base={
                    "repo_id": entry["base_model"], "revision": entry["base_revision"],
                    "files_sha256": self.hashes(self.base_files)})
            raw = json.dumps(manifest).encode()
            self.packages[model_id] = {**files, "MODEL_MANIFEST.json": raw}
            entry["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        self.catalog_path.write_text(json.dumps(self.catalog))
        self.addCleanup(patch.stopall)
        patch("bake_model.CATALOG_PATH", self.catalog_path).start()
        patch("app.CATALOG", self.catalog).start()
        patch("model_storage.RUNPOD_CACHE_ROOT", self.root / "cache").start()
        patch.dict(os.environ, {}, clear=True).start()
        self.download = Mock(side_effect=self.fake_download)

    @staticmethod
    def hashes(files):
        return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

    @staticmethod
    def write_files(root, files):
        root.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def fake_download(self, *, repo_id, revision, local_dir):
        self.assertEqual(revision, self.catalog[repo_id]["revision"])
        self.write_files(Path(local_dir), self.packages[repo_id])

    def bake(self):
        return bake_bundled(self.models, self.metadata, download=self.download)

    def select(self, model_id=None, **env):
        if model_id is not None:
            env["MODEL_ID"] = model_id
        with patch.dict(os.environ, env):
            return selected_model_info(self.metadata)

    def test_bakes_only_lux_and_vega_and_no_qwen_base(self):
        info = self.bake()
        expected = {"vllm-sr/Decision-2.0-Lux-9B", "vllm-sr/Decision-2.0-Vega-27B"}
        self.assertEqual(set(info["models"]), expected)
        self.assertEqual({call.kwargs["repo_id"] for call in self.download.call_args_list}, expected)
        self.assertEqual(self.download.call_count, 2)
        self.assertEqual(info["default_model"], DEFAULT_MODEL)
        self.assertEqual(info["models"][VEGA]["base"], {
            "model_id": self.catalog[VEGA]["base_model"],
            "revision": self.catalog[VEGA]["base_revision"], "external": True})
        self.assertFalse((self.models / ".unused-base").exists())
        self.assertEqual(verify_baked_model(self.metadata), info)

    def test_lux_selects_baked_files_without_storage(self):
        info = self.bake()
        self.assertEqual(self.select()["model_id"], DEFAULT_MODEL)
        with patch("app.cached_snapshot") as cache, patch("app.ensure_stored_model") as download:
            selected = self.select(DEFAULT_MODEL)
            self.assertEqual(selected["source"], "baked")
            self.assertEqual(selected["path"], info["models"][DEFAULT_MODEL]["path"])
            cache.assert_not_called()
            download.assert_not_called()

    def test_smaller_models_load_from_cache_when_not_baked(self):
        self.bake()
        with patch("app.ensure_stored_model") as download:
            for model_id in set(self.catalog) - set(BAKED_MODELS):
                with self.subTest(model=model_id):
                    model = (self.root / "cache" / ("models--" + model_id.replace("/", "--"))
                             / "snapshots" / self.catalog[model_id]["revision"])
                    self.write_files(model, self.packages[model_id])
                    selected = self.select(model_id)
                    self.assertEqual(selected["path"], str(model))
                    self.assertEqual(selected["source"], "model-store")
        download.assert_not_called()

    def test_uncached_smaller_model_downloads_to_storage(self):
        self.bake()
        model_id = "vllm-sr/Decision-2.0-Kai-0.6B"
        model = self.root / "storage" / "kai"
        self.write_files(model, self.packages[model_id])
        with patch("app.ensure_stored_model", return_value=(model, None)) as download:
            selected = self.select(model_id, MODEL_ROOT=str(self.root / "storage"))
        self.assertEqual(download.call_args.args[1], model_id)
        self.assertEqual(download.call_args.kwargs, {"model_path": None, "base_path": None})
        self.assertEqual(selected["source"], "storage")
        self.assertEqual(selected["path"], str(model))

    def test_vega_uses_baked_adapter_and_cached_qwen_without_downloading(self):
        info = self.bake()
        entry = self.catalog[VEGA]
        base = (self.root / "cache" / ("models--" + entry["base_model"].replace("/", "--"))
                / "snapshots" / entry["base_revision"])
        self.write_files(base, self.base_files)
        with patch("app.ensure_stored_model") as download:
            selected = self.select(VEGA)
        download.assert_not_called()
        self.assertEqual(selected["source"], "baked")
        self.assertEqual(selected["path"], info["models"][VEGA]["path"])
        self.assertEqual(selected["base"]["path"], str(base))
        self.assertNotIn("external", selected["base"])

    def test_vega_uses_explicit_base_even_if_cache_has_another_revision(self):
        self.bake()
        entry = self.catalog[VEGA]
        wrong = self.root / "cache" / ("models--" + entry["base_model"].replace("/", "--"))
        wrong.mkdir(parents=True)
        base = self.root / "mounted-qwen"
        self.write_files(base, self.base_files)
        with patch("app.ensure_stored_model") as download:
            self.assertEqual(self.select(VEGA, BASE_MODEL_PATH=str(base))["base"]["path"], str(base))
        download.assert_not_called()

    def test_vega_passes_baked_adapter_to_downloader_for_base_only(self):
        info = self.bake()
        adapter = Path(info["models"][VEGA]["path"])
        base = self.root / "models" / "qwen"
        self.write_files(base, self.base_files)
        with patch("app.ensure_stored_model", return_value=(adapter, base)) as download:
            selected = self.select(VEGA, MODEL_ROOT=str(self.root / "models"))
        self.assertEqual(download.call_args.kwargs, {"model_path": adapter, "base_path": None})
        self.assertEqual(selected["base"]["path"], str(base))

    def test_vega_without_cache_or_storage_fails_with_setup_guidance(self):
        self.bake()
        with self.assertRaisesRegex(RuntimeError, "Attach a network volume"):
            self.select(VEGA, MODEL_ROOT=str(self.root / "unmounted" / "models"))

    def test_missing_baked_package_never_downloads_a_replacement(self):
        info = self.bake()
        model_id = DEFAULT_MODEL
        (Path(info["models"][model_id]["path"]) / "MODEL_MANIFEST.json").unlink()
        with patch("app.ensure_stored_model") as download:
            with self.assertRaisesRegex(RuntimeError, "Baked model files are missing"):
                self.select(model_id)
        download.assert_not_called()

    def test_collection_verification_rejects_missing_entry_and_corrupt_weights(self):
        info = self.bake()
        incomplete = {**info, "models": {key: value for key, value in info["models"].items() if key != VEGA}}
        self.metadata.write_text(json.dumps(incomplete))
        with self.assertRaisesRegex(ValueError, "collection differs"):
            verify_baked_model(self.metadata)
        self.metadata.write_text(json.dumps(info))
        (Path(info["models"][VEGA]["path"]) / "weights.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_baked_model(self.metadata)

    def test_external_base_revision_is_bound_to_catalog(self):
        info = self.bake()
        info["models"][VEGA]["base"]["revision"] = "0" * 40
        self.metadata.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "Baked base identity"):
            verify_baked_model(self.metadata)
        with self.assertRaisesRegex(RuntimeError, "Baked base identity"):
            self.select(VEGA)


if __name__ == "__main__":
    unittest.main()
