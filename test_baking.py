"""Packaging checks only. Real offline inference is checked with smoke_test.py."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app import CATALOG, DecisionWorker, baked_model_info
from bake_model import DEFAULT_MODEL, bake, base_spec, verify_baked_model, verify_files

VEGA_MODEL = "vllm-sr/Decision-2.0-Vega-27B"


class BakingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.content = b"packaging test fixture, not real model weights"
        (self.model / "fixture.bin").write_bytes(self.content)
        self.manifest = {"profile": "qwen-full", "files_sha256": {
            "fixture.bin": hashlib.sha256(self.content).hexdigest()
        }}
        self.write_manifest()

    def write_manifest(self):
        (self.model / "MODEL_MANIFEST.json").write_text(json.dumps(self.manifest))

    def test_intact_package_then_corrupted_weights(self):
        verify_files(self.model)
        (self.model / "fixture.bin").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_files(self.model)

    def test_missing_or_extra_files_fail(self):
        (self.model / "unexpected.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_files(self.model)
        (self.model / "unexpected.bin").unlink()
        (self.model / "fixture.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_files(self.model)

    def test_manifest_path_traversal_rejected(self):
        for name in ("../outside", "/outside", "C:/outside", "folder\\outside", "a/../b", "."):
            with self.subTest(name=name):
                self.manifest["files_sha256"] = {name: "a" * 64}
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    verify_files(self.model)

    def test_missing_identity_does_not_trigger_download(self):
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            baked_model_info(self.root / "missing.json")

    def test_identity_must_match_baked_revision_independently_of_runtime_selection(self):
        model_id = "vllm-sr/Decision-2.0-Lux-9B"
        info = {"model_id": model_id, "revision": CATALOG[model_id]["revision"], "path": str(self.model)}
        metadata = self.root / "baked-model.json"
        metadata.write_text(json.dumps(info))
        with patch.dict("os.environ", {"MODEL_ID": model_id}):
            self.assertEqual(baked_model_info(metadata), info)
        with patch.dict("os.environ", {"MODEL_ID": "vllm-sr/Decision-2.0-Kai-0.6B"}):
            self.assertEqual(baked_model_info(metadata), info)
        info["revision"] = "0" * 40
        metadata.write_text(json.dumps(info))
        with self.assertRaisesRegex(RuntimeError, "identity differs"):
            baked_model_info(metadata)


class AdapterBakingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.base = self.root / "base"
        self.metadata = self.root / "baked-model.json"
        self.entry = CATALOG[VEGA_MODEL]
        self.model_files = {"adapter/weights.bin": b"adapter fixture"}
        self.base_files = {"config.json": b"{}", "weights.bin": b"base fixture"}
        self.manifest = {
            "model_name": VEGA_MODEL.split("/")[-1], "profile": "qwen-adapter",
            "files_sha256": self.hashes(self.model_files),
            "base": {"repo_id": self.entry["base_model"], "revision": self.entry["base_revision"],
                     "files_sha256": self.hashes(self.base_files)},
        }
        self.download = Mock(side_effect=self.fake_download)

    @staticmethod
    def hashes(files):
        return {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}

    def fake_download(self, *, repo_id, revision, local_dir, allow_patterns=None):
        if repo_id == VEGA_MODEL:
            files = {**self.model_files, "MODEL_MANIFEST.json": json.dumps(self.manifest).encode()}
        else:
            files = {**self.base_files, "LICENSE": b"base license fixture"}
        for name, content in files.items():
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def run_bake(self):
        return bake(VEGA_MODEL, self.model, self.base, self.metadata, download=self.download)

    def test_bakes_pinned_adapter_and_base_and_verifies_final_copy(self):
        info = self.run_bake()
        self.assertEqual(self.download.call_count, 2)
        calls = self.download.call_args_list
        self.assertEqual(calls[0].kwargs["revision"], self.entry["revision"])
        self.assertEqual(calls[1].kwargs["repo_id"], self.entry["base_model"])
        self.assertEqual(calls[1].kwargs["revision"], self.entry["base_revision"])
        self.assertEqual(calls[1].kwargs["allow_patterns"], ["LICENSE", "config.json", "weights.bin"])
        self.assertEqual(verify_baked_model(self.metadata), info)
        with patch.dict("os.environ", {"MODEL_ID": VEGA_MODEL}):
            self.assertEqual(baked_model_info(self.metadata), info)

    def test_corrupted_or_missing_base_fails_final_verification(self):
        self.run_bake()
        (self.base / "weights.bin").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_baked_model(self.metadata)
        (self.base / "weights.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_baked_model(self.metadata)

    def test_wrong_upstream_base_is_rejected_before_base_download(self):
        self.manifest["base"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "Base model identity"):
            self.run_bake()
        self.assertEqual(self.download.call_count, 1)
        self.assertFalse(self.metadata.exists())

    def test_missing_license_fails_inventory(self):
        self.run_bake()
        (self.base / "LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_baked_model(self.metadata)

    def test_missing_or_mismatched_base_metadata_fails_before_loading(self):
        info = self.run_bake()
        with patch.dict("os.environ", {"MODEL_ID": VEGA_MODEL}):
            for bad_base in (None, {**info["base"], "revision": "0" * 40}):
                self.metadata.write_text(json.dumps({**info, "base": bad_base}))
                with self.assertRaisesRegex(RuntimeError, "Baked base identity"):
                    baked_model_info(self.metadata)
            self.metadata.write_text(json.dumps(info))
            (self.base / "config.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Baked base files are missing"):
                baked_model_info(self.metadata)

    def test_runtime_passes_local_base_to_custom_loader(self):
        info = self.run_bake()
        loader = Mock()
        with patch("app.selected_model_info", return_value={**info, "source": "baked"}), \
                patch.dict("os.environ", {"DEVICE": "cpu"}), \
                patch.dict("sys.modules", {"torch": SimpleNamespace(),
                                          "transformers": SimpleNamespace(AutoModel=loader)}):
            DecisionWorker()
        self.assertEqual(loader.from_pretrained.call_args.args, (str(self.model.resolve()),))
        self.assertEqual(loader.from_pretrained.call_args.kwargs["base_path"], str(self.base.resolve()))
        self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])

    def test_full_package_has_no_base_dependency(self):
        lux = "vllm-sr/Decision-2.0-Lux-9B"
        self.assertIsNone(base_spec({"profile": "qwen-full"}, CATALOG[lux]))
        self.manifest = {"profile": "qwen-full", "model_name": lux.split("/")[-1],
                         "files_sha256": self.hashes(self.model_files)}
        def full_download(**kwargs):
            kwargs["repo_id"] = VEGA_MODEL
            return self.fake_download(**kwargs)
        download = Mock(side_effect=full_download)
        info = bake(lux, self.model, self.base, self.metadata, download=download)
        self.assertEqual(download.call_count, 1)
        self.assertNotIn("base", info)
        self.assertEqual(verify_baked_model(self.metadata), info)

    def test_existing_base_is_verified_but_not_downloaded_or_modified(self):
        self.fake_download(repo_id=self.entry["base_model"], revision=self.entry["base_revision"],
                           local_dir=self.base)
        extra = self.base / "unused-tokenizer.json"
        extra.write_text("{}")
        info = bake(VEGA_MODEL, self.model, self.root / "unused", self.metadata,
                    download=self.download, existing_base=self.base)
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(self.download.call_args.kwargs["repo_id"], VEGA_MODEL)
        self.assertEqual(info["base"]["path"], str(self.base.resolve()))
        self.assertTrue(extra.is_file())
        (self.base / "weights.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            bake(VEGA_MODEL, self.model, self.root / "unused", self.metadata,
                 download=self.download, existing_base=self.base)

    def test_existing_model_downloads_only_base_and_rejects_modified_manifest(self):
        self.fake_download(repo_id=VEGA_MODEL, revision=self.entry["revision"], local_dir=self.model)
        manifest = self.model / "MODEL_MANIFEST.json"
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps({VEGA_MODEL: {**self.entry,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}))
        with patch("bake_model.CATALOG_PATH", catalog):
            info = bake(VEGA_MODEL, self.root / "unused", self.base, self.metadata,
                        download=self.download, existing_model=self.model)
            self.assertEqual(self.download.call_count, 1)
            self.assertEqual(self.download.call_args.kwargs["repo_id"], self.entry["base_model"])
            self.assertEqual(info["path"], str(self.model.resolve()))
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "manifest differs from the pinned"):
                bake(VEGA_MODEL, self.root / "unused", self.base, self.metadata,
                     download=self.download, existing_model=self.model)
            self.assertEqual(self.download.call_count, 1)


if __name__ == "__main__":
    unittest.main()
