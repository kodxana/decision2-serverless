"""Exercise first-use download/reuse and failures without real network requests."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_storage import ensure_stored_model, storage_paths


class ModelStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "models"
        self.model_id = "vllm-sr/Decision-2.0-Vega-27B"
        self.raw = json.dumps({"files_sha256": {"weights.bin": "unused-fixture-hash"},
                               "base": {"files_sha256": {"base.bin": "unused-fixture-hash"}}}).encode()
        self.entry = {"revision": "a" * 40, "base_model": "Qwen/Qwen3.8-27B",
                      "base_revision": "b" * 40, "manifest_sha256": hashlib.sha256(self.raw).hexdigest()}
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, clear=True).start()
        self.lock = Mock(side_effect=lambda *args, **kwargs: nullcontext())
        patch.dict("sys.modules", {"filelock": SimpleNamespace(FileLock=self.lock)}).start()

    def finish_download(self, *args, **kwargs):
        model, base, metadata = storage_paths(self.root, self.model_id, self.entry)
        model.mkdir(parents=True, exist_ok=True)
        base.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        (model / "MODEL_MANIFEST.json").write_bytes(self.raw)
        (model / "weights.bin").write_bytes(b"model fixture")
        (base / "base.bin").write_bytes(b"base fixture")
        metadata.write_text(json.dumps({"model_id": self.model_id, "revision": self.entry["revision"],
            "base": {"model_id": self.entry["base_model"], "revision": self.entry["base_revision"]}}))

    def ensure(self):
        return ensure_stored_model(self.root, self.model_id, self.entry)

    def test_first_use_downloads_then_reuses_without_network_or_lock_writes(self):
        with patch("model_storage.subprocess.run", side_effect=self.finish_download) as download:
            paths = self.ensure()
            self.assertEqual(self.ensure(), paths)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(self.lock.call_count, 1)
            self.assertIn(self.model_id, download.call_args.args[0])
            self.assertEqual(download.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_missing_file_triggers_resume_instead_of_claiming_cache_ready(self):
        self.finish_download()
        model, _ = self.ensure()
        (model / "weights.bin").unlink()
        with patch("model_storage.subprocess.run", side_effect=self.finish_download) as download:
            self.ensure()
        download.assert_called_once()

    def test_downloader_failure_propagates_without_returning_baked_fallback(self):
        with patch("model_storage.subprocess.run", side_effect=subprocess.CalledProcessError(1, "download")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.ensure()

    def test_incomplete_download_does_not_become_ready(self):
        with patch("model_storage.subprocess.run"):
            with self.assertRaisesRegex(RuntimeError, "without a complete pinned"):
                self.ensure()

    def test_waiting_worker_rechecks_after_acquiring_shared_lock(self):
        def other_worker_finishes(*args, **kwargs):
            self.finish_download()
            return nullcontext()
        self.lock.side_effect = other_worker_finishes
        with patch("model_storage.subprocess.run") as download:
            self.ensure()
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
