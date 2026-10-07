"""Download a selected pinned model once, with a shared-volume lock and reuse."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

LOGGER = logging.getLogger(__name__)
RUNPOD_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")


def cached_snapshot(model_id, revision):
    """Resolve only our pinned commit; never substitute main or another snapshot."""
    repository = RUNPOD_CACHE_ROOT / ("models--" + model_id.replace("/", "--"))
    snapshot = repository / "snapshots" / revision
    if snapshot.is_dir():
        return snapshot
    if repository.exists():
        raise RuntimeError(
            f"Runpod Model Store has no pinned snapshot for {model_id}@{revision}. "
            "Set the endpoint's cached model revision to the value in models.json, "
            "or use an explicit MODEL_PATH / BASE_MODEL_PATH."
        )
    return None


def storage_paths(root, model_id, entry):
    root = Path(root)
    model = root / model_id.split("/")[-1] / entry["revision"]
    base = (root / entry["base_model"].split("/")[-1] / entry["base_revision"]
            if entry.get("base_model") else root / ".unused-base")
    metadata = root / ".state" / f"{model_id.split('/')[-1]}-{entry['revision']}.json"
    return model, base, metadata


def prepared(root, model_id, entry, *, model_path=None, base_path=None):
    model, base, metadata = storage_paths(root, model_id, entry)
    model = Path(model_path) if model_path is not None else model
    base = Path(base_path) if base_path is not None else base
    try:
        info = json.loads(metadata.read_text(encoding="utf-8"))
        raw = (model / "MODEL_MANIFEST.json").read_bytes()
        if (info.get("model_id") != model_id or info.get("revision") != entry["revision"]
                or hashlib.sha256(raw).hexdigest() != entry["manifest_sha256"]):
            return False
        manifest = json.loads(raw)
        if not all((model / name).is_file() for name in manifest["files_sha256"]):
            return False
        if entry.get("base_model"):
            stored_base = info.get("base", {})
            if (stored_base.get("model_id") != entry["base_model"]
                    or stored_base.get("revision") != entry["base_revision"]):
                return False
            if not all((base / name).is_file() for name in manifest["base"]["files_sha256"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def ensure_stored_model(root, model_id, entry, *, model_path=None, base_path=None):
    """Prepare the volume before loading; the completed copy is reused read-only."""
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("MODEL_ROOT must be an absolute local directory path")
    model, base, metadata = storage_paths(root, model_id, entry)
    model = Path(model_path) if model_path is not None else model
    base = Path(base_path) if base_path is not None else base
    reuse = {"model_path": model_path, "base_path": base_path}
    if prepared(root, model_id, entry, **reuse):
        return model, base
    if not root.parent.is_dir():
        raise RuntimeError(f"Storage parent is missing: {root.parent}. Attach a network volume or set MODEL_ROOT")
    root.mkdir(exist_ok=True)
    metadata.parent.mkdir(exist_ok=True)
    timeout = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "1800"))
    if timeout <= 0:
        raise ValueError("MODEL_DOWNLOAD_TIMEOUT must be positive seconds")
    from filelock import FileLock

    # Every worker preparing this same pinned artifact uses the same file lock.
    with FileLock(str(metadata.with_suffix(".lock")), timeout=timeout):
        if not prepared(root, model_id, entry, **reuse):
            LOGGER.info("Preparing %s@%s on storage at %s", model_id, entry["revision"], root)
            reuse_args = []
            if model_path is not None:
                reuse_args += ["--existing-model", str(model)]
            if base_path is not None:
                reuse_args += ["--existing-base", str(base)]
            subprocess.run(
                [sys.executable, "-u", str(Path(__file__).with_name("bake_model.py")),
                 "--model-id", model_id, "--destination", str(model),
                 "--base-destination", str(base), "--metadata", str(metadata), *reuse_args],
                check=True, timeout=timeout,
                # Only the downloader gets network access to HF. The parent stays offline.
                env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
            )
            if not prepared(root, model_id, entry, **reuse):
                raise RuntimeError("Model download finished without a complete pinned storage package")
    return model, base


def main():
    catalog = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Preload a pinned model on a volume before deploying workers")
    parser.add_argument("--model-id", required=True, choices=catalog)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    model, base = ensure_stored_model(args.root, args.model_id, catalog[args.model_id])
    print(json.dumps({"model_id": args.model_id, "path": str(model),
                      "base_path": str(base) if catalog[args.model_id].get("base_model") else None}))


if __name__ == "__main__":
    main()
