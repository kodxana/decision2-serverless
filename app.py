"""Runpod adapter for the pinned Decision 2.0 System One API."""

import hashlib
import json
import logging
import os
from pathlib import Path

from bake_model import base_spec, validate_file_map
from model_storage import cached_snapshot, ensure_stored_model, storage_paths

LOGGER = logging.getLogger(__name__)
CATALOG = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))


def baked_model_info(metadata_path=None):
    """Fail before model loading if the baked artifact is absent or inconsistent."""
    path = Path(metadata_path) if metadata_path else Path(__file__).with_name("baked-model.json")
    if not path.is_file():
        raise RuntimeError("Baked model metadata is missing; build this image with bake_model.py")
    info = json.loads(path.read_text(encoding="utf-8"))
    model_id = info.get("model_id")
    if model_id not in CATALOG or info.get("revision") != CATALOG[model_id]["revision"]:
        raise RuntimeError("Baked model identity differs from models.json; rebuild the image")
    model_path = Path(info["path"])
    if not model_path.is_dir() or not (model_path / "MODEL_MANIFEST.json").is_file():
        raise RuntimeError("Baked model files are missing; do not mount over /opt/models")
    entry = CATALOG[model_id]
    base = info.get("base")
    if entry.get("base_model"):
        if (not isinstance(base, dict) or base.get("model_id") != entry["base_model"]
                or base.get("revision") != entry["base_revision"]):
            raise RuntimeError("Baked base identity is missing or differs from models.json")
        if not base.get("path") or not (Path(base["path"]) / "config.json").is_file():
            raise RuntimeError("Baked base files are missing; do not mount over /opt/models")
    elif base:
        raise RuntimeError("Unexpected baked base for a self-contained model")
    return info


def local_model_path(value, name):
    """Model overrides are existing local directories, never remote identifiers."""
    path = Path(value)
    if not value or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute local directory path")
    if not path.is_dir():
        raise RuntimeError(f"{name} directory is missing: {path}. Preload the model on the attached volume")
    return path


def check_stored_files(root, files):
    """Check completeness and code/config hashes before executing the native loader.

    The native runtime verifies all weight hashes when loading. Avoid hashing the
    large weights twice over network storage on every cold start.
    """
    validate_file_map(files)
    for name, digest in files.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Model file is missing: {path}. Finish preloading the volume")
        if path.suffix in (".py", ".json"):
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"Stored model code/config checksum mismatch: {path}")


def selected_model_info(metadata_path=None):
    """Select baked weights, explicit paths, or pinned cache/storage snapshots."""
    baked = baked_model_info(metadata_path)
    model_id = os.getenv("MODEL_ID", baked["model_id"])
    if model_id not in CATALOG:
        raise ValueError("MODEL_ID must name a Decision 2.0 model from models.json")
    entry = CATALOG[model_id]
    override = os.getenv("MODEL_PATH")
    base_override = os.getenv("BASE_MODEL_PATH")
    if base_override is not None and not entry.get("base_model"):
        raise ValueError("BASE_MODEL_PATH is only used by models with an external base, such as Vega")
    if override is None and model_id == baked["model_id"] and base_override is None:
        return {**baked, "source": "baked"}

    root = Path(os.getenv("MODEL_ROOT", "/runpod-volume/models"))
    base_path = None
    if base_override is not None:
        base_path = local_model_path(base_override, "BASE_MODEL_PATH")
    elif entry.get("base_model"):
        if model_id == baked["model_id"] and baked.get("base"):
            base_path = Path(baked["base"]["path"])
        else:
            base_path = cached_snapshot(entry["base_model"], entry["base_revision"])

    source = "storage"
    if override is not None:
        model_path = local_model_path(override, "MODEL_PATH")
    elif model_id == baked["model_id"]:
        model_path = local_model_path(baked["path"], "MODEL_PATH")
        source = "baked"
    else:
        model_path = cached_snapshot(model_id, entry["revision"])
        if model_path is not None:
            source = "model-store"
        if model_path is None or (entry.get("base_model") and base_path is None):
            # Download only the missing component; mounted cache files stay untouched.
            model_path, stored_base = ensure_stored_model(
                root, model_id, entry, model_path=model_path, base_path=base_path,
            )
            if entry.get("base_model"):
                base_path = stored_base
    manifest_path = model_path / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Model manifest is missing: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != entry["manifest_sha256"]:
        raise ValueError(f"Model manifest differs from the pinned {model_id}@{entry['revision']}")
    manifest = json.loads(raw_manifest)
    if manifest.get("model_name") != model_id.split("/")[-1]:
        raise ValueError("Stored model manifest names a different model")
    base = base_spec(manifest, entry)
    check_stored_files(model_path, manifest["files_sha256"])
    info = {"model_id": model_id, "revision": entry["revision"],
            "path": str(model_path), "source": source}
    if base:
        base_path = local_model_path(
            str(base_path if base_path is not None else storage_paths(root, model_id, entry)[1]),
            "BASE_MODEL_PATH",
        )
        check_stored_files(base_path, base["files_sha256"])
        info["base"] = {"model_id": base["repo_id"], "revision": base["revision"], "path": str(base_path)}
    return info


def positive_env(name, default):
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_input(job, max_questions=8, max_input_bytes=131072):
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise ValueError("Job must contain an input object")
    payload = job["input"]
    if set(payload) != {"state", "questions"}:
        raise ValueError("input must contain exactly state and questions")
    if not isinstance(payload["state"], (str, dict, list)):
        raise ValueError("state must be text, an object, or an array")
    questions = payload["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= max_questions:
        raise ValueError(f"questions must contain 1..{max_questions} entries")
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, dict):
            raise ValueError("Each question needs a nonempty string ID and an object")
        if question.get("type") not in ("choice", "noul", "score"):
            raise ValueError(f"Question {key}: type must be choice, noul, or score")
        instructions = question.get("instructions")
        if not isinstance(instructions, (str, dict, list)) or instructions == "":
            raise ValueError(f"Question {key}: instructions must be text or structured data")
    try:
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("input must contain finite, serializable JSON data") from exc
    if len(encoded) > max_input_bytes:
        raise ValueError(f"input exceeds the configured {max_input_bytes}-byte limit")
    # Criteria and exact token budgets are checked by the pinned native runtime.
    return payload


class DecisionWorker:
    def __init__(self):
        import torch
        from transformers import AutoModel

        selected = selected_model_info()
        self.model_id = selected["model_id"]
        self.revision = selected["revision"]
        self.max_questions = positive_env("MAX_QUESTIONS", 8)
        self.max_input_bytes = positive_env("MAX_INPUT_BYTES", 131072)
        device = os.getenv("DEVICE", "cuda:0")
        if device not in ("cpu", "cuda:0"):
            raise ValueError("DEVICE must be cuda:0 or cpu; this runtime uses one device")
        if device == "cuda:0" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("This worker requires a CUDA GPU with BF16 support")
        LOGGER.info("Loading %s at %s from %s (%s) on %s",
                    self.model_id, self.revision, selected["path"], selected["source"], device)
        # Decision 2.0 has its own loader: explicit dtype/quantization overrides
        # are unsupported. Preserve its BF16-resident / FP32-head numerics.
        base_options = {"base_path": selected["base"]["path"]} if selected.get("base") else {}
        self.model = AutoModel.from_pretrained(
            selected["path"],
            local_files_only=True,
            trust_remote_code=True,
            device=device,
            bf16_resident=True,
            **base_options,
        )
        LOGGER.info("Model loaded; accepting jobs")

    def handle(self, job):
        payload = validate_input(job, self.max_questions, self.max_input_bytes)
        result = self.model.system_one(
            state=payload["state"], questions=payload["questions"]
        )
        answers = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers, dict) or set(answers) != set(payload["questions"]):
            raise RuntimeError("Model returned an unexpected answer set")
        # Native runtime can return per-question failures (including token limits).
        # Surface these as FAILED Runpod jobs rather than silent partial success.
        errors = {
            key: answer["error"]
            for key, answer in answers.items()
            if isinstance(answer, dict) and "error" in answer
        }
        if errors:
            raise ValueError("Decision questions failed: " + json.dumps(errors))
        return result
