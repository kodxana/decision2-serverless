"""Runpod adapter for the pinned Decision 2.0 System One API."""

import json
import logging
import os
from pathlib import Path

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
    if os.getenv("MODEL_ID", model_id) != model_id:
        raise ValueError("MODEL_ID differs from the baked model; select a different model at build time")
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

        baked = baked_model_info()
        self.model_id = baked["model_id"]
        self.revision = baked["revision"]
        self.max_questions = positive_env("MAX_QUESTIONS", 8)
        self.max_input_bytes = positive_env("MAX_INPUT_BYTES", 131072)
        device = os.getenv("DEVICE", "cuda:0")
        if device not in ("cpu", "cuda:0"):
            raise ValueError("DEVICE must be cuda:0 or cpu; this runtime uses one device")
        if device == "cuda:0" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("This worker requires a CUDA GPU with BF16 support")
        LOGGER.info("Loading %s at %s on %s", self.model_id, self.revision, device)
        # Decision 2.0 has its own loader: explicit dtype/quantization overrides
        # are unsupported. Preserve its BF16-resident / FP32-head numerics.
        base_options = {"base_path": baked["base"]["path"]} if baked.get("base") else {}
        self.model = AutoModel.from_pretrained(
            baked["path"],
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
