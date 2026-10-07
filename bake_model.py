"""Bake pinned Decision packages; the all-model image leaves Qwen external."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

CATALOG_PATH = Path(__file__).with_name("models.json")
DEFAULT_MODEL = "vllm-sr/Decision-2.0-Lux-9B"


def validate_file_map(expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Model manifest must contain a nonempty file checksum map")
    for name, digest in expected.items():
        if not isinstance(name, str):
            raise ValueError("Unsafe model manifest path")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or ":" in name or not name or relative.as_posix() != name or name == "."):
            raise ValueError(f"Unsafe model manifest path: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid model checksum: {name}")


def verify_hashes(root, expected, *, allow_symlinks=False):
    """Verify pinned content; HF snapshots may link to their repository's blobs."""
    root = Path(root)
    validate_file_map(expected)
    for name, digest in expected.items():
        path = root / name
        if not allow_symlinks and (path.is_symlink() or any(p.is_symlink() for p in path.parents)):
            raise ValueError(f"Model file must not be a symlink: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Model checksum mismatch: {name}")


def verify_tree(root, expected, extra_files=()):
    """Hash regular files, reject traversal/symlinks and unexpected inventory."""
    root = Path(root)
    verify_hashes(root, expected)
    actual_files = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in model package: {name}")
        if name.startswith(".cache/huggingface/") or name == ".gitattributes":
            continue
        if path.is_file():
            actual_files.add(name)
    if actual_files != set(expected) | set(extra_files):
        raise ValueError("Downloaded package inventory differs from its manifest")


def verify_files(destination):
    """Verify every file against the package manifest without loading model tensors."""
    root = Path(destination)
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("profile") not in ("qwen-full", "qwen-adapter"):
        raise ValueError("This image requires a qwen-full or qwen-adapter model package")
    verify_tree(root, manifest["files_sha256"], {"MODEL_MANIFEST.json"})
    return manifest


def base_spec(manifest, entry):
    """Bind the upstream manifest's base identity to our pinned catalog."""
    if not entry.get("base_model"):
        if manifest.get("profile") != "qwen-full" or manifest.get("base"):
            raise ValueError("Unexpected external base in the model manifest")
        return None
    base = manifest.get("base", {})
    if (manifest.get("profile") != "qwen-adapter"
            or base.get("repo_id") != entry["base_model"]
            or base.get("revision") != entry["base_revision"]):
        raise ValueError("Base model identity differs from models.json")
    validate_file_map(base.get("files_sha256"))
    return base


def verify_baked_entry(metadata, catalog, *, allow_external_base=False):
    """Verify one package and either its baked base or external base identity."""
    entry = catalog[metadata["model_id"]]
    if metadata["revision"] != entry["revision"]:
        raise ValueError("Baked model revision differs from models.json")
    manifest = verify_files(metadata["path"])
    if allow_external_base:
        raw = (Path(metadata["path"]) / "MODEL_MANIFEST.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("Baked model manifest differs from the pinned revision")
    if manifest["model_name"] != metadata["model_id"].split("/")[-1]:
        raise ValueError("Downloaded manifest names a different model")
    base = base_spec(manifest, entry)
    baked_base = metadata.get("base")
    if base:
        if (not isinstance(baked_base, dict) or baked_base.get("model_id") != base["repo_id"]
                or baked_base.get("revision") != base["revision"]):
            raise ValueError("Baked base identity differs from its manifest")
        if allow_external_base and baked_base.get("external") is True:
            if "path" in baked_base:
                raise ValueError("External base must not claim a baked path")
        else:
            verify_tree(baked_base["path"], base["files_sha256"], {"LICENSE"})
    elif baked_base:
        raise ValueError("Unexpected baked base for a self-contained model")
    return metadata


def verify_baked_model(metadata_path):
    """Check the final image's complete inventory without fetching external bases."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if "models" not in metadata:
        return verify_baked_entry(metadata, catalog)
    if (metadata.get("default_model") != DEFAULT_MODEL
            or set(metadata["models"]) != set(catalog)):
        raise ValueError("Baked model collection differs from models.json")
    for model_id, info in metadata["models"].items():
        if info.get("model_id") != model_id:
            raise ValueError("Baked model collection identity mismatch")
        verify_baked_entry(info, catalog, allow_external_base=True)
    return metadata


def bake(model_id, destination, base_destination, metadata_path, *, download,
         existing_model=None, existing_base=None, skip_base=False):
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if model_id not in catalog:
        raise ValueError("Select a model from models.json")
    destination = Path(existing_model if existing_model is not None else destination).resolve()
    base_destination = Path(existing_base if existing_base is not None else base_destination).resolve()
    if destination.is_relative_to(base_destination) or base_destination.is_relative_to(destination):
        raise ValueError("Model and base destinations must be separate directories")
    entry = catalog[model_id]
    revision = entry["revision"]
    print(f"Baking {model_id}@{revision} into {destination}", flush=True)
    if existing_model is None:
        download(repo_id=model_id, revision=revision, local_dir=destination)
        manifest = verify_files(destination)
    else:
        raw = (destination / "MODEL_MANIFEST.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("Existing model manifest differs from the pinned revision")
        manifest = json.loads(raw)
        verify_hashes(destination, manifest["files_sha256"], allow_symlinks=True)
    if manifest["model_name"] != model_id.split("/")[-1]:
        raise ValueError("Downloaded manifest names a different model")
    if skip_base:
        raw = (destination / "MODEL_MANIFEST.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("Model manifest differs from the pinned revision")
    base = base_spec(manifest, entry)
    if existing_base is not None and base is None:
        raise ValueError("This model does not use an external base")
    metadata = {
        "model_id": model_id,
        "revision": revision,
        "path": str(destination),
        "files_verified": len(manifest["files_sha256"]),
    }
    if skip_base and existing_base is not None:
        raise ValueError("Cannot skip and supply the base at the same time")
    if base and skip_base:
        metadata["base"] = {"model_id": base["repo_id"], "revision": base["revision"], "external": True}
    elif base:
        print(f"Baking base {base['repo_id']}@{base['revision']} into {base_destination}", flush=True)
        if existing_base is None:
            download(
                repo_id=base["repo_id"],
                revision=base["revision"],
                local_dir=base_destination,
                allow_patterns=sorted(set(base["files_sha256"]) | {"LICENSE"}),
            )
            # Keep the upstream license alongside the pinned base weights.
            verify_tree(base_destination, base["files_sha256"], {"LICENSE"})
        else:
            # Model Store holds the full upstream repo, including unrelated files.
            verify_hashes(base_destination, base["files_sha256"], allow_symlinks=True)
        metadata["base"] = {
            "model_id": base["repo_id"], "revision": base["revision"],
            "path": str(base_destination), "files_verified": len(base["files_sha256"]),
        }
    if metadata_path is not None:
        Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return metadata


def bake_all(destination, metadata_path, *, download):
    """Include every Decision package, with no download of Vega's Qwen base."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    root = Path(destination)
    models = {}
    for model_id in catalog:
        models[model_id] = bake(
            model_id, root / model_id.split("/")[-1], root / ".unused-base", None,
            download=download, skip_base=True,
        )
    metadata = {"default_model": DEFAULT_MODEL, "models": models}
    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--destination", type=Path, default=Path("/opt/models/decision2"))
    parser.add_argument("--base-destination", type=Path, default=Path("/opt/models/base"))
    parser.add_argument("--metadata", type=Path, default=Path(__file__).with_name("baked-model.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--existing-model", type=Path)
    parser.add_argument("--existing-base", type=Path)
    parser.add_argument("--all", action="store_true", help="Bake all Decision packages without external bases")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_baked_model(args.metadata)), flush=True)
        print("All baked files verified; external base weights are supplied at runtime", flush=True)
    elif args.all:
        from huggingface_hub import snapshot_download

        bake_all(args.destination, args.metadata, download=snapshot_download)
    else:
        from huggingface_hub import snapshot_download

        bake(args.model_id, args.destination, args.base_destination, args.metadata,
             download=snapshot_download, existing_model=args.existing_model,
             existing_base=args.existing_base)


if __name__ == "__main__":
    main()
