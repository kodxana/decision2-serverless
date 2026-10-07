# Decision 2.0 Serverless — baked Lux, selectable models

The image bakes **Decision-2.0-Lux-9B** (~17.95 GB) and defaults to:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Lux-9B
```

Use that same value as the **Hub template's `MODEL_ID` environment default**.
Users can change just `MODEL_ID` to another supported Decision 2.0 model. Lux uses
its baked copy. Other selections download once into an attached network volume,
then reuse those files on subsequent worker starts. Only the selected model is
loaded into memory; the baked Lux files remain in the container image.

This is a queue-based worker calling the native `system_one()` API. Requests use
`input.state` and `input.questions`; responses contain decisions, probabilities
and usage. It is not a chat-completions HTTP server.

## Hub and GitHub configuration

- Branch: **`main`**; Dockerfile: **`Dockerfile`**; build context: repository root.
- Worker type: queue-based Serverless. Leave the command override empty.
- Hub environment default: **`MODEL_ID=vllm-sr/Decision-2.0-Lux-9B`**.
- Start with 0 minimum and 1 maximum worker, one GPU per worker.
- For Lux, the previous image passed GPU inference on an NVIDIA A40 (48 GB VRAM).
  80 GB container disk and at least 64 GB host RAM are sensible starting values.
  Larger models need different hardware; see below.
- Ensure Runpod's GitHub App has access to the private repository when importing.
  An existing GitHub integration watches releases for updates; a commit push alone
  does not deploy a new image. [Integration guide](https://docs.runpod.io/serverless/workers/github-integration)

The full Vega Docker build previously exceeded Runpod's 30-minute builder limit.
This version returns to baking only Lux. Its managed build still needs testing;
the smaller download does not guarantee completion within that limit.

## Switch to Vega with one environment variable

Attach a network volume with sufficient space, then change:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
```

No custom model path is required. The worker downloads both Vega's **14.98 GB**
adapter package and its **55.59 GB** Qwen base at the pinned revisions in
[`models.json`](models.json). Allow **at least 100 GB of free volume space** for
Vega as a starting allowance; more is needed when retaining multiple models.

[Runpod network volumes](https://docs.runpod.io/storage/network-volumes) mount at
`/runpod-volume` on Serverless. This worker stores models under
`/runpod-volume/models/<repository-name>/<revision>/`, keeping different model
versions separate. It records completed downloads under `models/.state/` and
uses a shared file lock to coordinate first-use downloads across workers. The
volume must already be attached; the worker does not create cloud storage.

First-use downloads happen **inside the worker**, so that startup time is
billable. This mode does not use Runpod's platform Model Store. For large models,
extend the worker initialization window; an initial configuration to test is:

```dotenv
RUNPOD_INIT_TIMEOUT=3600
MODEL_DOWNLOAD_TIMEOUT=1800
```

`MODEL_DOWNLOAD_TIMEOUT` limits each lock wait and the downloader process in
seconds. The platform's initialization deadline also has to allow model hashing
and loading. Actual startup time depends on network, volume speed and hardware.
Subsequent workers reuse completed downloads, including after scale-to-zero.

The downloader runs in a separate process with Hugging Face networking enabled.
Inference stays in offline mode. Failed downloads fail startup and can resume
later; the worker never silently substitutes baked Lux for a requested model.
Selecting Lux again immediately returns to the baked copy.

A single attached volume constrains workers to its data center. Multiple attached
volumes do not synchronize automatically: each needs its own model files.

## Optional preload and custom layouts

To avoid downloading during GPU worker startup, preload the same volume using a
CPU-capable environment with this repository's dependencies installed. On a Pod
where that volume is mounted at `/workspace`, run:

```sh
python model_storage.py \
  --model-id vllm-sr/Decision-2.0-Vega-27B \
  --root /workspace/models
```

The layout and completion records are reusable when the same volume mounts at
`/runpod-volume` in Serverless. No GPU is needed for downloading or verification.
Do not modify a completed model directory while workers are reading it.

For existing downloads stored elsewhere, set explicit paths:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
MODEL_PATH=/runpod-volume/my-vega
BASE_MODEL_PATH=/runpod-volume/my-qwen-base
```

Explicit `MODEL_PATH` means **load pre-existing files**, without automatic
downloads. Vega also needs its pinned Qwen base at `BASE_MODEL_PATH` (or the
standard revision directory under `MODEL_ROOT`). Download complete repositories
as regular files, such as with Hugging Face's `--local-dir` option. Keep the model
manifest and its referenced files intact. Do not mount over `/opt/decision2` or
`/opt/models`, which contain the application and baked default.

## Supported models and sizing

Only compatible Decision 2.0 models in the pinned catalog are accepted. Arbitrary
chat models cannot be substituted into this native decision API.

| `MODEL_ID` | Approximate downloaded files | Placement |
| --- | ---: | --- |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 1.52 GB | Volume on first selection |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 2.04 GB | Volume on first selection |
| `vllm-sr/Decision-2.0-Sol-2B` | 4.81 GB | Volume on first selection |
| `vllm-sr/Decision-2.0-Nox-4B` | 9.72 GB | Volume on first selection |
| `vllm-sr/Decision-2.0-Lux-9B` | 17.95 GB | Baked default |
| `vllm-sr/Decision-2.0-Vega-27B` | 70.57 GB including base | Volume on first selection |

Storage sizes are decimal GB, not GPU memory requirements. Every revision and
manifest SHA-256 is pinned in `models.json`. Startup checks the selected manifest,
code/config hashes and file completeness before loading. The native runtime
verifies weight hashes and the decision model identity.

Vega's manifest reports 29,365,153,792 loaded parameters. Its loader initially
materializes FP32 weights on the CPU, about 117.5 GB for parameters alone. Start
hardware testing with **192 GB or more host RAM** and a **single BF16-capable GPU
with 80–96 GB VRAM**, preferring 96 GB for headroom. These are provisional targets,
not verified Vega sizing. Some tensors remain FP32, and longer contexts require
more memory. This worker does not implement quantization or multi-GPU sharding.

## Environment variables

[`.env.example`](.env.example) lists example values; the worker does not read a
`.env` file automatically. Set values in Hub/Runpod's environment configuration.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `vllm-sr/Decision-2.0-Lux-9B` | Model selection at worker startup |
| `MODEL_ROOT` | `/runpod-volume/models` | Persistent download root |
| `MODEL_PATH` | Unset | Optional preloaded model directory; disables auto-download |
| `BASE_MODEL_PATH` | Unset | Optional preloaded base directory for Vega |
| `MODEL_DOWNLOAD_TIMEOUT` | `1800` | Downloader and lock-wait timeouts in seconds |
| `DEVICE` | `cuda:0` | `cpu` also supported for local checks with sufficient RAM |
| `MAX_QUESTIONS` | `8` | Maximum questions per job |
| `MAX_INPUT_BYTES` | `131072` | Serialized request size bound |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` | Keep enabled in the inference process |
| `HF_TOKEN` | Unset | Optional HF download authentication; configure as a secret |

Models are initialized once per worker. Changing `MODEL_ID` requires replacement
workers/restart, not a per-request parameter. The shipped public models need no
HF token. CUDA inference needs driver support compatible with CUDA 12.8.

## Build and tests

```sh
docker build --platform linux/amd64 -t decision2-lux-serverless:0.3.0 .
python -m unittest -v test_baking test_contract test_model_selection test_model_storage
```

The public PyTorch base is pinned by digest and dependencies by version. Build
stages verify the complete downloaded model package and the copied final image
filesystem. The build loads no tensors and needs no GPU. The build argument
`MODEL_ID` can change the baked model, but runtime selection normally avoids that.

After building, exercise real baked Lux inference on a suitable GPU:

```sh
docker run --rm --gpus all --network none --entrypoint python \
  decision2-lux-serverless:0.3.0 /opt/decision2/smoke_test.py
```

[`examples/request.json`](examples/request.json) exercises `choice`, `noul`
(yes/no probability), and `score`. The smoke test checks expected decisions,
finite normalized probabilities, repeat requests and native invalid-job recovery.
For an intentionally deployed endpoint, set `RUNPOD_API_KEY` and
`RUNPOD_ENDPOINT_ID`, then run `python check_endpoint.py` to submit real jobs.

## Verification status

- **Passed:** 34 packaging, model selection, storage lifecycle and request-contract
  tests on Windows and in the Linux runtime; Dockerfile build validation.
- **Passed with real weights:** Kai 0.6B first-use download and CPU inference using
  the updated source in the existing Linux runtime; a fresh container then passed
  the same smoke test with networking disabled and the volume mounted read-only.
  Valid decisions, repeat requests and invalid-job recovery passed in both runs.
- **Passed:** selection of the actual baked Lux files and their pinned local-path
  override. No new full Docker image was built during these source checks.
- **Previously passed:** baked Lux GPU inference on real Runpod NVIDIA A40 workers.
- **Pending:** this revision's managed GitHub build, Vega download/inference on a
  real network volume, and GPU tests of the updated worker.

Local deployment records, logs, test downloads and credentials are excluded from
Git and the Docker build context. Upstream model licenses are retained alongside
both baked and downloaded weights.
