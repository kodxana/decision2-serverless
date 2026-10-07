# Decision 2.0 for Runpod Serverless

A queue-based inference worker for classification, yes/no probability scoring,
and ordinal scoring with [Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20).

**Lux 9B is included in the image.** Set `MODEL_ID` to select another supported
model; the worker downloads it to an attached network volume on first use and
reuses it on subsequent starts. Vega's required Qwen base is handled automatically.

- One model initialization per worker; sequential request processing.
- Pinned model revisions, manifest checks and native weight verification.
- Persistent model storage with coordinated downloads and offline inference.

## Deploy

### Default: Lux 9B

Create a **queue-based Serverless endpoint** from the Hub template or import this
repository through Runpod's GitHub integration. For a source build, select
`main`, `Dockerfile`, and the repository root as the build context.

Use these initial settings:

| Setting | Value |
| --- | --- |
| Model environment variable | `MODEL_ID=vllm-sr/Decision-2.0-Lux-9B` |
| GPU | One BF16-capable GPU; start with 48 GB VRAM |
| Host RAM | At least 64 GB recommended |
| Container disk | 80 GB starting allocation |
| Minimum / maximum workers | `0` / `1` for initial validation |
| Concurrent jobs per worker | `1` |
| Container command | Leave unset; use the image entrypoint |
| Network volume | Not required for baked Lux |
| HTTP ports | None |

The image requires an NVIDIA driver compatible with CUDA 12.8. The worker serves
Runpod queue jobs, not an OpenAI-compatible chat API. Hardware guidance is a
starting point; see [validation](#validation) before promoting a deployment.

For a private GitHub repository, grant the Runpod GitHub App access. Existing
GitHub integrations deploy updates through **GitHub releases**, not commit pushes
alone. See the [Runpod integration guide](https://docs.runpod.io/serverless/workers/github-integration).
For Hub publishers, expose `MODEL_ID` with the Lux value above as its default.

### Select another model

1. Attach a network volume with enough free space for the selected model.
2. Change `MODEL_ID` to a value from the [supported models](#supported-models) table.
3. Select suitable GPU and host memory, then start replacement workers.
4. Submit a test job and confirm `output.model` identifies the intended model.

For Vega:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
```

No path variables are required for the standard layout. Vega downloads about
**70.57 GB**: its 14.98 GB decision package plus the 55.59 GB Qwen base. Start with
at least **100 GB of free volume space** and allow more for additional models.
The baked Lux files remain in the image, but only the selected model is loaded
into memory.

For a large first-use download, an initialization configuration to test is:

```dotenv
MODEL_DOWNLOAD_TIMEOUT=1800
RUNPOD_INIT_TIMEOUT=3600
```

The download timeout applies separately to lock acquisition and downloading.
Allow enough total initialization time for those stages, hashing and model
loading. **Downloads performed by this worker consume billable startup time.**
[Preload the volume](#preload-a-volume) to avoid downloading during GPU startup.
This workflow uses attached storage rather than Runpod's platform Model Store.

To return to baked Lux, restore its `MODEL_ID` and unset `MODEL_PATH` and
`BASE_MODEL_PATH`. Configuration changes take effect when workers initialize;
model selection is not a per-request option.

## API usage

Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in your client environment. These are
client credentials/settings, not additional worker configuration. Shell examples
below use Bash and assume the repository root as the working directory.

### Submit a job

The request contains exactly `input.state` and `input.questions`:

```json
{
  "input": {
    "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "returns": "Refunds, replacements and damaged deliveries",
          "billing": "Payments, invoices and charges",
          "technical": "Product setup and faults"
        }
      },
      "receipt": {
        "type": "noul",
        "instructions": "Does the customer have a receipt?"
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Soon", "Today"]
      }
    }
  }
}
```

The same request is saved in [`examples/request.json`](examples/request.json):

```bash
curl --fail-with-body --silent --show-error \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data-binary @examples/request.json
```

Save the returned job `id`. Set `RUNPOD_JOB_ID` to that value and poll:

```bash
curl --fail-with-body --silent --show-error \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${RUNPOD_JOB_ID}" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

A completed job has `status: "COMPLETED"` and an `output` object:

| Output field | Meaning |
| --- | --- |
| `model` | Loaded model name, for example `Decision-2.0-Lux-9B` |
| `answers` | Results keyed by the question IDs supplied in the request |
| `usage.input_tokens` | Input tokens reported by the native runtime |
| `usage.output_tokens` | `0`; this API returns decisions rather than generated text |

| Question type | Result |
| --- | --- |
| `choice` | Selected criterion in `choice`, with `probabilities` |
| `noul` | Yes/no probability in `noul` |
| `score` | Ordinal value in `score`, with `probabilities` |

`state` accepts text, an object or an array. Each question needs a nonempty ID,
a supported `type`, and `instructions`; criteria are validated by the native
runtime. The default limits are **8 questions** and **131,072 bytes** of serialized
input. Model token limits apply separately.

Validation and native question errors produce failed jobs; inspect job status
and error details even when the HTTP request succeeds. Continue polling queued
or running jobs. A client timeout does not cancel a job—reuse its ID instead of
submitting the same work again. Asynchronous `/run` is appropriate for cold starts.

## Supported models

Only the following Decision 2.0 models are supported. Exact revisions, manifest
hashes and base dependencies are recorded in [`models.json`](models.json).

| `MODEL_ID` | Model files | Input token limit |
| --- | ---: | ---: |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 1.52 GB | 8,192 |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 2.04 GB | 16,384 |
| `vllm-sr/Decision-2.0-Sol-2B` | 4.81 GB | 16,384 |
| `vllm-sr/Decision-2.0-Nox-4B` | 9.72 GB | 16,384 |
| **`vllm-sr/Decision-2.0-Lux-9B`** | **17.95 GB, baked** | **16,384** |
| `vllm-sr/Decision-2.0-Vega-27B` | 70.57 GB, including base | 32,768 |

Sizes are approximate decimal GB of downloaded files, not VRAM requirements.
Native prompt construction and question budgeting determine usable input length.
Arbitrary language models cannot be substituted into this decision API.

**Vega hardware remains unvalidated.** Start capacity testing with a single
80–96 GB BF16-capable GPU, preferably 96 GB, and at least 192 GB host RAM. The
loader initially materializes FP32 weights on the CPU; Vega's parameters alone
represent about 117.5 GB before loading overhead. Some tensors remain FP32 on the
GPU. Longer inputs and more questions increase memory demand. Quantization and
multi-GPU sharding are not implemented by this worker.

## Configuration

Set environment variables in the Hub template or endpoint configuration.
[`.env.example`](.env.example) is a reference; the application does not load a
`.env` file automatically.

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `vllm-sr/Decision-2.0-Lux-9B` | Model selected when the worker starts |
| `MODEL_ROOT` | `/runpod-volume/models` | Persistent directory for managed downloads |
| `MODEL_PATH` | Unset | Explicit preloaded model directory; disables automatic download |
| `BASE_MODEL_PATH` | Unset | Explicit preloaded Qwen base directory for Vega |
| `MODEL_DOWNLOAD_TIMEOUT` | `1800` | Positive seconds allowed for each lock wait and downloader process |
| `RUNPOD_INIT_TIMEOUT` | Unset by this image | Runpod-controlled worker initialization deadline; size for the workload |
| `DEVICE` | `cuda:0` | Inference device; `cpu` is available for local checks |
| `MAX_QUESTIONS` | `8` | Positive maximum number of questions per job |
| `MAX_INPUT_BYTES` | `131072` | Positive maximum serialized input size |
| `HF_HUB_OFFLINE` | `1` | Keep enabled for offline inference |
| `TRANSFORMERS_OFFLINE` | `1` | Keep enabled for offline inference |
| `HF_TOKEN` | Unset | Optional Hugging Face authentication; configure as a secret |

All catalog models are public and can be downloaded without a token. Only the
downloader process enables Hugging Face networking; model inference uses local
files. The queue worker still needs network access to the Runpod API.

## Persistent storage

[Runpod network volumes](https://docs.runpod.io/storage/network-volumes) mount at
`/runpod-volume` on Serverless. The default layout is:

```text
/runpod-volume/models/
  <repository-name>/<revision>/   Model or base files
  .state/                        Download records and locks
```

The worker coordinates downloads through a shared file lock. Completed copies
are reused across restarts and scale-to-zero. Failed or incomplete downloads
fail initialization and can resume on a later start. The requested model is
never silently replaced with Lux.

A single volume restricts placement to its data center. Multiple volumes do not
synchronize automatically; populate each one before relying on it. Monitor free
space: model revisions accumulate and there is no automatic eviction. Do not
modify files that active workers are using. Keep `/opt/decision2` and `/opt/models`
unmounted so the application and baked default remain available.

### Preload a volume

Run the downloader from the repository root in a Python 3.11+ environment. It
requires only the following download dependencies, not a GPU or PyTorch:

```bash
python -m pip install huggingface-hub==1.33.0 filelock==4.0.12
python model_storage.py \
  --model-id vllm-sr/Decision-2.0-Vega-27B \
  --root /workspace/models
```

This example assumes the target volume is mounted at `/workspace` on a Pod.
Its directory layout and download records remain usable when that same volume
mounts at `/runpod-volume` on Serverless. Use a writable volume for preparation;
a completed copy can be loaded read-only.

### Use existing model directories

For a custom layout, set both Vega paths explicitly:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
MODEL_PATH=/runpod-volume/my-vega
BASE_MODEL_PATH=/runpod-volume/my-qwen-base
```

`MODEL_PATH` must be an absolute directory containing the complete pinned model
package. Explicit paths are never automatically downloaded or repaired. Vega's
base must match the catalog revision. Use regular files, for example downloads
created with Hugging Face's `--local-dir` option, and retain all manifest-listed
files. A missing file, wrong manifest or checksum mismatch fails initialization.

## Build and validate

Build from the repository root with a Linux Docker engine:

```bash
docker build --platform linux/amd64 -t decision2-serverless:local .
```

The Dockerfile uses a public PyTorch base pinned by digest, installs locked
Python dependencies, and verifies downloaded weights before and after copying
into the final image. A GPU is not required to build. Reserve disk headroom for
build stages, cached layers and image export in addition to the model files.
The build argument `MODEL_ID` changes the baked model and its default selection;
use the runtime environment variable for normal deployment changes.

Run the CPU-only packaging and configuration checks without model downloads:

```bash
python -m unittest -v test_baking test_contract test_model_selection test_model_storage
```

Validate baked Lux inference on a suitable GPU:

```bash
docker run --rm --gpus all --network none --entrypoint python \
  decision2-serverless:local /opt/decision2/smoke_test.py
```

This checks expected decisions, finite normalized probabilities, repeated
requests and recovery after an invalid native question. Network isolation is for
this local test only; a deployed queue worker must reach Runpod.

For an endpoint you have deployed, set the client environment variables described
in [API usage](#api-usage), then run:

```bash
python check_endpoint.py
```

This submits billable jobs and verifies valid → invalid → valid behavior. It
polls each job for up to 15 minutes. If a first startup takes longer, inspect the
printed job ID and continue polling; a script timeout does not cancel that job.

### Validation

| Area | Evidence |
| --- | --- |
| Packaging, request validation and model selection | 34 automated checks passed on Windows and in the Linux runtime |
| Persistent download and reuse | Real Kai 0.6B CPU inference passed after first download and after an offline restart with read-only storage |
| Baked Lux | GPU inference passed on Runpod A40 workers with an earlier image; current loader path selection was checked against the actual baked files |
| Current complete image and managed GitHub build | Require release validation |
| Updated worker on a real network volume; Vega GPU inference | Not yet validated |

Validate the exact image, selected model, storage backend and hardware before
routing production traffic. Successful Docker builds or healthy workers alone
do not establish inference correctness.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `Storage parent is missing` | Attach a network volume or point `MODEL_ROOT` to a persistent location whose parent exists. |
| Download or lock timeout | Inspect worker logs, volume space, connectivity and other workers preparing the same model. Preload the volume or adjust both download and initialization timeouts. |
| Missing files or checksum mismatch | Restore the complete catalog-pinned package, or preload a fresh directory and switch paths. Do not edit model files used by active workers. |
| `MODEL_ID must name a Decision 2.0 model` | Copy an exact ID from the supported-model table. |
| `BASE_MODEL_PATH` error | Use it only for Vega. Unset it when returning to Lux or another self-contained model. |
| CUDA or memory failure | Confirm BF16 support, CUDA 12.8 driver compatibility, host RAM and VRAM. Reduce request size or choose larger hardware as appropriate. |
| Job remains queued during startup | Check worker readiness and initialization logs; model preparation and GPU allocation can delay execution. Keep polling the existing job ID. |
| Output still names the previous model | Confirm the new environment on replacement workers and allow old workers to drain. |
| GitHub push has not changed the endpoint | Publish the intended GitHub release and verify which release/image Runpod deployed. |

Upstream model licenses are retained with the baked and downloaded artifacts.
See each model repository for its license and usage terms.
