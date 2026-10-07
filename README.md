# Decision 2.0 Vega 27B — baked Runpod Serverless worker

This Dockerfile defaults to **Decision-2.0-Vega-27B**, the largest model in the
[Decision 2.0 collection](https://huggingface.co/collections/vllm-sr/decision-20).
It bakes both Vega's adapter package and its required Qwen base into the image:

| Artifact | Pinned revision | Download size, decimal GB |
| --- | --- | ---: |
| [Vega 27B](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) | `7aec49ae11a18741706da549ab626b9052795fe7` | 14.983 |
| [Qwen3.8-27B base](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 55.586 |
| **Total model files** | | **70.569 GB / about 65.72 GiB** |

The base is downloaded during the Docker build, alongside its license. The
worker passes `/opt/models/base` explicitly to Vega's custom loader. Hugging Face
offline mode and `local_files_only=True` prevent runtime model downloads. Missing
or inconsistent artifacts fail startup. Do not mount a volume over `/opt/models`.

This is a **queue-based worker** using the model's native `system_one()` API.
Requests contain `input.state` and `input.questions`; responses contain decisions
and probabilities. It does not expose a chat-completions API or HTTP port.

## Runpod GitHub integration build test

Use this private repository as the source for a new Serverless endpoint in the
[Runpod console](https://console.runpod.io/serverless):

1. Allow the Runpod GitHub App to access this private repository if it is not
   already covered by the installation's permissions.
2. Import the repository, select **`main`**, and use **`Dockerfile`** at the root.
   The Docker build context is the repository root. No build argument or Hugging
   Face token is needed for the default public Vega/base artifacts.
3. Use the queue worker type and leave the container command override empty.
   The image entrypoint is `python -u /opt/decision2/handler.py`.
4. Watch the build logs for both pinned downloads and
   **`Final model and base checksums verified`**. A successful Docker build alone
   does not prove that a GPU worker can initialize and answer a job.
5. For a subsequent worker test, start with minimum **0**, maximum **1** worker,
   one sufficiently large BF16-capable GPU, and the request in
   [`examples/request.json`](examples/request.json). See the provisional hardware
   guidance below before selecting a GPU pool.

The [GitHub integration documentation](https://docs.runpod.io/serverless/workers/github-integration#limitations)
currently lists an **80 GB image limit**, a **30-minute Docker build limit**, and
a **160-minute total pipeline limit**. The model files alone take about 70.57 GB.
The previously measured runtime filesystem adds roughly 8.3 GB, giving an
**estimated 78.9 GB visible filesystem**. This is close to the limit; the final
Vega image and Runpod's size accounting have **not** been measured. Download,
two checksum passes, image assembly and transfer may also hit the time limit.
Treat this as a build-capacity experiment, not a guaranteed successful import.

The Docker build needs no GPU and never loads model tensors. The managed worker
test after a build does need a GPU. A private source repository still downloads
the public pinned models when Runpod builds it. No weights or credentials belong
in Git. For later updates, Runpod's documented integration watches **GitHub
releases**; pushing a commit by itself does not roll out an update.

## Build design

- The public, digest-pinned PyTorch runtime provides Torch 2.8.0 and CUDA 12.8.
  `requirements.resolved.txt` pins the Python dependencies, and `pip check`
  validates them during the build.
- A separate model stage downloads regular files directly into
  `/opt/models/decision2` and `/opt/models/base`, without a second blob-cache copy.
  The catalog and upstream Vega manifest must agree on the base identity.
- The bake script checks every manifest-listed model and base SHA-256, rejects
  missing/unexpected package files, and retains both upstream licenses.
- The final stage copies `/opt/models` once and verifies both artifact trees
  again. This deliberately checks the copied filesystem before publication.
- The small worker source is copied last, allowing dependency and model layers
  to be reused for source-only changes when the builder retains its cache.
- `baked-model.json` is generated inside the image and records both revisions,
  paths and verified-file counts. No model download occurs during Python import.

The model's `trust_remote_code=True` loader is bound to the downloaded immutable
revision. Its native numerical behavior is preserved; no quantization, dtype
override, multi-GPU sharding, or monkeypatch of the model code is applied.

## Local build and verification

Run from the repository root with a Linux Docker engine:

```sh
docker build --platform linux/amd64 -t decision2-vega-serverless:0.1.0 .
```

Allow substantial build-disk headroom beyond the final image: the download stage,
final copy, layer export and builder cache can coexist. More than 200 GB of free
builder disk is a prudent starting allowance, not a measured minimum.

Fast packaging and request-contract checks require only Python 3.11 or newer:

```sh
python -m unittest -v test_baking test_contract
```

These use tiny fixtures to test both pinned downloads, checksum failures, missing
base files, identity mismatches, local base-path forwarding, Lux compatibility,
and input validation. They do **not** exercise real Vega weights or GPU inference.

After building, verify real inference on a suitable GPU with model networking
disabled:

```sh
docker run --rm --gpus all --network none --entrypoint python \
  decision2-vega-serverless:0.1.0 /opt/decision2/smoke_test.py
```

The smoke test requires a sensible return-routing decision, normalized finite
probabilities, repeated valid requests and recovery after an invalid native
question. `--network none` is for this local smoke test; a deployed queue worker
needs network access to the Runpod API.

To build the previously exercised smaller Lux model instead:

```sh
docker build --platform linux/amd64 \
  --build-arg MODEL_ID=vllm-sr/Decision-2.0-Lux-9B \
  -t decision2-lux-serverless:local .
```

Changing `MODEL_ID` at runtime cannot switch baked weights. Rebuild for a different
model. `models.json` records collection sizes and immutable revisions; this
packaging route accepts the `qwen-full` and `qwen-adapter` manifest profiles.

## GPU worker configuration

**Vega hardware sizing is provisional until tested.** Its manifest reports
29,365,153,792 loaded parameters. The native loader materializes FP32 weights on
the CPU before converting eligible GPU linear weights to BF16. FP32 parameters
alone account for about 117.5 GB of host memory; additional loading buffers need
headroom. A host with **192 GB RAM or more** is a conservative starting target,
not a verified requirement or a guarantee provided by a GPU pool selection.

Use a **single GPU with at least 80 GB VRAM**, preferably 96 GB for initial tests,
and CUDA driver support for CUDA 12.8. Confirm the chosen worker's actual host
RAM. Some tensors stay FP32; long contexts, question count and graph buffers
increase memory demand. No 80 GB or 96 GB Vega inference run has been verified.
The custom loader uses one device and does not split across multiple GPUs.

Suggested initial settings, to validate on deployment:

| Setting | Starting value |
| --- | --- |
| Worker type | Queue-based Serverless |
| GPU count | 1 |
| Container disk | 120 GB, then check actual use |
| Minimum / maximum workers | 0 / 1 |
| Concurrency | 1 job per worker |
| Idle timeout | 5 seconds; increase to keep a worker warm |
| Execution timeout | 120 seconds; larger jobs need benchmarking |
| Environment | Defaults below; no HF token needed |
| Ports / network volume | None required |

Baking avoids the Hugging Face download during startup. An uncached host still
pulls the large image, and each worker verifies and loads the tensors. Cold-start
latency for Vega has not been measured.

## Requests and runtime settings

[`examples/request.json`](examples/request.json) exercises `choice`, `noul`
(yes/no probability), and `score`. Send that JSON to
`https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run`, authenticate with your Runpod key,
and poll the returned job ID at `/status/JOB_ID`. Use asynchronous submission for
cold workers so a client wait timeout does not trigger duplicate jobs.

For an endpoint you have deliberately deployed, set `RUNPOD_API_KEY` and
`RUNPOD_ENDPOINT_ID` in your environment, then run:

```sh
python check_endpoint.py
```

This submits real, potentially billable jobs: valid inference, an invalid request
that should return `FAILED`, then another valid request.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `cuda:0` | `cpu` is available for debugging with sufficient RAM |
| `MAX_QUESTIONS` | `8` | Maximum questions per job |
| `MAX_INPUT_BYTES` | `131072` | Serialized JSON input bound |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | Keep enabled for baked deployment |
| `MODEL_ID` | Unset | Optional consistency check; must match the baked model |

The native runtime enforces criteria and token limits. Vega supports up to 32,768
input tokens subject to its native budgeting. Per-question errors are surfaced
as failed jobs rather than partial successes.

## Verification status

- The earlier Lux 9B image passed offline inference and real Runpod A40 endpoint
  tests. That result validates the existing worker path, not Vega's capacity.
- Vega's pinned base requirements were checked against its manifest and loader.
  All 28 required base filenames exist at the pinned Qwen revision.
- All **18 fixture-based packaging, runtime handoff and request-contract tests**
  passed on Windows and in the existing Linux Python 3.11 runtime with networking
  disabled. Docker's `build --check` completed without warnings. These checks do
  not download or load the 70.57 GB artifacts.
- The complete Vega image, managed GitHub build, and real Vega GPU inference are
  **unverified** until the build experiment and worker smoke test complete.

Local deployment records, logs, research files, credentials and model weights are
excluded from this source repository. Both upstream model repositories identify
their licenses as Apache 2.0; their license files are retained in the image.
