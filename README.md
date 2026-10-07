# Decision 2.0 for Runpod Serverless

Run classification, yes/no probability scoring, and ordinal scoring with
[Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20).

**All six Decision packages are included in the image. Lux 9B is selected by
default.** Change `MODEL_ID` to switch models. Only Vega needs an additional
Qwen base model from Runpod Model Store or network storage.

## Quick start

1. Deploy the Hub template as a **queue-based Serverless endpoint**.
2. Keep the default model:

   ```dotenv
   MODEL_ID=vllm-sr/Decision-2.0-Lux-9B
   ```

3. For Lux, start with these settings:

   | Setting | Value |
   | --- | --- |
   | GPU | One BF16-capable GPU with 48 GB VRAM |
   | Host RAM | 64 GB or more recommended |
   | Container disk | 100 GB recommended |
   | Minimum / maximum workers | `0` / `1` |
   | Container command | Leave unset |
   | Network volume | Not required for Kai, Eos, Sol, Nox, or Lux |

4. Send a request using the [API example](#send-a-request) below.

## Choose a model

Change `MODEL_ID` in the endpoint's environment variables and restart the
workers. For example:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Nox-4B
```

| `MODEL_ID` | Included in the image |
| --- | ---: |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 1.52 GB |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 2.04 GB |
| `vllm-sr/Decision-2.0-Sol-2B` | 4.81 GB |
| `vllm-sr/Decision-2.0-Nox-4B` | 9.72 GB |
| `vllm-sr/Decision-2.0-Lux-9B` | 17.95 GB |
| `vllm-sr/Decision-2.0-Vega-27B` | 14.98 GB; requires the separate Qwen base below |

The image contains approximately **51.02 GB of model files**, plus runtime
dependencies. These are storage sizes, not GPU memory requirements. One selected
model is loaded per worker.

## Using Vega 27B

Vega's adapter is included, but it needs **Qwen/Qwen3.8-27B** (approximately
**55.59 GB**) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
```

Choose one way to supply Qwen:

- **Runpod Model Store:** set the endpoint's **Model** field to
  `Qwen/Qwen3.8-27B` with the revision above. The worker automatically finds
  the cached base and loads it offline. Keep `MODEL_ID` set to Vega.
- **Network volume:** attach a volume with at least **70 GB free**. The worker
  downloads only the Qwen base to `/runpod-volume/models` on first use and
  reuses it on later starts.
- **Preloaded or mounted base:** set `BASE_MODEL_PATH` to the directory
  containing the pinned Qwen files, for example:

  ```dotenv
  BASE_MODEL_PATH=/runpod-volume/my-qwen-base
  ```

See [Runpod's cached-model setup](https://docs.runpod.io/serverless/endpoints/model-caching)
for the endpoint configuration. Changing `MODEL_ID` alone does not configure
Model Store. Mount storage under `/runpod-volume`; keep `/opt/models` available
for the included Decision packages.

For Vega, a single **80–96 GB GPU** and **192 GB or more host RAM** are provisional
starting targets; Vega GPU inference has not yet been validated.

Downloads performed by the worker consume billable worker time.
For the first Qwen download, allow a longer startup window:

```dotenv
RUNPOD_INIT_TIMEOUT=3600
MODEL_DOWNLOAD_TIMEOUT=1800
```

To return to Lux, restore its `MODEL_ID` and remove any `MODEL_PATH` or
`BASE_MODEL_PATH` overrides.

## Optional settings

Set these in the endpoint's environment configuration only when needed:

| Variable | Purpose |
| --- | --- |
| `MODEL_ROOT` | Qwen download location; defaults to `/runpod-volume/models` |
| `MODEL_PATH` | Override the included Decision package with an existing directory |
| `BASE_MODEL_PATH` | Existing Qwen base directory for Vega |
| `MODEL_DOWNLOAD_TIMEOUT` | Download timeout in seconds; defaults to `1800` |
| `RUNPOD_INIT_TIMEOUT` | Maximum worker initialization time allowed by Runpod |
| `HF_TOKEN` | Optional Hugging Face token for authenticated downloads |

Custom directories must contain the complete versions listed in
[`models.json`](models.json). Missing or incompatible files fail startup.

## Send a request

Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in your client environment. This
Bash example asks whether the customer has a receipt:

```bash
curl --fail-with-body --silent --show-error \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data '{
    "input": {
      "state": "The customer has a receipt and wants to return a damaged order.",
      "questions": {
        "receipt": {
          "type": "noul",
          "instructions": "Does the customer have a receipt?"
        }
      }
    }
  }'
```

Set `RUNPOD_JOB_ID` to the returned job `id`, then poll for the result:

```bash
curl --fail-with-body --silent --show-error \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${RUNPOD_JOB_ID}" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

When `status` is `COMPLETED`, `output` contains the model name, answers and token
usage. For this example, `output.answers.receipt.noul` is the yes/no probability.
If the job fails, inspect its error. If it is still queued or running, keep
polling the same job ID instead of resubmitting.

Supported question types:

| Type | Use |
| --- | --- |
| `choice` | Choose between named criteria |
| `noul` | Return a yes/no probability |
| `score` | Score against an ordered list of criteria |

See [`examples/request.json`](examples/request.json) for all three types in one
request. The default request limits are **8 questions** and **128 KiB** of input;
each model also has an input token limit.
