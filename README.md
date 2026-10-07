# Decision 2.0 for Runpod Serverless

Run classification, yes/no probability scoring, and ordinal scoring with
[Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20).

**Lux 9B is included in the image and selected by default.** Change `MODEL_ID` to
use another supported model. It downloads to your network volume on first use
and is reused on subsequent starts.

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
   | Container disk | 80 GB |
   | Minimum / maximum workers | `0` / `1` |
   | Container command | Leave unset |
   | Network volume | Not required for Lux |

4. Send a request using the [API example](#send-a-request) below.

## Choose a model

To switch models, attach a network volume, change `MODEL_ID` in the endpoint's
environment variables, and restart the workers. For example:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
```

| `MODEL_ID` | Download size |
| --- | ---: |
| `vllm-sr/Decision-2.0-Kai-0.6B` | 1.52 GB |
| `vllm-sr/Decision-2.0-Eos-0.8B` | 2.04 GB |
| `vllm-sr/Decision-2.0-Sol-2B` | 4.81 GB |
| `vllm-sr/Decision-2.0-Nox-4B` | 9.72 GB |
| `vllm-sr/Decision-2.0-Lux-9B` | 17.95 GB, included in the image |
| `vllm-sr/Decision-2.0-Vega-27B` | 70.57 GB, including its required base |

Download sizes are storage requirements, not GPU memory requirements.

Models are stored under `/runpod-volume/models`. Attach enough storage for all
models you intend to keep. Vega's base model downloads automatically; allow at
least **100 GB of free volume space** for Vega.

For Vega, a single **80–96 GB GPU** and **192 GB or more host RAM** are provisional
starting targets; Vega GPU inference has not yet been validated.

The first download can take several minutes and consumes billable worker time.
For larger models, allow a longer startup window:

```dotenv
RUNPOD_INIT_TIMEOUT=3600
MODEL_DOWNLOAD_TIMEOUT=1800
```

To return to the included Lux model, restore its `MODEL_ID` and remove any
`MODEL_PATH` or `BASE_MODEL_PATH` overrides.

## Optional settings

Set these in the endpoint's environment configuration only when needed:

| Variable | Purpose |
| --- | --- |
| `MODEL_ROOT` | Download location; defaults to `/runpod-volume/models` |
| `MODEL_PATH` | Load an existing model directory instead of downloading |
| `BASE_MODEL_PATH` | Existing Qwen base directory when using a custom Vega path |
| `MODEL_DOWNLOAD_TIMEOUT` | Download timeout in seconds; defaults to `1800` |
| `RUNPOD_INIT_TIMEOUT` | Maximum worker initialization time allowed by Runpod |
| `HF_TOKEN` | Optional Hugging Face token for authenticated downloads |

For preloaded Vega files:

```dotenv
MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
MODEL_PATH=/runpod-volume/my-vega
BASE_MODEL_PATH=/runpod-volume/my-qwen-base
```

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
