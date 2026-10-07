# Official stable PyTorch runtime, Linux/amd64; digest verified on 2026-10-06.
# The previously selected Runpod tag contained a 2.8 development build.
# This smaller runtime reuses stable PyTorch/CUDA without a second framework stack.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385 AS runtime

WORKDIR /opt/decision2
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/model-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    DEVICE=cuda:0 \
    MAX_QUESTIONS=8 \
    MAX_INPUT_BYTES=131072

# Keep dependency layers before source files so code changes reuse those layers.
COPY requirements.resolved.txt constraints.txt ./
RUN python -c "import torch; assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__" \
    && python -m pip install --no-cache-dir -c constraints.txt -r requirements.resolved.txt \
    && python -m pip check \
    && python -c "import torch, transformers, runpod; assert torch.__version__.split('+')[0] == '2.8.0'" \
    && mkdir -p /opt/model-cache

# Download the complete pinned package as regular files (no cache symlinks).
# Vega needs both its adapter package and the pinned Qwen base (~70.57 GB).
# A code-only edit below this layer reuses these downloads when cache is available.
FROM runtime AS model
COPY models.json bake_model.py ./
ARG MODEL_ID=vllm-sr/Decision-2.0-Vega-27B
RUN python bake_model.py --model-id "$MODEL_ID" --destination /opt/models/decision2

# Copy the checked package into its own immutable layer, independent of download
# staging. Verify the final filesystem as well as the download-stage filesystem.
FROM runtime AS final
COPY --from=model /opt/models /opt/models
COPY --from=model /opt/decision2/baked-model.json ./
COPY models.json bake_model.py ./
RUN python bake_model.py --verify-only

# Offline applies to Hugging Face only; the Runpod queue SDK still needs networking.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
COPY app.py handler.py smoke_test.py ./
COPY examples ./examples
# Explicitly replace any inherited Pod startup script with the queue worker.
ENTRYPOINT ["python", "-u", "/opt/decision2/handler.py"]
CMD []
