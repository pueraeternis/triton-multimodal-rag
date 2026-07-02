# Pinned versions — update only after maintainer validation (see docs/QUICKSTART.md)
ARG TRITON_VERSION=25.05-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}

ARG VLLM_BACKEND_SHA=b41f716d15100dc7bcbea27ebea20906452dadf5
ARG VLLM_VERSION=0.10.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY infra/config/requirements.txt /tmp/requirements.txt

# Container-only pins: Triton base image constrains numpy; vLLM is not in pyproject (GPU serving).
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4 "vllm==${VLLM_VERSION}"

RUN mkdir -p /opt/tritonserver/backends/vllm \
    && git clone https://github.com/triton-inference-server/vllm_backend /tmp/vllm_backend \
    && cd /tmp/vllm_backend \
    && git checkout ${VLLM_BACKEND_SHA} \
    && cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm/

RUN rm -rf /tmp/*

WORKDIR /
