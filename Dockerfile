ARG TRITON_VERSION=25.05-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY infra/config/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4 vllm==0.10.2

RUN mkdir -p /opt/tritonserver/backends/vllm \
    && git clone https://github.com/triton-inference-server/vllm_backend /tmp/vllm_backend \
    && cd /tmp/vllm_backend \
    && git checkout r25.05 \
    && cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm/

RUN rm -rf /tmp/*

WORKDIR /
