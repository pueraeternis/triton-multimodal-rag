# Observability

This document describes the observability surface of the reference implementation. **No Grafana dashboards or OpenTelemetry integration are included** — this covers only what the repository actually exposes.

---

## Debug Trace Schema

Every BLS inference response includes a JSON debug trace alongside the generated answer. The client (`client.py`) parses and pretty-prints this trace.

### Top-Level Response

```json
{
  "answer": "<generated text>",
  "debug": {
    "input_query": "<user query string>",
    "steps": [ ... ],
    "total_latency_ms": <float>
  }
}
```

### Step Objects

Each entry in `debug.steps` corresponds to one pipeline stage:

#### YOLOv8 (Vision)

```json
{
  "component": "YOLOv8 (Vision)",
  "latency_ms": 47.31,
  "status": "Success",
  "details": "Output resided on GPU (Optimization)"
}
```

#### Qdrant (Retrieval)

```json
{
  "component": "Qdrant (Retrieval)",
  "latency_ms": 13.64,
  "candidates_found": 5,
  "top_candidate_preview": {
    "score": 0.82,
    "category": "Router",
    "issue": "Red status LED blinking continuously...",
    "full_solution": "Check the router logs to identify..."
  }
}
```

#### Cross-Encoder (Reranker)

```json
{
  "component": "Cross-Encoder (Reranker)",
  "latency_ms": 10.27,
  "best_score": -2.6081,
  "selected_context_preview": "Check the router logs to identify the specific error code..."
}
```

#### vLLM (Generation)

```json
{
  "component": "vLLM (Generation)",
  "latency_ms": 4496.82,
  "generated_length": 512
}
```

### Complete Example

```json
{
  "answer": "If the red status LED on your router is blinking continuously...",
  "debug": {
    "input_query": "Red status LED is blinking continuously on my Router. What to do?",
    "steps": [
      {
        "component": "YOLOv8 (Vision)",
        "latency_ms": 47.31,
        "status": "Success",
        "details": "Output resided on GPU (Optimization)"
      },
      {
        "component": "Qdrant (Retrieval)",
        "latency_ms": 13.64,
        "candidates_found": 5,
        "top_candidate_preview": {
          "score": 0.82,
          "category": "Router",
          "issue": "Red status LED blinking continuously...",
          "full_solution": "Check the router logs to identify the specific error code..."
        }
      },
      {
        "component": "Cross-Encoder (Reranker)",
        "latency_ms": 10.27,
        "best_score": -2.6081,
        "selected_context_preview": "Check the router logs to identify the specific error code associated with the LED pattern. Verify power stab..."
      },
      {
        "component": "vLLM (Generation)",
        "latency_ms": 4496.82,
        "generated_length": 487
      }
    ],
    "total_latency_ms": 4568.04
  }
}
```

> Latencies and scores above are representative. Maintainer-validated evidence is recorded in [plan-02-validation.md](validation/plan-02-validation.md).

---

## Triton Prometheus Metrics

Triton exposes Prometheus-format metrics on port **8002** (configured in `docker-compose.yml`).

### Accessing Metrics

```bash
curl -s localhost:8002/metrics | head -30
```

### Relevant Metric Families

| Metric Pattern | Description |
|----------------|-------------|
| `nv_inference_request_success` | Successful inference requests per model |
| `nv_inference_request_failure` | Failed inference requests per model |
| `nv_inference_queue_duration_us` | Time spent in Triton's scheduling queue |
| `nv_inference_compute_infer_duration_us` | Model inference compute time |
| `nv_inference_compute_input_duration_us` | Input tensor preparation time |
| `nv_inference_compute_output_duration_us` | Output tensor extraction time |
| `nv_gpu_utilization` | GPU utilization percentage |
| `nv_gpu_memory_used_bytes` | GPU memory used by Triton |
| `nv_gpu_memory_total_bytes` | Total GPU memory |

Filter by model:

```bash
curl -s localhost:8002/metrics | grep 'bls_orchestrator'
```

---

## Reference Prometheus Scrape Config

> **Not deployed.** This snippet is provided as a reference for operators who want to scrape Triton metrics externally.

```yaml
# prometheus.yml (reference only)
scrape_configs:
  - job_name: triton
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:8002"]
        labels:
          service: triton-multimodal-rag
```

No Grafana dashboards, alerting rules, or Prometheus deployment are included in this repository.

---

## What Is Not Included

| Capability | Status |
|------------|--------|
| Grafana dashboards | Not included |
| OpenTelemetry tracing | Not included |
| Structured logging (JSON) | Not included — Triton verbose logs only |
| Request ID propagation | Not included |
| Per-stage Prometheus counters in BLS | Not included — trace is per-response JSON only |

For production deployments, consider adding external observability tooling on top of the surfaces documented here.
