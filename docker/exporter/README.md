System & GPU Prometheus Exporter

This small exporter exposes CPU and GPU metrics at /metrics for Prometheus.

Usage (Docker Compose):
- Build and run using the project's docker-compose. The service is named `system-gpu-exporter`.

Environment variables:
- SCRAPE_INTERVAL_S: interval in seconds between metric collections (default 5)
- EXPORTER_PORT: port to bind the metrics server (default 8000)

Notes:
- On NVIDIA systems the exporter will attempt to use NVML via `pynvml` to collect GPU metrics. The host must have NVIDIA drivers and the NVIDIA Container Toolkit configured if you want GPU passthrough into the container.
- On macOS, `torch` can be optionally installed to detect MPS availability. This is optional and not installed by default because `torch` is large.

