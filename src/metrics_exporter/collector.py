"""
Collector for system and GPU metrics.
Updates Prometheus Gauges periodically; robust to missing NVML / Torch.
"""
import os
import time
import logging
from typing import Dict

import psutil
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Try optional imports
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    pynvml = None
    _HAS_NVML = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

# Gauges (global)
CPU_USAGE = Gauge('system_cpu_percent', 'System CPU usage percent')
LOAD_1 = Gauge('system_load_1m', 'System load average 1m')
LOAD_5 = Gauge('system_load_5m', 'System load average 5m')
MEM_USED = Gauge('system_memory_used_bytes', 'System memory used in bytes')
MEM_TOTAL = Gauge('system_memory_total_bytes', 'System memory total in bytes')

GPU_NVML_PRESENT = Gauge('gpu_nvml_present', '1 if NVML is available and initialized, 0 otherwise')
GPU_COUNT = Gauge('gpu_nvml_count', 'Number of GPUs visible via NVML')
GPU_MPS_AVAILABLE = Gauge('gpu_mps_available', '1 if Apple MPS backend appears available via torch, 0 otherwise')

# dynamic per-GPU gauges stored here keyed by (index, uuid)
_gpu_util_gauges: Dict[str, Gauge] = {}
_gpu_mem_used_gauges: Dict[str, Gauge] = {}
_gpu_mem_total_gauges: Dict[str, Gauge] = {}
_gpu_temp_gauges: Dict[str, Gauge] = {}


def _ensure_nvml_initialized():
    if not _HAS_NVML:
        return False
    try:
        # Safe init: if already initialized, NVML may raise. Use try/except.
        pynvml.nvmlInit()
        return True
    except Exception:
        try:
            # If already initialized, consider it ok
            _ = pynvml.nvmlSystemGetDriverVersion()
            return True
        except Exception:
            return False


class SystemGpuCollector:
    def __init__(self):
        self._nvml_ready = False
        self._last_gpu_count = 0
        # Try NVML initialization once
        try:
            self._nvml_ready = _ensure_nvml_initialized()
        except Exception:
            self._nvml_ready = False

    def update(self):
        try:
            self._collect_cpu_memory()
        except Exception:
            logger.exception("Failed to collect CPU/memory metrics")

        # GPU via NVML
        if _HAS_NVML and self._nvml_ready:
            try:
                self._collect_nvml()
            except Exception:
                logger.exception("Failed to collect NVML metrics")
                GPU_NVML_PRESENT.set(0)
        else:
            GPU_NVML_PRESENT.set(0)
            GPU_COUNT.set(0)

        # MPS (Apple) detection via torch
        try:
            if _HAS_TORCH:
                mps_avail = 1 if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available() else 0
                GPU_MPS_AVAILABLE.set(mps_avail)
            else:
                # Torch not present: set as 0 (not detected)
                GPU_MPS_AVAILABLE.set(0)
        except Exception:
            logger.exception("Failed to detect MPS via torch")
            GPU_MPS_AVAILABLE.set(0)

    def _collect_cpu_memory(self):
        CPU_USAGE.set(psutil.cpu_percent(interval=None))
        try:
            load1, load5, _ = psutil.getloadavg()
            LOAD_1.set(load1)
            LOAD_5.set(load5)
        except Exception:
            # getloadavg not available on Windows; set to 0
            LOAD_1.set(0)
            LOAD_5.set(0)

        vm = psutil.virtual_memory()
        MEM_USED.set(vm.used)
        MEM_TOTAL.set(vm.total)

    def _collect_nvml(self):
        # Ensure NVML init
        try:
            if not _ensure_nvml_initialized():
                GPU_NVML_PRESENT.set(0)
                GPU_COUNT.set(0)
                return
        except Exception:
            GPU_NVML_PRESENT.set(0)
            GPU_COUNT.set(0)
            return

        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            count = 0

        GPU_NVML_PRESENT.set(1)
        GPU_COUNT.set(count)

        for idx in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8') if isinstance(pynvml.nvmlDeviceGetUUID(handle), bytes) else str(pynvml.nvmlDeviceGetUUID(handle))
                # utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    util = 0
                # memory
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem.used
                    mem_total = mem.total
                except Exception:
                    mem_used = 0
                    mem_total = 0
                # temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = 0

                key = f"{idx}:{uuid}"
                if key not in _gpu_util_gauges:
                    # create labeled gauges per GPU
                    _gpu_util_gauges[key] = Gauge('gpu_utilization_percent', 'GPU utilization percent', ['gpu_index', 'gpu_uuid'])
                    _gpu_mem_used_gauges[key] = Gauge('gpu_memory_used_bytes', 'GPU memory used bytes', ['gpu_index', 'gpu_uuid'])
                    _gpu_mem_total_gauges[key] = Gauge('gpu_memory_total_bytes', 'GPU memory total bytes', ['gpu_index', 'gpu_uuid'])
                    _gpu_temp_gauges[key] = Gauge('gpu_temperature_celsius', 'GPU temperature Celsius', ['gpu_index', 'gpu_uuid'])

                _gpu_util_gauges[key].labels(gpu_index=str(idx), gpu_uuid=uuid).set(util)
                _gpu_mem_used_gauges[key].labels(gpu_index=str(idx), gpu_uuid=uuid).set(mem_used)
                _gpu_mem_total_gauges[key].labels(gpu_index=str(idx), gpu_uuid=uuid).set(mem_total)
                _gpu_temp_gauges[key].labels(gpu_index=str(idx), gpu_uuid=uuid).set(temp)

            except Exception:
                logger.exception(f"Failed to collect NVML metrics for GPU {idx}")
                continue


if __name__ == '__main__':
    # quick local smoke run
    logging.basicConfig(level=logging.INFO)
    c = SystemGpuCollector()
    while True:
        c.update()
        time.sleep(int(os.getenv('SCRAPE_INTERVAL_S', '5')))

