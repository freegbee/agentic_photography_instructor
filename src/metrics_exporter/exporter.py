"""
Prometheus exporter entrypoint.
Starts HTTP server and periodically updates metrics.
"""
import os
import signal
import time
import logging
from threading import Event, Thread

from prometheus_client import start_http_server
from collector import SystemGpuCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRAPE_INTERVAL_S = int(os.getenv('SCRAPE_INTERVAL_S', '5'))
EXPORTER_PORT = int(os.getenv('MONITORING_PROMETHEUS_GPU_EXPORTER_PORT', '5021'))
BIND_ADDR = os.getenv('EXPORTER_BIND_ADDR', '0.0.0.0')

stop_event = Event()


def run_collector_loop(collector: SystemGpuCollector):
    logger.info('Starting collector loop: interval=%ds', SCRAPE_INTERVAL_S)
    while not stop_event.is_set():
        try:
            collector.update()
        except Exception:
            logger.exception('Collector update failed')
        stop_event.wait(SCRAPE_INTERVAL_S)
    logger.info('Collector loop stopped')


def main():
    collector = SystemGpuCollector()

    # Start Prometheus HTTP server
    logger.info('Starting HTTP metrics server on %s:%d', BIND_ADDR, EXPORTER_PORT)
    start_http_server(EXPORTER_PORT, addr=BIND_ADDR)

    # start collector thread
    t = Thread(target=run_collector_loop, args=(collector,), daemon=True)
    t.start()

    def _signal_handler(signum, frame):
        logger.info('Signal received: %s, shutting down', signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    t.join(timeout=5)
    logger.info('Exporter shutdown complete')


if __name__ == '__main__':
    main()
