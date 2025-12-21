import logging
import queue
import sys
from logging.config import dictConfig
from logging.handlers import QueueListener

LOG_QUEUE: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)-7s %(process)5d %(threadName)s %(name)-40s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "json": {
            # Platzhalter: bei Bedarf JSON-Formatter hier einbauen
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": "ext://utils.LoggingUtils.LOG_QUEUE",
            "level": "DEBUG",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["queue"],
    },
}

# ANSI-Farben für Levels
_COLOR_MAP = {
    logging.DEBUG: "\x1b[37m",     # dunkelgrau
    logging.INFO: "\x1b[67m",      # mittelgrau
    logging.WARNING: "\x1b[33m",   # gelb
    logging.ERROR: "\x1b[31m",     # rot
    logging.CRITICAL: "\x1b[35m",  # magenta
}
_RESET = "\x1b[0m"

class ColoredFormatter(logging.Formatter):
    """
    Ein Formatter, der ganze formatierte Zeilen basierend auf dem Log-Level einfärbt.
    Färbung wird nur aktiviert, wenn Stream ein TTY ist (oder wenn colorama ANSI unterstützt).
    """
    def __init__(self, fmt, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if self.use_colors:
            color = _COLOR_MAP.get(record.levelno, "")
            return f"{color}{formatted}{_RESET}"
        return formatted

def configure_logging() -> QueueListener:
    """
    Konfiguriert das Logging-System mit einem QueueListener.
    Der QueueListener wird gestartet und zurückgegeben, damit der Aufrufer
    ihn bei Programmende stoppen kann.

    returns: Der gestartete QueueListener

    Nutzung im Hauptprogramm:
        listener = configure_logging()
        ...
        listener.stop()

    """
    dictConfig(LOGGING_CONFIG)

    # Erstelle die eigentliche Console-Handler-Instanz (wird vom Listener verwendet)
    fmt_cfg = LOGGING_CONFIG["formatters"]["standard"]
    # formatter = logging.Formatter(fmt_cfg["format"], datefmt=fmt_cfg.get("datefmt"))
    formatter = ColoredFormatter(fmt_cfg["format"], datefmt=fmt_cfg.get("datefmt"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Listener starten und zurückgeben
    listener = QueueListener(LOG_QUEUE, console_handler, respect_handler_level=True)
    listener.start()
    return listener
