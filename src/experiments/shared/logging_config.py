from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(levelname)s %(name)s %(process)d %(threadName)s %(message)s",
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
        # "file": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "formatter": "standard",
        #     "level": "DEBUG",
        #     "filename": "logs/app.log",
        #     "maxBytes": 10 * 1024 * 1024,
        #     "backupCount": 5,
        #     "encoding": "utf-8",
        #},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
    # "loggers": {
    #     "myapp": {
    #         "level": "DEBUG",
    #         "handlers": ["console"],
    #         "propagate": False,
    #     },
    # },
}

def configure_logging():
    dictConfig(LOGGING_CONFIG)
