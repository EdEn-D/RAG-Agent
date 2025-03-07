from dotenv import load_dotenv, find_dotenv
import os
from datetime import datetime
import logging
import logging.config
from fastapi import FastAPI
import threading


from controller.controller import Orchestrator

load_dotenv(find_dotenv())

class UTF8LogFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, bytes):
            try:
                record.msg = record.msg.decode('utf-8')
            except UnicodeDecodeError:
                pass  # Keep the original message if decoding fails
        return True

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format":  "%(asctime)s: %(name)s > %(filename)s > %(funcName)s:%(lineno)d ~ %(levelname)s: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
            "filters": ["utf8_filter"]
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": os.path.join('data', 'logs', f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            "maxBytes": 1024*1024*10,
            "backupCount": 5,
            "encoding": "utf-8",
            "filters": ["utf8_filter"]
        }
    },
    "filters": {
        "utf8_filter": {
            "()": UTF8LogFilter
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# Create logs directory if it doesn't exist
os.makedirs('data/logs', exist_ok=True)
logging.config.dictConfig(logging_config)

# Function to run the FastAPI app
def run_server(logger):
    logger.info("FastAPI app started")
    orchestrator.run(port=5051, host="127.0.0.1")

def run_view(title="Bot name"):
    # Start FastAPI app in a separate thread
    fastapi_thread = threading.Thread(target=run_server, args=(orchestrator.logger,))
    fastapi_thread.daemon = True  # This makes the FastAPI thread exit when the main program exits
    fastapi_thread.start()

    # Start Streamlit app
    orchestrator.run_view(title)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    run_view("Novella") # use for streamlit
    