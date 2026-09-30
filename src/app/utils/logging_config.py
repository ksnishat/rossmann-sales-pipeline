"""
Rossmann Sales Pipeline Logging Configuration
Structured JSON logging with request context
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

from src.app.config import settings


# Context variable for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class RequestContextFilter(logging.Filter):
    """Add request context to log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "N/A"
        record.user_id = user_id_var.get() or "N/A"
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        # Standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = settings.app_name
        log_record["environment"] = settings.app_env

        # Request context
        log_record["request_id"] = getattr(record, "request_id", "N/A")
        log_record["user_id"] = getattr(record, "user_id", "N/A")

        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "exc_info", "exc_text",
                "stack_info", "request_id", "user_id"
            ]:
                log_record[key] = value


def get_json_handler() -> logging.Handler:
    """Get JSON handler for structured logging"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomJsonFormatter())
    handler.addFilter(RequestContextFilter())
    return handler


def get_console_handler() -> logging.Handler:
    """Get human-readable console handler for development"""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())
    return handler


def setup_logging(
    log_level: Optional[str] = None,
    json_format: bool = True,
    include_console: bool = False
) -> logging.Logger:
    """
    Configure application logging.
    """
    level = getattr(logging, (log_level or settings.app_log_level).upper())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if json_format or settings.app_env == "production":
        root_logger.addHandler(get_json_handler())

    if include_console or settings.app_env == "development":
        root_logger.addHandler(get_console_handler())

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "log_level": level,
            "json_format": json_format,
            "environment": settings.app_env,
        }
    )

    return root_logger


def set_request_context(request_id: str = None, user_id: str = None) -> None:
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context() -> None:
    request_id_var.set(None)
    user_id_var.set(None)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_prediction(
    logger: logging.Logger,
    store_id: int,
    predicted_sales: float,
    actual_sales: Optional[float] = None,
    processing_time_ms: Optional[float] = None,
) -> None:
    """Log sales prediction with structured data"""
    extra = {
        "event_type": "sales_prediction",
        "store_id": store_id,
        "predicted_sales": predicted_sales,
        "processing_time_ms": processing_time_ms,
    }
    if actual_sales is not None:
        extra["actual_sales"] = actual_sales
        extra["error"] = abs(predicted_sales - actual_sales)
    logger.info("Sales prediction completed", extra=extra)


def log_model_operation(
    logger: logging.Logger,
    operation: str,
    model_name: str,
    version: Optional[str] = None,
    status: str = "success",
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log model operations"""
    extra = {
        "event_type": "model_operation",
        "operation": operation,
        "model_name": model_name,
        "status": status,
    }
    if version:
        extra["version"] = version
    if duration_ms:
        extra["duration_ms"] = duration_ms
    if error:
        extra["error"] = error
        logger.error(f"Model operation failed: {operation}", extra=extra)
    else:
        logger.info(f"Model operation completed: {operation}", extra=extra)


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
) -> None:
    """Log API request"""
    logger.info(
        "API request",
        extra={
            "event_type": "api_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "request_size_bytes": request_size,
            "response_size_bytes": response_size,
        }
    )