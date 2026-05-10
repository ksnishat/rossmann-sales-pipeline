"""Shared pytest fixtures for all Rossmann tests."""

import os
import pytest


@pytest.fixture
def env_vars(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")