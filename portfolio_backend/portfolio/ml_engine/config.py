from __future__ import annotations

"""Central configuration for the ML pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Model quality and output configuration."""

    output_dir: Path = field(default_factory=lambda: Path(os.getenv("ML_OUTPUT_DIR", "./models")))
    min_cv_mean: float = float(os.getenv("MIN_CV_MEAN", "0.55"))
    min_wf_f1: float = float(os.getenv("MIN_WF_F1", "0.50"))
    min_sharpe: float = float(os.getenv("MIN_SHARPE", "0.5"))
    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", "-0.25"))


@dataclass(frozen=True)
class DataConfig:
    """Data acquisition and cache configuration."""

    history_years: int = int(os.getenv("HISTORY_YEARS", "2"))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_CACHE_DIR", "./.data_cache")))
    cache_ttl_hours: int = int(os.getenv("DATA_CACHE_TTL_HOURS", "24"))
    yfinance_timeout: int = int(os.getenv("YFINANCE_TIMEOUT", "10"))


@dataclass(frozen=True)
class IntegrationConfig:
    """Integration configuration for external services."""

    portfolio_backend_url: str = os.getenv("PORTFOLIO_BACKEND_URL", "http://localhost:8081")
    ml_push_secret: str = os.getenv("ML_PUSH_SECRET", "")
    deepseek_url: str = os.getenv("DEEPSEEK_API_URL", "http://localhost:8090")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-r1:8b")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    auto_push: bool = os.getenv("AUTO_PUSH_MODEL", "0") == "1"
    auto_deepseek_weight: bool = os.getenv("AUTO_DEEPSEEK_WEIGHT", "0") == "1"


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)


config = Config()
