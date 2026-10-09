"""Configuration module for Quotex Telegram Signal Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """Application settings."""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    ADMIN_CHAT_ID: int = int(os.getenv("ADMIN_CHAT_ID", "0"))
    CHANNEL_ID: int = int(os.getenv("CHANNEL_ID", "0"))

    # Quotex API
    QUOTEX_SSID: str = os.getenv("QUOTEX_SSID", "")
    QUOTEX_EMAIL: str = os.getenv("QUOTEX_EMAIL", "")
    QUOTEX_PASSWORD: str = os.getenv("QUOTEX_PASSWORD", "")
    QUOTEX_DEMO_MODE: bool = os.getenv("QUOTEX_DEMO_MODE", "true").lower() == "true"

    # Technical Indicators
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL: int = int(os.getenv("MACD_SIGNAL", "9"))
    BOLLINGER_PERIOD: int = int(os.getenv("BOLLINGER_PERIOD", "20"))

    # Risk Management
    MAX_TRADE_AMOUNT: float = float(os.getenv("MAX_TRADE_AMOUNT", "50.0"))
    MIN_WIN_RATE: float = float(os.getenv("MIN_WIN_RATE", "0.65"))
    MAX_DAILY_TRADES: int = int(os.getenv("MAX_DAILY_TRADES", "20"))
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "5"))
    TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "15"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./bot_database.db")

    # ML Model
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "./models/signal_model.pkl")

    # Server
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"


settings = Settings()
