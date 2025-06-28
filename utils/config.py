# utils/config.py
"""Configuration loader that ONLY reads from the local .env file.
No values are fetched from—or written to—your OS environment. We parse
.env manually using `dotenv_values`, which returns a dictionary without
polluting `os.environ`.
"""
from pathlib import Path
from dotenv import dotenv_values


class Config:
    """Strict configuration loader.

    All keys **must** exist in the project's .env file located next to
    the project root. If a key is missing, we raise an error. Nothing
    is pulled from the operating‑system level environment.
    """

    def __init__(self, env_path: str | Path = "env") -> None:
        env_path = Path(env_path)
        if not env_path.exists():
            raise FileNotFoundError(f".env file not found at {env_path}")

        # <- returns dict, does *not* touch os.environ
        self._values = dotenv_values(env_path)

        # 🔐 Credentials & Trading Setup
        self.app_id: str = self._req("APP_ID")
        self.api_token: str = self._req("API_TOKEN")
        self.llm_api_token: str = self._req("LLM_API_TOKEN")
        self.asset: str = self._req("ASSET")
        self.expiry: int = int(self._req("EXPIRY"))
        self.stake: float = float(self._req("STAKE"))
        self.max_stake: float = float(self._req("MAX_STAKE"))
        self.timeframe: int = int(self._req("TIMEFRAME"))
        self.backtest: bool = self._req("BACKTEST").lower() == "true"

        # 📡 Notifications
        self.telegram_token: str = self._req("TELEGRAM_TOKEN")
        self.telegram_chat_id: str = self._req("TELEGRAM_CHAT_ID")

        # 🧠 AI Strategy
        self.model_type: str = self._req(
            "MODEL_TYPE").lower()  # lstm, xgboost, hybrid, rule
        self.model_path_lstm: str = self._req("MODEL_PATH_LSTM")
        self.model_path_xgb: str = self._req("MODEL_PATH_XGB")
        self.model_path_rl: str = self._req("MODEL_PATH_RL")
        self.signal_threshold: float = float(self._req("SIGNAL_THRESHOLD"))
        self.sequence_length: int = int(self._req("SEQUENCE_LENGTH"))

        # ⚖️ Risk Management
        self.max_consecutive_losses: int = int(
            self._req("MAX_CONSECUTIVE_LOSSES"))
        self.trade_risk_percent: float = float(self._req("TRADE_RISK_PERCENT"))
        self.min_balance: float = float(self._req("MIN_BALANCE"))
        self.martingale_mode: str = self._req("MARTINGALE_MODE").lower()

        # 🧪 Backtesting
        self.historical_data_path: str = self._req("HISTORICAL_DATA_PATH")
        self.save_trade_history: bool = self._req(
            "SAVE_TRADE_HISTORY").lower() == "true"
        self.trade_history_path: str = self._req("TRADE_HISTORY_PATH")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _req(self, key: str) -> str:
        """Return the value for *key* or raise an error if missing/empty."""
        val = self._values.get(key)
        if val in (None, ""):
            raise KeyError(
                f"❌ Required .env variable '{key}' is missing or empty.")
        return val
