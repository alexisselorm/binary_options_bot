import asyncio
from utils.config import Config
from broker.deriv import DerivAPI, DerivAPIWrapper
from trading.executor import TradeExecutor
from notifier.telegram import TelegramNotifier


import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
pd.options.mode.chained_assignment = None        # or "warn" / "raise"


async def main():
    cfg = Config()
    notifier = TelegramNotifier(
        cfg.telegram_token, cfg.telegram_chat_id) if cfg.telegram_token else None

    api = DerivAPIWrapper(app_id=cfg.app_id)
    # No .connect(): the constructor handles that

    print(f"Loaded config: asset={cfg.asset}")

    auth = await api.authorize(cfg.api_token)
    # main.py  (snippet)
    if cfg.backtest:
        from trading.backtest import run_backtest
        # from trading.backtest_voting import run_voting_backtest
        run_backtest()
        # run_voting_backtest()
        return
    executor = TradeExecutor(api, cfg)
    await executor.run()


if __name__ == "__main__":
    asyncio.run(main())
