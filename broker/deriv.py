import asyncio
import logging
from typing import Dict, Any, List
from deriv_api import DerivAPI
from utils.config import Config

logger = logging.getLogger("bot.broker")


class DerivAPIWrapper:
    def __init__(self, app_id: str):
        logger.debug("Initializing DerivAPIWrapper")
        self.api = DerivAPI(
            app_id=app_id
        )
        self.valid_durations: List[int] = []
        self.cfg = Config()

    async def authorize(self, token: str) -> Dict[str, Any]:
        logger.debug("Calling authorize() with raw token")
        response = await self.api.authorize(token)
        logger.info("Authorized successfully")
        return response

    async def get_balance(self) -> float:
        resp = await self.api.balance()
        # print("Received balance response:", resp)
        balance = resp.get("balance")

        if balance is None:
            raise ValueError("❌ Balance response missing 'balance' field.")
        return float(balance["balance"])

    async def fetch_trading_durations(self, asset: str) -> List[int]:
        """
        Fetches and returns valid durations (in seconds) for the given asset,
        limited to Forex and Synthetic Indices markets only.
        """
        logger.debug("📡 Sending trading_durations request")
        resp = await self.api.trading_durations({"trading_durations": 1})
        results = resp.get("trading_durations", [])
        valid_secs: List[int] = []

        for entry in results:  # iterate markets/submarkets
            market = entry.get("market")
            if market["name"] not in ("forex", "synthetic_index"):
                print(f"Skipping market: {market}")
                continue

            logger.debug(f"Processing market: {market}")
            for data_grp in entry.get("data", []):
                for sym in data_grp.get("symbol", []):
                    if sym.get("name") == asset:
                        logger.info(f"✅ Found asset: {asset} in {market}")
                        for td in data_grp.get("trade_durations", []):
                            for dur in td.get("durations", []):
                                unit = dur.get("name")
                                min_v = dur.get("min", 0)
                                max_v = dur.get("max", 0)
                                if unit == "s":
                                    valid_secs += list(range(min_v, max_v + 1))
                                elif unit == "m":
                                    valid_secs += [i *
                                                   60 for i in range(min_v, max_v + 1)]

        valid = sorted(set(valid_secs))
        logger.info(f"✔ Valid durations (sec) for {asset}: {valid}")
        # print(f"Valid durations (sec) for {asset}: {valid[0]}")
        return valid

    async def get_candles(self, symbol: str, count: int, granularity: int) -> Dict[str, Any]:
        print(
            f"Fetching candles for {symbol}: count={count}, gran={granularity}")
        return await self.api.ticks_history({
            "ticks_history": symbol,
            "count": count,
            "granularity": granularity,
            "style": "candles",
            "start": 1,
            "end": "latest"
        })

    async def proposal(self, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Requesting proposal: {args}")
        return await self.api.proposal(args)

    async def buy(self, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Buying contract: {args}")
        return await self.api.buy(args)

    async def subscribe(self, args: Dict[str, Any]):
        logger.debug(f"Subscribing with parameters: {args}")
        return await self.api.subscribe(args)
