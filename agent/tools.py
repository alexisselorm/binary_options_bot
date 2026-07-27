from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from broker.deriv import DerivAPIWrapper
from trading.executor import TradeExecutor
from trading.strategy import add_indicators, generate_signals
from utils.config import Config
from agent.schemas import ProposalArgs

logger = logging.getLogger("bot.agent")


class FetchCandlesInput(BaseModel):
    symbol: str
    count: int = Field(..., gt=0)
    granularity: int = Field(..., gt=0)


class ComputeIndicatorsInput(BaseModel):
    candles: List[Dict[str, Any]]


class EvaluateSignalInput(BaseModel):
    candles: List[Dict[str, Any]]
    balance: float = Field(..., gt=0)
    strategy: str


class PlaceTradeInput(ProposalArgs):
    signal_id: str = Field(..., min_length=1)


class AgentToolRuntime:
    def __init__(self, api: DerivAPIWrapper, executor: TradeExecutor, cfg: Config):
        self.api = api
        self.executor = executor
        self.cfg = cfg
        self.last_signal_ids: Set[str] = set()

    async def fetch_candles(self, symbol: str, count: int, granularity: int) -> Dict[str, Any]:
        logger.info("tool_call=fetch_candles symbol=%s count=%s granularity=%s", symbol, count, granularity)
        resp = await self.api.get_candles(symbol, count, granularity)
        candles = resp.get("candles")
        if not candles:
            raise ValueError("No candles returned")
        return {"candles": candles}

    async def compute_indicators(self, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("tool_call=compute_indicators candle_count=%s", len(candles))
        df = pd.DataFrame(candles)
        out = add_indicators(df)
        rows = out.tail(200).reset_index(drop=False)
        for col in rows.columns:
            rows[col] = rows[col].astype(str) if rows[col].dtype == "datetime64[ns]" else rows[col]
        return {"candles_with_indicators": rows.to_dict(orient="records")}

    async def evaluate_signal(self, candles: List[Dict[str, Any]], balance: float, strategy: str) -> Dict[str, Any]:
        logger.info("tool_call=evaluate_signal strategy=%s", strategy)
        df = pd.DataFrame(candles)
        signal, stake, confidence = generate_signals(df, balance, use_ai=False, strategy=strategy)
        return {
            "signal": signal,
            "stake": stake,
            "confidence": confidence,
        }

    async def get_account_state(self) -> Dict[str, Any]:
        logger.info("tool_call=get_account_state")
        balance = await self.api.get_balance()
        open_positions = await self.api.get_open_positions()
        return {"balance": balance, "open_positions": open_positions}

    async def place_trade(self, data: PlaceTradeInput, dry_run: bool) -> Dict[str, Any]:
        logger.info("tool_call=place_trade signal_id=%s symbol=%s", data.signal_id, data.symbol)
        proposal_args = data.model_dump(exclude={"signal_id"})
        balance = await self.api.get_balance()
        open_positions = await self.api.get_open_positions()
        allowed_symbols = set(getattr(self.cfg, "agent_allowed_symbols", [self.cfg.asset]))
        max_positions = int(getattr(self.cfg, "agent_max_concurrent_positions", 1))
        max_stake = min(float(self.cfg.max_stake), float(getattr(self.cfg, "agent_max_stake", self.cfg.max_stake)))

        if data.signal_id in self.last_signal_ids:
            raise ValueError(f"Duplicate signal_id blocked: {data.signal_id}")
        if proposal_args["symbol"] not in allowed_symbols:
            raise ValueError(f"Symbol not allowed: {proposal_args['symbol']}")
        if len(open_positions) >= max_positions:
            raise ValueError(f"Max concurrent positions reached: {len(open_positions)} >= {max_positions}")
        if proposal_args["amount"] > max_stake:
            raise ValueError(f"Stake exceeds max stake: {proposal_args['amount']} > {max_stake}")
        if proposal_args["amount"] > balance:
            raise ValueError(f"Insufficient balance: {proposal_args['amount']} > {balance}")

        self.last_signal_ids.add(data.signal_id)

        if dry_run:
            return {
                "buy": {
                    "contract_id": f"dryrun-{uuid.uuid4().hex[:12]}",
                    "price": proposal_args["amount"],
                },
                "dry_run": True,
                "proposal_args": proposal_args,
                "timestamp": int(time.time()),
            }

        result = await self.executor.execute_trade(proposal_args)
        if not result:
            raise RuntimeError("Trade execution failed")
        return result

    async def place_trade_tool(
        self,
        proposal: int,
        amount: float,
        basis: str,
        contract_type: str,
        currency: str,
        duration: int,
        duration_unit: str,
        symbol: str,
        signal_id: str,
    ) -> Dict[str, Any]:
        payload = PlaceTradeInput(
            proposal=proposal,
            amount=amount,
            basis=basis,
            contract_type=contract_type,
            currency=currency,
            duration=duration,
            duration_unit=duration_unit,
            symbol=symbol,
            signal_id=signal_id,
        )
        return await self.place_trade(payload, dry_run=bool(getattr(self.cfg, "agent_dry_run", True)))

    def build_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.fetch_candles,
                name="fetch_candles",
                description="Fetch OHLC candles from broker data layer",
                args_schema=FetchCandlesInput,
            ),
            StructuredTool.from_function(
                coroutine=self.compute_indicators,
                name="compute_indicators",
                description="Compute pandas-ta indicators from candles",
                args_schema=ComputeIndicatorsInput,
            ),
            StructuredTool.from_function(
                coroutine=self.evaluate_signal,
                name="evaluate_signal",
                description="Evaluate strategy signal from candles",
                args_schema=EvaluateSignalInput,
            ),
            StructuredTool.from_function(
                coroutine=self.get_account_state,
                name="get_account_state",
                description="Read account balance and open positions",
            ),
            StructuredTool.from_function(
                coroutine=self.place_trade_tool,
                name="place_trade",
                description="Place fixed-time trade through execution layer with code-side risk checks",
                args_schema=PlaceTradeInput,
            ),
        ]
