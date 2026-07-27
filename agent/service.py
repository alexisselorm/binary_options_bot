from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from broker.deriv import DerivAPIWrapper
from trading.executor import TradeExecutor
from utils.config import Config
from agent.provider import get_llm
from agent.schemas import AgentAction, DecisionOutput, DecisionRequest, ProposalArgs
from agent.tools import AgentToolRuntime, PlaceTradeInput

logger = logging.getLogger("bot.agent")


class TradingAgentService:
    def __init__(self, api: DerivAPIWrapper, executor: TradeExecutor, cfg: Config):
        self.cfg = cfg
        self.runtime = AgentToolRuntime(api=api, executor=executor, cfg=cfg)
        self.llm = get_llm(cfg)
        self.tools = self.runtime.build_tools()
        self.tool_map = {t.name: t for t in self.tools}

    async def _run_tool_loop(self, request: DecisionRequest) -> List[Any]:
        llm_tools = self.llm.bind_tools(self.tools)
        messages: List[Any] = [
            SystemMessage(
                content=(
                    "You are a trading orchestration agent. Use tools to gather candles, compute indicators, "
                    "evaluate signal, and account state. Only rely on tool outputs."
                )
            ),
            HumanMessage(
                content=(
                    f"symbol={request.symbol}, timeframe_minutes={request.timeframe_minutes}, "
                    f"strategy={request.strategy}, candle_count={request.candle_count}, expiry_seconds={request.expiry_seconds}. "
                    "Use tools in sequence and then stop."
                )
            ),
        ]

        for _ in range(int(getattr(self.cfg, "agent_max_tool_steps", 6))):
            ai_msg: AIMessage = await llm_tools.ainvoke(messages)
            messages.append(ai_msg)
            tool_calls = ai_msg.tool_calls or []
            if not tool_calls:
                break
            for call in tool_calls:
                tool = self.tool_map.get(call["name"])
                if not tool:
                    continue
                result = await tool.ainvoke(call["args"])
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
        return messages

    async def _build_structured_decision(self, request: DecisionRequest, transcript: List[Any]) -> DecisionOutput:
        schema_llm = self.llm.with_structured_output(DecisionOutput)
        retries = int(getattr(self.cfg, "agent_schema_retries", 2))
        transcript_dump = []
        for m in transcript:
            msg_type = m.__class__.__name__
            content = getattr(m, "content", "")
            transcript_dump.append({"type": msg_type, "content": str(content)})

        prompt = (
            "Return a valid DecisionOutput JSON object only. "
            "If signal is missing or confidence is low, set action=NO_TRADE. "
            f"Expiry must be {request.expiry_seconds} seconds and symbol must be {request.symbol}. "
            "Use confidence in range 0..1 and keep rationale short."
        )

        last_error = None
        for _ in range(retries + 1):
            try:
                return await schema_llm.ainvoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=json.dumps(transcript_dump)),
                    ]
                )
            except Exception as exc:  # schema or provider errors
                last_error = exc

        raise RuntimeError(f"Failed structured decision generation after retries: {last_error}")

    async def decide(self, request: DecisionRequest) -> DecisionOutput:
        dry_run = bool(getattr(self.cfg, "agent_dry_run", True))
        transcript = await self._run_tool_loop(request)
        decision = await self._build_structured_decision(request, transcript)
        decision.dry_run = dry_run

        existing_receipt = None
        for msg in transcript:
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "place_trade":
                try:
                    existing_receipt = json.loads(msg.content)
                except Exception:
                    existing_receipt = {"raw": msg.content}

        if decision.action == AgentAction.NO_TRADE:
            logger.info("final_decision=%s", decision.model_dump_json())
            return decision

        if not decision.proposal_args:
            decision.action = AgentAction.NO_TRADE
            decision.errors.append("proposal_args missing for TRADE action")
            logger.info("final_decision=%s", decision.model_dump_json())
            return decision

        if existing_receipt is not None:
            decision.trade_receipt = existing_receipt
            decision.tools_used = sorted(set(decision.tools_used + ["place_trade"]))
            logger.info("final_decision=%s", decision.model_dump_json())
            return decision

        proposal = decision.proposal_args.model_dump()
        proposal["symbol"] = request.symbol
        proposal["duration"] = request.expiry_seconds
        signal_id = f"{request.symbol}:{request.strategy}:{proposal['contract_type']}:{proposal['duration']}"

        try:
            trade_receipt = await self.runtime.place_trade(
                PlaceTradeInput(**proposal, signal_id=signal_id),
                dry_run=dry_run,
            )
            decision.trade_receipt = trade_receipt
            decision.tools_used = sorted(set(decision.tools_used + ["place_trade"]))
        except Exception as exc:
            decision.action = AgentAction.NO_TRADE
            decision.proposal_args = None
            decision.errors.append(str(exc))

        logger.info("final_decision=%s", decision.model_dump_json())
        return decision
