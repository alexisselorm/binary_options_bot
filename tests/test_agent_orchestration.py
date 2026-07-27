import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

fake_strategy = types.ModuleType("trading.strategy")
fake_strategy.add_indicators = lambda df: df
fake_strategy.generate_signals = lambda df, balance, use_ai=False, strategy="sma_rsi": (None, None, None)
sys.modules["trading.strategy"] = fake_strategy

fake_executor = types.ModuleType("trading.executor")
fake_executor.TradeExecutor = object
sys.modules["trading.executor"] = fake_executor

from agent.schemas import AgentAction, DecisionOutput, DecisionRequest, ProposalArgs
from agent.service import TradingAgentService
from agent.tools import AgentToolRuntime, PlaceTradeInput


class TestAgentTools(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api = SimpleNamespace()
        self.executor = SimpleNamespace()
        self.cfg = SimpleNamespace(
            asset="R_100",
            max_stake=10.0,
            agent_allowed_symbols=["R_100"],
            agent_max_concurrent_positions=1,
            agent_max_stake=10.0,
        )
        self.runtime = AgentToolRuntime(self.api, self.executor, self.cfg)

    async def test_fetch_candles_tool(self):
        self.api.get_candles = AsyncMock(return_value={"candles": [{"open": 1, "close": 2}]})
        out = await self.runtime.fetch_candles("R_100", 10, 60)
        self.assertIn("candles", out)

    @patch("agent.tools.add_indicators")
    async def test_compute_indicators_tool(self, mock_add):
        import pandas as pd
        mock_add.return_value = pd.DataFrame([{"epoch": 1, "close": 2, "RSI_14": 50}])
        out = await self.runtime.compute_indicators([{"epoch": 1, "open": 1, "high": 2, "low": 1, "close": 2}])
        self.assertIn("candles_with_indicators", out)

    @patch("agent.tools.generate_signals")
    async def test_evaluate_signal_tool(self, mock_gen):
        mock_gen.return_value = ("CALL", 5.0, 0.8)
        out = await self.runtime.evaluate_signal(
            [{"epoch": 1, "open": 1, "high": 2, "low": 1, "close": 2}],
            100.0,
            "sma_rsi",
        )
        self.assertEqual(out["signal"], "CALL")

    async def test_dry_run_prevents_real_trade(self):
        self.api.get_balance = AsyncMock(return_value=100.0)
        self.api.get_open_positions = AsyncMock(return_value=[])
        self.executor.execute_trade = AsyncMock(return_value={"buy": {"contract_id": "1"}})

        args = PlaceTradeInput(
            proposal=1,
            amount=5,
            basis="stake",
            contract_type="CALL",
            currency="USD",
            duration=60,
            duration_unit="s",
            symbol="R_100",
            signal_id="s1",
        )
        out = await self.runtime.place_trade(args, dry_run=True)
        self.assertTrue(out["dry_run"])
        self.executor.execute_trade.assert_not_awaited()


class _FakeSchemaRunner:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    async def ainvoke(self, _messages):
        item = self.responses[self.index]
        self.index += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeLLM:
    def __init__(self, schema_responses):
        self.schema = _FakeSchemaRunner(schema_responses)

    def bind_tools(self, _tools):
        class _ToolLLM:
            async def ainvoke(self, _messages):
                return AIMessage(content="done", tool_calls=[])
        return _ToolLLM()

    def with_structured_output(self, _schema):
        return self.schema


class TestAgentService(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_output_retried(self):
        cfg = SimpleNamespace(agent_schema_retries=2, agent_max_tool_steps=1, agent_dry_run=True)
        api = SimpleNamespace()
        executor = SimpleNamespace()

        good = DecisionOutput(
            action=AgentAction.NO_TRADE,
            confidence=0.5,
            rationale="No valid signal",
            tools_used=[],
            dry_run=True,
        )
        service = TradingAgentService.__new__(TradingAgentService)
        service.cfg = cfg
        service.runtime = MagicMock()
        service.tools = []
        service.tool_map = {}
        service.llm = _FakeLLM([ValueError("bad schema"), good])

        out = await service._build_structured_decision(
            DecisionRequest(symbol="R_100", timeframe_minutes=1, strategy="sma_rsi", candle_count=20, expiry_seconds=60),
            [],
        )
        self.assertEqual(out.action, AgentAction.NO_TRADE)

    async def test_end_to_end_decision_maps_to_place_trade(self):
        cfg = SimpleNamespace(agent_dry_run=False)
        service = TradingAgentService.__new__(TradingAgentService)
        service.cfg = cfg
        service.runtime = MagicMock()
        service.llm = MagicMock()
        service.tools = []
        service.tool_map = {}
        service._run_tool_loop = AsyncMock(return_value=[])
        service._build_structured_decision = AsyncMock(
            return_value=DecisionOutput(
                action=AgentAction.TRADE,
                proposal_args=ProposalArgs(
                    proposal=1,
                    amount=5,
                    basis="stake",
                    contract_type="CALL",
                    currency="USD",
                    duration=120,
                    duration_unit="s",
                    symbol="OTHER",
                ),
                confidence=0.8,
                rationale="Signal aligned",
                tools_used=["fetch_candles", "evaluate_signal"],
                dry_run=False,
            )
        )
        service.runtime.place_trade = AsyncMock(return_value={"buy": {"contract_id": "abc"}})

        req = DecisionRequest(symbol="R_100", timeframe_minutes=1, strategy="sma_rsi", candle_count=200, expiry_seconds=60)
        out = await TradingAgentService.decide(service, req)

        self.assertEqual(out.action, AgentAction.TRADE)
        self.assertIsNotNone(out.trade_receipt)
        call_input = service.runtime.place_trade.await_args.args[0]
        self.assertEqual(call_input.symbol, "R_100")
        self.assertEqual(call_input.duration, 60)


if __name__ == "__main__":
    unittest.main()
