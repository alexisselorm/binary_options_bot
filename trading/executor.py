import logging
import asyncio
import pandas as pd
from typing import Dict, Optional
from trading.strategy import collect_rule_signals, generate_signals, RULE_STRATEGY_NAMES
from broker.deriv import DerivAPIWrapper
import random
import csv
import os
from datetime import datetime
# Record the result in confidence data
from trading.confidence import get_confidence, record_result

logger = logging.getLogger("bot.executor")


class TradeExecutor:
    def __init__(self, api: DerivAPIWrapper, cfg: any):
        self.api = api
        self.cfg = cfg
        self.valid_durations = []

        # ── log file setup ──────────────────────────────── #
        self.log_path = getattr(
            cfg, "trade_history_path", "logs/trade_history.csv"
        )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # write header once
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp", "contract_id", "strategy",
                        "signal", "stake",
                        "entry_tick", "exit_tick", "profit",
                        "status"
                    ]
                )

    async def run(self):
        granularity = self.cfg.timeframe * 60  # seconds
        # expiry_sec = self.cfg.expiry * 60      # trade duration in seconds
        expiry_sec = 60      # Temporary hardcoded value for testing

        self.valid_durations = await self.api.fetch_trading_durations(self.cfg.asset)
        if expiry_sec not in self.valid_durations:
            print(
                f"⛔ Expiry {expiry_sec}s not supported for {self.cfg.asset}. Valid: {self.valid_durations}")
            return

        print(f"✅ Starting trade executor loop (Granularity: {granularity}s)")

        while True:
            try:
                candle_resp = await self.api.get_candles(self.cfg.asset, 2000, granularity)
                candles = candle_resp.get("candles")
                if not candles:
                    print("⚠️ No candle data returned. Skipping this round.")
                    await asyncio.sleep(granularity)
                    continue

                print(f"Received {len(candles)} candles")
                df = pd.DataFrame(candles)

                balance = await self.api.get_balance()
                print(f"💰 Current balance: {balance}")
                if balance < self.cfg.min_balance:
                    raise RuntimeError(
                        f"⚠️ Balance too low: {balance} < {self.cfg.min_balance}")

                # use_ai = True if self.cfg.model_type != "rule" else False
                # strategy_names = [
                #     "sma_rsi",
                #     "macd_cross",
                #     "bollinger_bands",
                #     "adx_trend",
                #     "ichimoku_base_conversion",
                #     "breakout",
                #     "trend_follow",
                # ]

                # stra = random.choice(strategy_names)
                # signal, stake, conf = generate_signals(
                #     df, balance, use_ai=use_ai, strategy=stra)

                # Step 1: Loop through all strategies and collect signals (no confidence)
                votes = {}
                stakes = {}

                print("🔍 Evaluating strategy signals for ensemble agreement...")

                # Step 1: Collect signals and associated stakes
                import warnings

                for strat in RULE_STRATEGY_NAMES:
                    try:
                        with warnings.catch_warnings():
                            if strat == "_heikin_ashi":
                                warnings.simplefilter(
                                    "ignore", category=FutureWarning)
                                warnings.simplefilter(
                                    "ignore", category=UserWarning)

                            sig, stk, _ = generate_signals(
                                df, balance, use_ai=False, strategy=strat
                            )
                    except Exception as e:
                        print(f"💥 Strategy error [{strat}]: {e}")
                        sig, stk = None, None

                    if sig is not None:
                        print(
                            f"✅ Strategy '{strat}' voted for: {sig} with stake ${stk:.2f}")
                        votes.setdefault(sig, []).append(strat)
                        stakes[strat] = stk
                    else:
                        print(
                            f"❌ Strategy '{strat}' did not generate a signal.")
                signal, strats_agreed = collect_rule_signals(
                    df, balance, min_agree=4)

                if not signal:          # still no agreement → skip cycle
                    print("⚖️  No multi-strategy consensus. Waiting for next cycle.")
                    await asyncio.sleep(granularity)
                    continue

                print(f"🤝 Consensus direction = {signal}  "
                      f"from strategies: {strats_agreed}")

                # ----- NEW: choose the highest-confidence strategy among the agreers
                stra = None
                best_conf = -1.0
                stake = None

                # e.g. "macd,bollinger"
                for s in strats_agreed.split(", "):
                    # ← live score from JSON
                    current_conf = get_confidence(s)
                    if current_conf > best_conf:
                        # Re-compute stake for this specific strategy
                        _, stake_tmp, _ = generate_signals(
                            df, balance, use_ai=False, strategy=s
                        )
                        stra = s
                        best_conf = current_conf
                        stake = stake_tmp

                print(
                    f"🤝 Multi-strategy agreement → Signal: {signal}, Strategies: [{strats_agreed}], Strategy Used: {stra}, Stake: ${stake:.2f}"
                )

                if not signal:
                    print("🚫No trade signal generated. Waiting for next cycle.")
                    # wait longer for 3 minutes
                    await asyncio.sleep(granularity)
                    continue

                # print(
                #     f"📈 Strategy: {stra}.  Signal: {signal}. The signal is ")

                proposal_args = {
                    "proposal": 1,
                    "amount": stake,
                    # "amount": self.cfg.stake,
                    "basis": "stake",
                    "contract_type": signal,
                    "currency": "USD",
                    "duration": expiry_sec,
                    "duration_unit": "s",
                    "symbol": self.cfg.asset  # use configured asset
                }

                buy_result = await self.execute_trade(proposal_args)
                if not buy_result:
                    print("⚠️ Trade not executed after retrying.")
                    await asyncio.sleep(granularity)
                    continue

                contract_id = buy_result["buy"]["contract_id"]
                print(f"✅ Contract bought successfully | ID={contract_id}")

                # Wait for trade to complete
                finished = asyncio.Event()

                def on_update(msg: Dict):
                    self._on_contract_update(
                        msg, signal, contract_id, stra, stake)
                    poc = msg.get("proposal_open_contract", {})
                    if poc.get("is_sold"):
                        finished.set()

                subscription = await self.api.subscribe({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                    "subscribe": 1
                })
                subscription.subscribe(on_update)

                print("⏳ Waiting for trade to settle...")
                await finished.wait()
                print("✅ Trade settled. Resuming...")

            except Exception as e:
                print(f"💥 Unhandled error: {e}")
                await asyncio.sleep(granularity)

    async def execute_trade(self, proposal_args: dict) -> Optional[dict]:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            print(f"🚀 Attempt {attempt} to execute trade...")
            if attempt > 1:
                print("🔁 Retrying trade execution...")

            try:
                proposal = await self.api.proposal(proposal_args)

                if not proposal or "proposal" not in proposal:
                    raise Exception(
                        "Invalid proposal structure or empty response")

                proposal_data = proposal["proposal"]
                proposal_id = proposal_data["id"]
                price = proposal_data["ask_price"]

                print(f"✅ Proposal received: ID={proposal_id}, Price={price}")
                return await self.api.buy({"buy": proposal_id, "price": price})

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}] Trade execution failed: {e}")
                await asyncio.sleep(1)

        print("❌ All trade execution attempts failed.")
        return None

    def _on_contract_update(self, msg: Dict, signal: str, cid: str, stra,
                            stake):
        print(f"📩 Contract update received for ID={cid}")
        # print("Full message:")
        # print(msg)

        poc = msg.get("proposal_open_contract", {})
        if poc.get("is_sold"):
            profit = poc.get("profit")
            entry_price = poc.get("entry_tick")
            exit_price = poc.get("exit_tick")
            sell_price = poc.get("sell_price")
            sell_time = poc.get("sell_time")
            status = poc.get("status")
            # log *once* when closed
            self._log_trade(cid=cid,
                            strategy=stra,
                            signal=signal,
                            stake=stake,
                            entry=poc.get("entry_tick"),
                            exit_=poc.get("exit_tick"),
                            profit=poc.get("profit"),
                            status=poc.get("status"),
                            )

            record_result(
                strategy_name=stra,
                won=(profit > 0),
            )

            print(f"💰 Contract {cid} | Signal={signal}")
            print(f"🔹 Entry: {entry_price} | Exit: {exit_price}")
            print(
                f"🔸 Sold for: {sell_price} | Status: {status} | Profit/Loss: {profit}")
            print(f"🕒 Sell Time: {sell_time}")
            print("📉 Trade has been settled.")

    def _log_trade(self, *, cid: str, strategy: str, signal: str, stake: float, entry: float, exit_: float, profit: float, status: str,):
        """Append a single trade to CSV."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(timespec="seconds"),
                    cid,
                    strategy,
                    signal,
                    f"{stake:.2f}",
                    entry,
                    exit_,
                    profit,
                    status,
                ]
            )
        print(f"📝 Logged trade {cid} → {profit}")
