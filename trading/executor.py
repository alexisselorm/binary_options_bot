import logging
import asyncio
import time
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
        self.consecutive_losses = 0
        self.martingale_multiplier = 2.0
        self.wait_on_loss = False

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
        expiry_sec = 60  # Temporary hardcoded value for testing

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

                if self.cfg.model_type != "rule":
                    print("🤖 Running AI-based strategy (LLM)…")
                    signal, stake, conf = generate_signals(
                        df, balance, use_ai=True, strategy=None)
                    if not signal:
                        print("🚫 No AI signal generated. Skipping.")
                        await asyncio.sleep(granularity)
                        continue

                    stra = "llm"
                    print(
                        f"🧠 LLM signal={signal}  conf={conf:.2f}  stake={stake:.2f}")

                else:
                    # RULE-BASED STRATEGY ENSEMBLE BLOCK
                    votes = {}
                    stakes = {}

                    print("🔍 Evaluating strategy signals for ensemble agreement…")
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

                    if not signal:
                        print(
                            "⚖️  No multi-strategy consensus. Waiting for next cycle.")
                        await asyncio.sleep(granularity*3)
                        continue
                    print(
                        f"🤝 Consensus direction = {signal}  from strategies: {strats_agreed}")

                    stra = None
                    best_conf = -1.0
                    stake = None

                    for s in strats_agreed.split(", "):
                        current_conf = get_confidence(s)
                        if current_conf > best_conf:
                            _, stake_tmp, _ = generate_signals(
                                df, balance, use_ai=False, strategy=s)
                            stra = s
                            best_conf = current_conf
                            stake = stake_tmp

                    print(
                        f"🤝 Multi-strategy agreement → Signal: {signal}, Strategies: [{strats_agreed}], Strategy Used: {stra}, Stake: ${stake:.2f}")

                    strategies_list = strats_agreed.split(", ")
                    strats_confidence = [get_confidence(
                        strategy) for strategy in strategies_list]
                    conf_average = sum(strats_confidence) / \
                        len(strats_confidence)
                    print(
                        f"Averagely strategies, [{strats_agreed}] are: {conf_average}% confident")

                    if conf_average < self.cfg.signal_threshold:
                        print("Not confident enough, skipping this trade")
                        print(
                            f"Confidence Average: {conf_average}, Threshold:{self.cfg.signal_threshold}")
                        await asyncio.sleep(granularity*3)
                        continue

                    base_stake = stake
                    if self.cfg.martingale_mode == "on" and self.cfg.max_consecutive_losses > self.consecutive_losses:
                        # Cap at 3 steps
                        steps = min(self.consecutive_losses, 2)
                        stake_multiplier = self.martingale_multiplier ** steps
                        adjusted_stake = base_stake * stake_multiplier

                        # Ensure stake doesn't exceed 95% of balance
                        max_stake = balance * 0.95
                        stake = min(adjusted_stake, max_stake)

                        print(
                            f"♠️ Martingale activated (step={steps}, multiplier={stake_multiplier:.2f})")
                        print(
                            f"   Base stake: ${base_stake:.2f} → Adjusted stake: ${stake:.2f}")

                proposal_args = {
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": signal,
                    "currency": "USD",
                    "duration": expiry_sec,
                    "duration_unit": "s",
                    "symbol": self.cfg.asset
                }

                buy_result = await self.execute_trade(proposal_args)
                if not buy_result:
                    print("⚠️ Trade not executed after retrying.")
                    await asyncio.sleep(granularity)
                    continue

                contract_id = buy_result["buy"]["contract_id"]
                print(f"✅ Contract bought successfully | ID={contract_id}")

                finished = asyncio.Event()

                def on_update(msg: Dict):
                    self._on_contract_update(
                        msg, signal, contract_id, strategies_list, stake)
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
                if self.wait_on_loss:
                    print("⏳ Waiting for 5 minutes cooldown after loss...")
                    await asyncio.sleep(300)
                    self.wait_on_loss = False

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

    def _on_contract_update(self, msg: Dict, signal: str, cid: str, strategies_list, stake):
        print(f"📩 Contract update received for ID={cid}")

        poc = msg.get("proposal_open_contract", {})
        if poc.get("is_sold"):
            profit = poc.get("profit")
            if profit <= 0:
                self.consecutive_losses += 1

            else:
                self.consecutive_losses = 0
            entry_price = poc.get("entry_tick")
            exit_price = poc.get("exit_tick")
            sell_price = poc.get("sell_price")
            sell_time = poc.get("sell_time")
            status = poc.get("status")
            self._log_trade(cid=cid, strategy=strategies_list, signal=signal, stake=stake,
                            entry=entry_price, exit_=exit_price,
                            profit=profit, status=status)

            record_result(strategy_names=strategies_list, won=(profit > 0))

            print(f"💰 Contract {cid} | Signal={signal}")
            print(f"🔹 Entry: {entry_price} | Exit: {exit_price}")
            martingale_log = ''
            if self.cfg.martingale_mode == 'on':
                martingale_log += f"Martingale {'won' if profit > 0 else 'lost'} at step {self.consecutive_losses} |"

            print(
                f"🔸 Sold for: {sell_price} | Status: {status} | Profit/Loss: {profit} | {martingale_log}")
            print(f"🕒 Sell Time: {sell_time}")
            print("📉 Trade has been settled.")

            # Wait for 5 minutes
            if profit < 0:
                print(
                    "🔻 Trade resulted in a loss. Will wait for 5 minutes before next trade.")
                self.wait_on_loss = True
            else:
                self.wait_on_loss = False

    def _log_trade(self, *, cid: str, strategy: str, signal: str, stake: float, entry: float, exit_: float, profit: float, status: str):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(timespec="seconds"),
                cid, strategy, signal,
                f"{stake:.2f}", entry, exit_, profit, status
            ])
        print(f"📝 Logged trade {cid} → {profit}")
