import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from trading.strategy import generate_signals, add_indicators, RULE_STRATEGY_NAMES
from utils.config import Config

# --------------------------------------------------------------------
# CONFIG & DATA
# --------------------------------------------------------------------
cfg = Config()
RAW_CSV = "data/historical_candles.csv"
df = pd.read_csv(RAW_CSV)
df = add_indicators(df)

SEQ_LEN = cfg.sequence_length
STRATEGIES = RULE_STRATEGY_NAMES
_current_strategy = None
TRADE_DURATIONS = [1, 2, 3, 4, 5]  # Minutes until expiry

# --------------------------------------------------------------------
# BACKTEST ENGINE
# --------------------------------------------------------------------


def run_backtest(trade_duration=1):
    """Run backtest for specific trade duration"""
    balance = 1000.0
    equity_curve = [balance]
    win_log = []

    trade_range = range(SEQ_LEN, len(df) - max(TRADE_DURATIONS))
    pbar = tqdm(trade_range, desc=f"{trade_duration}min trades", leave=False)

    for i in pbar:
        try:
            sliced = df.iloc[:i + 1].copy()
            signal, stake, conf = generate_signals(
                sliced, balance, use_ai=False, strategy=_current_strategy
            )

            if signal is None or stake is None:
                equity_curve.append(balance)
                continue

            future_close = df.iloc[i + trade_duration]["close"]
            current_close = df.iloc[i]["close"]
            outcome = "CALL" if future_close > current_close else "PUT"

            if signal == outcome:
                balance += stake * 0.9
                win_log.append(1)
            else:
                balance -= stake
                win_log.append(0)

            equity_curve.append(balance)

            pbar.set_postfix({
                'balance': f"${balance:.2f}",
                'win_rate': f"{sum(win_log)/len(win_log)*100:.1f}%" if win_log else "0%"
            })

        except Exception as e:
            print(f"⚠️ Error during {trade_duration}min backtest: {e}")
            equity_curve.append(balance)
            continue

    total_trades = len(win_log)
    wins = sum(win_log)
    losses = total_trades - wins
    win_rate = wins / total_trades * 100 if total_trades else 0.0

    return equity_curve, balance, win_rate, total_trades, wins, losses


# --------------------------------------------------------------------
# MAIN LOOP WITH IMPROVED DIRECTORY STRUCTURE
# --------------------------------------------------------------------
all_results = []

for strat in tqdm(STRATEGIES, desc="Strategies"):
    _current_strategy = strat

    # Create strategy-specific directory
    strategy_dir = f"backtest_charts/{strat}"
    os.makedirs(strategy_dir, exist_ok=True)

    strategy_results = []

    for duration in tqdm(TRADE_DURATIONS, desc="Durations", leave=False):
        equity_curve, final_balance, win_rate, total_trades, wins, losses = run_backtest(
            duration)

        # Save results
        strategy_results.append({
            "Strategy": strat,
            "Duration": f"{duration}min",
            "Final Balance": round(final_balance, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Return (%)": round((final_balance - 1000) / 1000 * 100, 2),
            "Total Trades": total_trades,
            "Wins": wins,
            "Losses": losses,
            "Profit Factor": round((wins * 0.9) / losses, 2) if losses else float('inf')
        })

        # Plot and save to strategy directory
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)

        stats_text = (
            f"Strategy: {strat}\n"
            f"Duration: {duration}min\n"
            f"Final: ${final_balance:.2f}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Trades: {total_trades} (W:{wins} L:{losses})\n"
            f"Return: {((final_balance-1000)/1000*100):.1f}%"
        )

        plt.title(f"{strat} - {duration}min expiry")
        plt.xlabel("Trade Number")
        plt.ylabel("Balance ($)")
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(f"{strategy_dir}/{duration}min.png", dpi=120)
        plt.close()

    # Save strategy-specific CSV
    pd.DataFrame(strategy_results).to_csv(
        f"{strategy_dir}/results.csv", index=False)
    all_results.extend(strategy_results)

# --------------------------------------------------------------------
# GLOBAL RESULTS SUMMARY
# --------------------------------------------------------------------
print("\n📈 Comprehensive Backtest Summary:")

# Create master results directory
os.makedirs("backtest_charts/_SUMMARY", exist_ok=True)

# Save complete results
summary_df = pd.DataFrame(all_results)
summary_df.to_csv("backtest_charts/_SUMMARY/full_results.csv", index=False)
summary_df.to_excel("backtest_charts/_SUMMARY/full_results.xlsx", index=False)

# Generate and save summary tables
with open("backtest_charts/_SUMMARY/summary_report.txt", "w") as f:
    f.write("STRATEGY PERFORMANCE SUMMARY\n")
    f.write("="*40 + "\n\n")

    # Best performers by duration
    for duration in TRADE_DURATIONS:
        dur_df = summary_df[summary_df["Duration"] == f"{duration}min"]
        top_3 = dur_df.nlargest(3, "Final Balance")

        f.write(f"TOP PERFORMERS ({duration}min trades):\n")
        f.write(top_3[["Strategy", "Final Balance",
                "Win Rate (%)", "Return (%)"]].to_string(index=False))
        f.write("\n\n")

    # Best overall strategies
    f.write("OVERALL BEST STRATEGIES:\n")
    f.write(summary_df.groupby("Strategy")[
            "Final Balance"].mean().nlargest(5).to_string())
    f.write("\n")

print("✅ Saved complete results to:")
print("  - backtest_charts/_SUMMARY/full_results.csv")
print("  - backtest_charts/_SUMMARY/full_results.xlsx")
print("  - backtest_charts/_SUMMARY/summary_report.txt")

# Print quick summary to console
print("\n🔍 Quick Performance Summary:")
print(summary_df.groupby(["Strategy", "Duration"])[
      "Final Balance"].mean().unstack().to_string())
