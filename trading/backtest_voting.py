import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from trading.strategy import generate_signals, add_indicators, RULE_BASED_STRATEGIES
from utils.config import Config

# Configuration
cfg = Config()
RAW_CSV = "data/historical_candles.csv"
df = pd.read_csv(RAW_CSV)
df = add_indicators(df)
SEQ_LEN = cfg.sequence_length
TRADE_DURATIONS = [1, 2, 3, 4, 5]  # Minutes until expiry
MIN_AGREEMENT = 3  # Minimum strategies needed for consensus


def collect_rule_signals(df_slice):
    """Collect signals from all rule-based strategies"""
    votes = {"CALL": [], "PUT": []}
    for name, strat_func in RULE_BASED_STRATEGIES.items():
        try:
            signal, _ = strat_func(df_slice)
            if signal in votes:
                votes[signal].append(name)
        except Exception:
            continue
    return votes


def run_voting_backtest(trade_duration=5):
    """Run voting backtest for specific trade duration"""
    balance = 1000.0
    equity_curve = [balance]
    win_log = []

    pbar = tqdm(range(SEQ_LEN, len(df) - max(TRADE_DURATIONS)),
                desc=f"Voting ({trade_duration}min)")

    for i in pbar:
        sliced = df.iloc[:i + 1].copy()
        votes = collect_rule_signals(sliced)

        # Determine strongest signal
        signal = None
        for direction, strategies in votes.items():
            if len(strategies) >= MIN_AGREEMENT:
                signal = direction
                break

        if not signal:
            equity_curve.append(balance)
            pbar.set_postfix_str("No consensus")
            continue

        stake = balance * 0.05
        future_close = df.iloc[i + trade_duration]["close"]
        current_close = df.iloc[i]["close"]
        outcome = "CALL" if future_close > current_close else "PUT"

        if signal == outcome:
            balance += stake * 0.9
            win_log.append(1)
            pbar.set_postfix_str(f"✅ Win ({signal})")
        else:
            balance -= stake
            win_log.append(0)
            pbar.set_postfix_str(f"❌ Loss ({signal})")

        equity_curve.append(balance)
        pbar.set_postfix({
            'balance': f"${balance:.2f}",
            'win_rate': f"{sum(win_log)/len(win_log)*100:.1f}%" if win_log else "0%"
        })

    total_trades = len(win_log)
    wins = sum(win_log)
    losses = total_trades - wins
    win_rate = wins / total_trades * 100 if total_trades else 0.0

    return equity_curve, balance, win_rate, total_trades, wins, losses


# Main execution
os.makedirs("backtest_charts/voting_system", exist_ok=True)
all_results = []

for duration in tqdm(TRADE_DURATIONS, desc="Testing durations"):
    equity_curve, final_balance, win_rate, total_trades, wins, losses = run_voting_backtest(
        duration)

    results = {
        "Duration": f"{duration}min",
        "Final Balance": round(final_balance, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Return (%)": round((final_balance - 1000) / 10, 2),
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Profit Factor": round((wins * 0.9) / losses, 2) if losses else float('inf')
    }
    all_results.append(results)

    # Enhanced plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)

    stats_text = (
        f"Voting System ({duration}min expiry)\n"
        f"Minimum Agreement: {MIN_AGREEMENT} strategies\n"
        f"Final Balance: ${final_balance:,.2f}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Trades: {total_trades} (W:{wins} L:{losses})\n"
        f"Return: {((final_balance-1000)/10):.1f}%\n"
        f"Profit Factor: {results['Profit Factor']:.2f}"
    )

    plt.title(f"Voting System Performance - {duration}min Trades")
    plt.xlabel("Trade Number")
    plt.ylabel("Balance ($)")
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"backtest_charts/voting_system/{duration}min.png", dpi=120)
    plt.close()

# Save comprehensive results
summary_df = pd.DataFrame(all_results)
summary_df.to_csv("backtest_charts/voting_system/summary.csv", index=False)
summary_df.to_excel("backtest_charts/voting_system/summary.xlsx", index=False)

# Print formatted results
print("\n📊 Voting System Performance Summary:")
print(summary_df.to_string(index=False))
print("\n🔍 Best Performing Duration:")
print(summary_df.loc[summary_df['Final Balance'].idxmax()].to_string())
