#!/usr/bin/env python3
"""
Estimate the cost of running a TradingAgents backtest.

Anthropic (default) — static estimate, no API calls:
    python scripts/estimate_backtest_cost.py --ticker AAPL

Anthropic — real token sample (runs one trading day, ~$0.20):
    python scripts/estimate_backtest_cost.py --ticker AAPL --sample

Ollama / local model (RunPod or local machine) — API cost is $0:
    python scripts/estimate_backtest_cost.py --ticker AAPL --provider ollama \
        --quick-model qwen3:latest --deep-model qwen3:latest

The paper (arXiv 2412.20138v7) tested Q1 2024 (Jan 1 – Mar 29) on AAPL, GOOGL,
and AMZN with ~11 LLM calls and 20+ tool calls per trading day.
"""

import argparse
import sys
from pathlib import Path

import yfinance as yf

# Allow running from the repo root: `python scripts/estimate_backtest_cost.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Anthropic pricing ($/million tokens), as of 2025
ANTHROPIC_PRICING = {
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-8":   {"input": 15.00, "output": 75.00},
}

# Model combos to display in the table (quick, deep)
MODEL_COMBOS = [
    ("claude-haiku-4-5",  "claude-haiku-4-5"),
    ("claude-haiku-4-5",  "claude-sonnet-4-6"),
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-sonnet-4-6", "claude-opus-4-8"),
]

# Static token budget anchored to paper footnote (~11 LLM calls, 20+ tool calls/day)
# Split: ~60% of tokens go through quick-think agents, ~40% through deep-think agents
STATIC_TOKENS_IN  = 62_000
STATIC_TOKENS_OUT = 12_000
QUICK_RATIO = 0.60
DEEP_RATIO  = 0.40


def count_trading_days(ticker: str, start_date: str, end_date: str) -> int:
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    return len(data)


def cost_per_day(
    tokens_in: int,
    tokens_out: int,
    quick_model: str,
    deep_model: str,
) -> float:
    qp = ANTHROPIC_PRICING.get(quick_model, {"input": 3.0, "output": 15.0})
    dp = ANTHROPIC_PRICING.get(deep_model,  {"input": 3.0, "output": 15.0})

    in_cost = (
        tokens_in * QUICK_RATIO * qp["input"] / 1_000_000
        + tokens_in * DEEP_RATIO  * dp["input"] / 1_000_000
    )
    out_cost = (
        tokens_out * QUICK_RATIO * qp["output"] / 1_000_000
        + tokens_out * DEEP_RATIO  * dp["output"] / 1_000_000
    )
    return in_cost + out_cost


def run_sample(ticker: str, sample_date: str, quick_model: str, deep_model: str) -> tuple[int, int]:
    """Run one real propagate() call and return (tokens_in, tokens_out)."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from cli.stats_handler import StatsCallbackHandler

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["quick_think_llm"] = quick_model
    config["deep_think_llm"]  = deep_model
    config["temperature"] = 0.1

    stats = StatsCallbackHandler()
    ta = TradingAgentsGraph(
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config=config,
        callbacks=[stats],
    )

    print(f"  Running sample analysis: {ticker} on {sample_date} ...")
    ta.propagate(ticker, sample_date)

    s = stats.get_stats()
    return s["tokens_in"], s["tokens_out"]


def print_ollama_table(
    ticker: str,
    start_date: str,
    end_date: str,
    n_days: int,
    quick_model: str,
    deep_model: str,
) -> None:
    """Print a RunPod / local-Ollama cost summary (API cost = $0)."""
    paper_targets = {
        "AAPL":  "CR +26.62% | AR +30.5% | SR 8.21 | MDD 0.91%",
        "GOOGL": "CR +24.36% | AR +27.58% | SR 6.39 | MDD 1.69%",
        "AMZN":  "CR +23.21% | AR +24.90% | SR 5.60 | MDD 2.11%",
    }

    # Rough timing: ~5–10 min per day on an 8B model on an RTX 4090
    min_hours = n_days * 5  / 60
    max_hours = n_days * 10 / 60
    runpod_cost_lo = min_hours * 0.74
    runpod_cost_hi = max_hours * 0.74

    print()
    print("=" * 70)
    print(f"  TradingAgents Backtest Cost Estimate  [Ollama / local model]")
    print("=" * 70)
    print(f"  Ticker      : {ticker}")
    print(f"  Period      : {start_date} → {end_date}")
    print(f"  Trading days: {n_days}")
    if ticker.upper() in paper_targets:
        print(f"  Paper target: {paper_targets[ticker.upper()]}")
    print()
    print(f"  LLM API cost         : $0.00  (models run locally via Ollama)")
    print(f"  Models               : quick={quick_model}  /  deep={deep_model}")
    print()
    print(f"  --- RunPod estimate (RTX 4090 @ ~$0.74/hr) ---")
    print(f"  Time per trading day : ~5–10 minutes  (8B model, GPU-accelerated)")
    print(f"  Est. total GPU time  : {min_hours:.1f} – {max_hours:.1f} hours")
    print(f"  Est. RunPod cost     : ${runpod_cost_lo:.2f} – ${runpod_cost_hi:.2f}")
    print()
    print(f"  Tip: use --resume so a restart doesn't re-run completed days.")
    print(f"  Tip: qwen3:14b (14B) is higher quality than qwen3:latest (8B)")
    print(f"       if your GPU has 16+ GB VRAM.")
    print("=" * 70)
    print()


def print_table(
    ticker: str,
    start_date: str,
    end_date: str,
    n_days: int,
    tokens_in: int,
    tokens_out: int,
    source: str,
) -> None:
    paper_targets = {
        "AAPL":  "CR +26.62% | AR +30.5% | SR 8.21 | MDD 0.91%",
        "GOOGL": "CR +24.36% | AR +27.58% | SR 6.39 | MDD 1.69%",
        "AMZN":  "CR +23.21% | AR +24.90% | SR 5.60 | MDD 2.11%",
    }

    print()
    print("=" * 70)
    print(f"  TradingAgents Backtest Cost Estimate")
    print("=" * 70)
    print(f"  Ticker     : {ticker}")
    print(f"  Period     : {start_date} → {end_date}")
    print(f"  Trading days: {n_days}")
    if ticker.upper() in paper_targets:
        print(f"  Paper target: {paper_targets[ticker.upper()]}")
    print(f"  Token budget: {tokens_in:,} in / {tokens_out:,} out per day  [{source}]")
    print()
    print(f"  {'Model combo (quick / deep)':<42}  {'$/day':>7}  {'Total':>9}")
    print(f"  {'-'*42}  {'-'*7}  {'-'*9}")

    for quick, deep in MODEL_COMBOS:
        cpd = cost_per_day(tokens_in, tokens_out, quick, deep)
        total = cpd * n_days
        label = f"{quick.replace('claude-','').replace('-20251001','')} / {deep.replace('claude-','').replace('-20251001','')}"
        print(f"  {label:<42}  ${cpd:>6.2f}  ${total:>8.2f}")

    print()
    print("  Recommendation: haiku-4-5 / haiku-4-5 to validate the pipeline (~$4),")
    print("  then haiku-4-5 / sonnet-4-6 for the full run (~$13).")
    print()
    print("  Run with --sample to measure real token usage from one live API call.")
    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate cost for a TradingAgents backtest (Anthropic or Ollama)."
    )
    parser.add_argument("--ticker",      default="AAPL",       help="Stock ticker (default: AAPL)")
    parser.add_argument("--start-date",  default="2024-01-01", help="Backtest start (default: paper's Q1 start)")
    parser.add_argument("--end-date",    default="2024-03-29", help="Backtest end (default: paper's Q1 end)")
    parser.add_argument("--provider",    default="anthropic",  choices=["anthropic", "ollama"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--sample-date", default="2024-02-15", help="Date used for --sample run")
    parser.add_argument("--quick-model", default=None,
                        help="Quick-think model (default: haiku-4-5 for anthropic, qwen3:latest for ollama)")
    parser.add_argument("--deep-model",  default=None,
                        help="Deep-think model (default: sonnet-4-6 for anthropic, qwen3:latest for ollama)")
    parser.add_argument("--sample", action="store_true",
                        help="Run one real trading day to measure actual token usage (Anthropic only)")
    args = parser.parse_args()

    # Apply per-provider model defaults
    if args.provider == "ollama":
        quick_model = args.quick_model or "qwen3:latest"
        deep_model  = args.deep_model  or "qwen3:latest"
    else:
        quick_model = args.quick_model or "claude-haiku-4-5"
        deep_model  = args.deep_model  or "claude-sonnet-4-6"

    print(f"Counting trading days for {args.ticker} ({args.start_date} → {args.end_date}) ...")
    n_days = count_trading_days(args.ticker, args.start_date, args.end_date)

    if args.provider == "ollama":
        print_ollama_table(
            args.ticker, args.start_date, args.end_date,
            n_days, quick_model, deep_model,
        )
        return

    # Anthropic path
    if args.sample:
        print(f"Sampling real token usage (this will call the Anthropic API) ...")
        tokens_in, tokens_out = run_sample(
            args.ticker, args.sample_date, quick_model, deep_model
        )
        source = f"measured on {args.sample_date}"
    else:
        tokens_in  = STATIC_TOKENS_IN
        tokens_out = STATIC_TOKENS_OUT
        source = "static estimate from paper footnote"

    print_table(
        args.ticker, args.start_date, args.end_date,
        n_days, tokens_in, tokens_out, source,
    )


if __name__ == "__main__":
    main()
