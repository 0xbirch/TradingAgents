#!/usr/bin/env python3
"""
Backtest runner that faithfully mirrors the TradingAgents paper experiments.

Paper: arXiv 2412.20138v7
Setup: Q1 2024 (Jan 1 – Mar 29), single ticker, daily decisions, no transaction costs.
Results: AAPL +26.62% CR / SR 8.21 / MDD 0.91%  (GPT-4o + o1-preview)

Usage (validate pipeline on 5 days first):
    python scripts/backtest.py --ticker AAPL --start-date 2024-01-02 --end-date 2024-01-08

Usage (full paper period, resume-safe):
    python scripts/backtest.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-03-29 --resume

Usage (cheaper model combo):
    python scripts/backtest.py --ticker AAPL --quick-model claude-haiku-4-5 --deep-model claude-haiku-4-5

Usage (Ollama local model — zero API cost, run on RunPod or local GPU):
    python scripts/backtest.py --ticker AAPL --provider ollama \
        --quick-model qwen3:latest --deep-model qwen3:latest --resume
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.stats_handler import StatsCallbackHandler

# Anthropic pricing ($/million tokens), 2025
ANTHROPIC_PRICING = {
    "claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-8":   {"input": 15.00, "output": 75.00},
}

SIGNAL_TO_POSITION = {
    "Buy":        1,
    "Overweight": 1,
    "Hold":       None,   # None = no change
    "Underweight": 0,
    "Sell":       0,
}


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def get_trading_days(ticker: str, start: str, end: str) -> list[str]:
    """Return list of trading day date strings from yfinance."""
    data = yf.Ticker(ticker).history(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in data.index]


def get_ohlcv(ticker: str, start: str, end: str):
    """Return OHLCV DataFrame with DatetimeIndex."""
    data = yf.Ticker(ticker).history(start=start, end=end)
    data.index = data.index.tz_localize(None)
    return data


# ---------------------------------------------------------------------------
# Decision loop
# ---------------------------------------------------------------------------

def load_existing_decisions(csv_path: Path) -> dict[str, dict]:
    """Load decisions already recorded in the CSV (for --resume)."""
    decisions = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                decisions[row["date"]] = row
    return decisions


def append_decision_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "decision", "tokens_in", "tokens_out"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_decision_loop(
    ticker: str,
    trading_days: list[str],
    existing: dict[str, dict],
    ta: TradingAgentsGraph,
    stats: StatsCallbackHandler,
    csv_path: Path,
    resume: bool,
) -> dict[str, str]:
    """Run propagate() for each day not already in CSV. Returns {date: decision}."""
    decisions = {d: existing[d]["decision"] for d in existing}

    pending = [d for d in trading_days if d not in existing] if resume else trading_days
    total = len(pending)

    for i, date in enumerate(pending, 1):
        # Reset stats so we count only this call
        with stats._lock:
            stats.tokens_in = 0
            stats.tokens_out = 0

        print(f"[{i:3d}/{total}] {date} ", end="", flush=True)
        try:
            _, signal = ta.propagate(ticker, date)
        except Exception as e:
            signal = "Hold"
            print(f"ERROR ({e}), defaulting to Hold")
        else:
            s = stats.get_stats()
            tok_in, tok_out = s["tokens_in"], s["tokens_out"]
            print(f"→ {signal:<12}  tokens: {tok_in//1000}K in / {tok_out//1000}K out")

        s = stats.get_stats()
        append_decision_row(csv_path, {
            "date":       date,
            "decision":   signal,
            "tokens_in":  stats.tokens_in,
            "tokens_out": stats.tokens_out,
        })
        decisions[date] = signal

    return decisions


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(
    decisions: dict[str, str],
    ohlcv,
    initial_capital: float,
) -> list[dict]:
    """
    Simulate a binary long/cash portfolio.
    Trade execution: at next day's Open (no look-ahead bias).
    Portfolio value tracked at each day's Close.
    """
    cash, shares, position = float(initial_capital), 0.0, 0
    records = []
    dates = list(ohlcv.index)
    n = len(dates)

    for i, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        decision = decisions.get(date_str, "Hold")
        target = SIGNAL_TO_POSITION.get(decision)   # 1, 0, or None

        # Execute at next day's open
        if target is not None and target != position and i + 1 < n:
            exec_price = float(ohlcv["Open"].iloc[i + 1])
            portfolio_val = cash + shares * exec_price
            if target == 1:
                shares, cash = portfolio_val / exec_price, 0.0
            else:
                cash, shares = portfolio_val, 0.0
            position = target

        value = cash + shares * float(ohlcv["Close"].iloc[i])
        records.append({
            "date":     date_str,
            "value":    value,
            "decision": decision,
            "position": position,
        })

    return records


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def simulate_buy_and_hold(ohlcv, initial_capital: float) -> list[dict]:
    first_open = float(ohlcv["Open"].iloc[0])
    shares = initial_capital / first_open
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": shares * float(ohlcv["Close"].iloc[i])}
        for i, d in enumerate(ohlcv.index)
    ]


def simulate_sma(ohlcv, initial_capital: float, short: int = 50, long: int = 200) -> list[dict]:
    """Buy when SMA(short) > SMA(long), else cash. Next-open execution."""
    closes = ohlcv["Close"].values.astype(float)
    opens  = ohlcv["Open"].values.astype(float)
    n = len(closes)

    def sma(prices, window, i):
        start = max(0, i - window + 1)
        return float(np.mean(prices[start:i + 1]))

    cash, shares, position = float(initial_capital), 0.0, 0
    records = []

    for i in range(n):
        s_short = sma(closes, short, i)
        s_long  = sma(closes, long,  i)
        target = 1 if s_short > s_long else 0

        if target != position and i + 1 < n:
            exec_price = opens[i + 1]
            portfolio_val = cash + shares * exec_price
            if target == 1:
                shares, cash = portfolio_val / exec_price, 0.0
            else:
                cash, shares = portfolio_val, 0.0
            position = target

        value = cash + shares * closes[i]
        records.append({
            "date":  ohlcv.index[i].strftime("%Y-%m-%d"),
            "value": value,
        })

    return records


# ---------------------------------------------------------------------------
# Metrics — exact formulas from paper Appendix S1.2
# ---------------------------------------------------------------------------

def calc_metrics(records: list[dict], initial_capital: float, risk_free_rate: float = 0.0) -> dict:
    values = [r["value"] for r in records]
    v_start = initial_capital
    v_end   = values[-1]
    n_days  = len(values)
    n_years = n_days / 252.0

    # CR (Eq S1)
    cr = (v_end - v_start) / v_start * 100.0

    # AR (Eq S2)
    ar = ((v_end / v_start) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else 0.0

    # Daily returns
    daily_returns = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] != 0
    ]

    # SR (Eq S3) — paper uses daily returns, no annualisation factor stated
    if len(daily_returns) > 1:
        mean_r = float(np.mean(daily_returns)) - risk_free_rate / 252.0
        std_r  = float(np.std(daily_returns, ddof=1))
        sr = mean_r / std_r if std_r > 0 else 0.0
    else:
        sr = 0.0

    # MDD (Eq S4)
    peak = values[0]
    mdd  = 0.0
    for v in values:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak * 100.0
        if drawdown > mdd:
            mdd = drawdown

    return {"cr": cr, "ar": ar, "sr": sr, "mdd": mdd}


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def estimate_cost(
    total_tokens_in: int,
    total_tokens_out: int,
    quick_model: str,
    deep_model: str,
) -> float:
    qp = ANTHROPIC_PRICING.get(quick_model, {"input": 3.0, "output": 15.0})
    dp = ANTHROPIC_PRICING.get(deep_model,  {"input": 3.0, "output": 15.0})
    # 60% quick / 40% deep split
    in_cost  = (total_tokens_in  * 0.6 * qp["input"]  + total_tokens_in  * 0.4 * dp["input"])  / 1_000_000
    out_cost = (total_tokens_out * 0.6 * qp["output"] + total_tokens_out * 0.4 * dp["output"]) / 1_000_000
    return in_cost + out_cost


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(
    ticker: str,
    start_date: str,
    end_date: str,
    ta_metrics: dict,
    bnh_metrics: dict,
    sma_metrics: dict,
    total_in: int,
    total_out: int,
    quick_model: str,
    deep_model: str,
    output_path: Path,
    provider: str = "anthropic",
) -> None:
    paper_targets = {
        "AAPL":  ("26.62", "30.5",  "8.21", "0.91"),
        "GOOGL": ("24.36", "27.58", "6.39", "1.69"),
        "AMZN":  ("23.21", "24.90", "5.60", "2.11"),
    }

    print()
    print("=" * 72)
    print(f"  BACKTEST RESULTS: {ticker} | {start_date} → {end_date}")
    print("=" * 72)
    print(f"  {'Strategy':<22} {'CR%':>8} {'AR%':>8} {'SR':>8} {'MDD%':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TradingAgents':<22} {ta_metrics['cr']:>+8.2f} {ta_metrics['ar']:>+8.2f} {ta_metrics['sr']:>8.2f} {ta_metrics['mdd']:>8.2f}")
    print(f"  {'Buy & Hold':<22} {bnh_metrics['cr']:>+8.2f} {bnh_metrics['ar']:>+8.2f} {bnh_metrics['sr']:>8.2f} {bnh_metrics['mdd']:>8.2f}")
    print(f"  {'SMA 50/200':<22} {sma_metrics['cr']:>+8.2f} {sma_metrics['ar']:>+8.2f} {sma_metrics['sr']:>8.2f} {sma_metrics['mdd']:>8.2f}")

    if ticker.upper() in paper_targets:
        t = paper_targets[ticker.upper()]
        print()
        print(f"  Paper targets (GPT-4o + o1-preview):")
        print(f"    CR {t[0]}% | AR {t[1]}% | SR {t[2]} | MDD {t[3]}%")
        print(f"  (Divergence expected — paper used OpenAI models)")

    print()
    print(f"  Token usage: {total_in:,} in / {total_out:,} out")
    if provider == "ollama":
        print(f"  Est. API cost: $0.00  (local Ollama model — {quick_model} / {deep_model})")
    else:
        cost = estimate_cost(total_in, total_out, quick_model, deep_model)
        print(f"  Est. API cost ({quick_model.replace('claude-','')} / {deep_model.replace('claude-','')}): ${cost:.2f}")
    print(f"  Results saved to: {output_path}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce TradingAgents paper backtest (arXiv 2412.20138v7)."
    )
    parser.add_argument("--ticker",            default="AAPL",
                        help="Single ticker (default: AAPL — paper's best result)")
    parser.add_argument("--start-date",        default="2024-01-01",
                        help="Backtest start date (paper: 2024-01-01)")
    parser.add_argument("--end-date",          default="2024-03-29",
                        help="Backtest end date (paper: 2024-03-29)")
    parser.add_argument("--initial-capital",   type=float, default=100_000,
                        help="Starting portfolio value in USD (default: 100000)")
    parser.add_argument("--provider",          default="anthropic",
                        help="LLM provider: anthropic or ollama (default: anthropic)")
    parser.add_argument("--base-url",          default=None,
                        help="Ollama base URL (default: http://localhost:11434/v1 or $OLLAMA_BASE_URL)")
    parser.add_argument("--quick-model",       default=None,
                        help="Quick-think model (default: haiku-4-5 for anthropic, qwen3:latest for ollama)")
    parser.add_argument("--deep-model",        default=None,
                        help="Deep-think model (default: sonnet-4-6 for anthropic, qwen3:latest for ollama)")
    parser.add_argument("--max-debate-rounds", type=int, default=1)
    parser.add_argument("--analysts",          nargs="+",
                        default=["market", "social", "news", "fundamentals"])
    parser.add_argument("--temperature",       type=float, default=0.1,
                        help="LLM temperature (lower = more reproducible)")
    parser.add_argument("--output",            default=None,
                        help="CSV file to record decisions (default: auto-named)")
    parser.add_argument("--resume",            action="store_true",
                        help="Skip dates already recorded in the CSV")
    args = parser.parse_args()

    # Auto-name output CSV
    if args.output is None:
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        safe_ticker = args.ticker.replace("/", "-")
        start_str = args.start_date.replace("-", "")
        end_str   = args.end_date.replace("-", "")
        csv_path = out_dir / f"backtest_{safe_ticker}_{start_str}_{end_str}.csv"
    else:
        csv_path = Path(args.output)

    # Apply per-provider model defaults
    import os
    if args.provider == "ollama":
        quick_model = args.quick_model or "qwen3:latest"
        deep_model  = args.deep_model  or "qwen3:latest"
    else:
        quick_model = args.quick_model or "claude-haiku-4-5"
        deep_model  = args.deep_model  or "claude-sonnet-4-6"

    # Load any existing decisions (for --resume)
    existing = load_existing_decisions(csv_path)
    if args.resume and existing:
        print(f"Resuming: {len(existing)} dates already recorded in {csv_path}")

    # Get full list of trading days
    print(f"Fetching trading calendar for {args.ticker} ({args.start_date} → {args.end_date}) ...")
    trading_days = get_trading_days(args.ticker, args.start_date, args.end_date)
    print(f"Trading days: {len(trading_days)}")

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"]            = args.provider
    config["quick_think_llm"]         = quick_model
    config["deep_think_llm"]          = deep_model
    config["max_debate_rounds"]       = args.max_debate_rounds
    config["max_risk_discuss_rounds"] = 1
    config["temperature"]             = args.temperature
    # Resolve Ollama base URL: CLI flag > env var > framework default (localhost:11434)
    if args.provider == "ollama":
        config["backend_url"] = (
            args.base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1"
        )

    # Init graph + stats handler
    stats = StatsCallbackHandler()
    ta = TradingAgentsGraph(
        selected_analysts=args.analysts,
        debug=False,
        config=config,
        callbacks=[stats],
    )

    # Seed total token counts from any previously completed days
    total_in  = sum(int(r.get("tokens_in",  0)) for r in existing.values())
    total_out = sum(int(r.get("tokens_out", 0)) for r in existing.values())

    # --- Decision loop ---
    print(f"\nRunning decision loop → {csv_path}\n")
    decisions = run_decision_loop(
        ticker=args.ticker,
        trading_days=trading_days,
        existing=existing,
        ta=ta,
        stats=stats,
        csv_path=csv_path,
        resume=args.resume,
    )

    # Accumulate tokens from this session
    all_rows = load_existing_decisions(csv_path)
    total_in  = sum(int(r.get("tokens_in",  0)) for r in all_rows.values())
    total_out = sum(int(r.get("tokens_out", 0)) for r in all_rows.values())

    # --- Fetch OHLCV for simulation ---
    print("\nFetching OHLCV data for portfolio simulation ...")
    ohlcv = get_ohlcv(args.ticker, args.start_date, args.end_date)

    # --- Simulate portfolio ---
    ta_records  = simulate_portfolio(decisions, ohlcv, args.initial_capital)
    bnh_records = simulate_buy_and_hold(ohlcv, args.initial_capital)
    sma_records = simulate_sma(ohlcv, args.initial_capital)

    # --- Metrics ---
    ta_metrics  = calc_metrics(ta_records,  args.initial_capital)
    bnh_metrics = calc_metrics(bnh_records, args.initial_capital)
    sma_metrics = calc_metrics(sma_records, args.initial_capital)

    # --- Save full results JSON ---
    json_path = csv_path.with_suffix(".json")
    result_payload = {
        "ticker":          args.ticker,
        "start_date":      args.start_date,
        "end_date":        args.end_date,
        "initial_capital": args.initial_capital,
        "provider":        args.provider,
        "quick_model":     quick_model,
        "deep_model":      deep_model,
        "metrics": {
            "TradingAgents": ta_metrics,
            "BuyAndHold":    bnh_metrics,
            "SMA_50_200":    sma_metrics,
        },
        "token_usage": {"tokens_in": total_in, "tokens_out": total_out},
        "estimated_cost_usd": (
            0.0 if args.provider == "ollama"
            else estimate_cost(total_in, total_out, quick_model, deep_model)
        ),
        "daily_portfolio": ta_records,
    }
    with open(json_path, "w") as f:
        json.dump(result_payload, f, indent=2, default=str)

    print_results(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        ta_metrics=ta_metrics,
        bnh_metrics=bnh_metrics,
        sma_metrics=sma_metrics,
        total_in=total_in,
        total_out=total_out,
        quick_model=quick_model,
        deep_model=deep_model,
        output_path=json_path,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
