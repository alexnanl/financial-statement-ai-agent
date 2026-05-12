"""
Peer comparison + benchmark analysis
"""
import pandas as pd
from typing import List, Dict
from utils.ratios import compute_ratios_for_year


# Generic industry benchmark thresholds (rough rules of thumb)
GENERIC_BENCHMARKS = {
    "Gross Margin": {"Excellent": 0.40, "Good": 0.25, "Fair": 0.15},
    "Operating Margin": {"Excellent": 0.20, "Good": 0.10, "Fair": 0.05},
    "Net Margin": {"Excellent": 0.15, "Good": 0.08, "Fair": 0.03},
    "ROA (Return on Assets)": {"Excellent": 0.10, "Good": 0.05, "Fair": 0.02},
    "ROE (Return on Equity)": {"Excellent": 0.20, "Good": 0.12, "Fair": 0.06},
    "Current Ratio": {"Excellent": 2.0, "Good": 1.5, "Fair": 1.0},
    "Quick Ratio": {"Excellent": 1.5, "Good": 1.0, "Fair": 0.7},
    "Debt to Assets": {"Excellent": 0.40, "Good": 0.60, "Fair": 0.75},  # Lower is better
    "Asset Turnover": {"Excellent": 1.0, "Good": 0.6, "Fair": 0.3},
    "Interest Coverage": {"Excellent": 8.0, "Good": 4.0, "Fair": 2.0},
}

# Metrics where lower is better
LOWER_IS_BETTER = {"Debt to Assets"}


def evaluate_against_benchmark(ratio_name: str, value: float) -> str:
    """Rate a single ratio against generic benchmarks."""
    if value is None or ratio_name not in GENERIC_BENCHMARKS:
        return "-"
    bm = GENERIC_BENCHMARKS[ratio_name]
    lower_better = ratio_name in LOWER_IS_BETTER

    if lower_better:
        if value <= bm["Excellent"]:
            return "Excellent"
        elif value <= bm["Good"]:
            return "Good"
        elif value <= bm["Fair"]:
            return "Fair"
        else:
            return "Weak"
    else:
        if value >= bm["Excellent"]:
            return "Excellent"
        elif value >= bm["Good"]:
            return "Good"
        elif value >= bm["Fair"]:
            return "Fair"
        else:
            return "Weak"


def compare_with_peers(target_ticker: str, peer_tickers: List[str],
                        target_year: int) -> pd.DataFrame:
    """
    Compare a target company against peers.
    Returns DataFrame: rows = metrics, columns = companies.
    """
    # Lazy import to avoid circular dependencies
    from utils.data_fetcher import fetch_company_info, fetch_financials

    all_tickers = [target_ticker] + peer_tickers
    result = {}

    for tk in all_tickers:
        try:
            financials = fetch_financials(tk)
            income_df = financials.get("income", pd.DataFrame())
            balance_df = financials.get("balance", pd.DataFrame())
            cashflow_df = financials.get("cashflow", pd.DataFrame())

            if income_df.empty:
                continue

            cols = sorted(income_df.columns, reverse=True)
            cols = [c for c in cols if c.year <= target_year]
            if not cols:
                continue
            year_col = cols[0]
            prev_col = cols[1] if len(cols) > 1 else None

            ratios = compute_ratios_for_year(income_df, balance_df, cashflow_df,
                                              year_col, prev_col)
            info = fetch_company_info(tk)
            display_name = f"{tk} ({info.get('name', tk)[:20]})"
            result[display_name] = ratios
        except Exception:
            continue

    df = pd.DataFrame(result)
    if df.empty:
        return df
    df = df[~df.index.str.startswith("_")]
    return df


def benchmark_analysis(ratios_dict: Dict) -> pd.DataFrame:
    """Build a benchmark comparison table for a single company's ratios."""
    rows = []
    for ratio_name, value in ratios_dict.items():
        if ratio_name.startswith("_"):
            continue
        if ratio_name not in GENERIC_BENCHMARKS:
            continue
        bm = GENERIC_BENCHMARKS[ratio_name]
        rows.append({
            "Metric": ratio_name,
            "Company": value,
            "Excellent": bm["Excellent"],
            "Good": bm["Good"],
            "Fair": bm["Fair"],
            "Rating": evaluate_against_benchmark(ratio_name, value),
        })
    return pd.DataFrame(rows)
