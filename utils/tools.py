"""
AI Agent callable tools.
Each tool is a Python function + a JSON Schema description.
"""
import pandas as pd
from typing import Dict, List, Optional
from utils.data_fetcher import (
    fetch_company_info, fetch_financials, search_ticker_by_name, looks_like_ticker,
    get_peer_suggestions_by_size
)
from utils.ratios import compute_ratios_for_year, compute_multi_year_ratios, dupont_analysis
from utils.benchmark import compare_with_peers, benchmark_analysis, evaluate_against_benchmark
from utils.report import generate_report


# ============================================================
# Helper: resolve company name or ticker to a US ticker
# ============================================================
def resolve_ticker(company: str) -> Optional[str]:
    """Convert a company name or ticker into a standard US ticker."""
    if not company:
        return None
    company = company.strip()
    if looks_like_ticker(company):
        return company.upper()
    matches = search_ticker_by_name(company)
    return matches[0]["ticker"] if matches else None


def get_year_col(income_df: pd.DataFrame, target_year: int):
    """Find the most recent fiscal year column <= target_year."""
    if income_df.empty:
        return None, None
    cols = sorted(income_df.columns, reverse=True)
    cols_filtered = [c for c in cols if c.year <= target_year]
    if not cols_filtered:
        return None, None
    year_col = cols_filtered[0]
    prev_col = cols_filtered[1] if len(cols_filtered) > 1 else None
    return year_col, prev_col


# ============================================================
# Tool implementations
# ============================================================

def tool_fetch_company(company: str) -> Dict:
    """Fetch basic company info."""
    ticker = resolve_ticker(company)
    if not ticker:
        return {"error": f"Could not find a US ticker for '{company}'"}
    info = fetch_company_info(ticker)
    if "error" in info:
        return {"error": f"Failed to fetch data: {info.get('error')}"}
    return {
        "ticker": info["ticker"],
        "name": info.get("name"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "market_cap": info.get("market_cap"),
        "summary": (info.get("summary") or "")[:500],
    }


def tool_compute_ratios(company: str, year: int = 2024) -> Dict:
    """Compute all financial ratios for a given company and fiscal year."""
    ticker = resolve_ticker(company)
    if not ticker:
        return {"error": f"Could not find a US ticker for '{company}'"}

    fin = fetch_financials(ticker)
    income = fin.get("income", pd.DataFrame())
    balance = fin.get("balance", pd.DataFrame())
    cashflow = fin.get("cashflow", pd.DataFrame())

    if income.empty:
        return {"error": f"Financial data not available for {ticker}"}

    year_col, prev_col = get_year_col(income, year)
    if year_col is None:
        return {"error": f"No fiscal data found for {year} or earlier"}

    ratios = compute_ratios_for_year(income, balance, cashflow, year_col, prev_col)
    result = {"ticker": ticker, "actual_year": year_col.year, "ratios": {}}
    for k, v in ratios.items():
        if k.startswith("_"):
            continue
        rating = evaluate_against_benchmark(k, v) if v is not None else "-"
        result["ratios"][k] = {
            "value": round(v, 4) if v is not None else None,
            "rating": rating,
        }
    return result


def tool_dupont_analysis(company: str, year: int = 2024) -> Dict:
    """DuPont decomposition: ROE = Net Margin x Asset Turnover x Equity Multiplier."""
    ticker = resolve_ticker(company)
    if not ticker:
        return {"error": f"Could not find a US ticker for '{company}'"}

    fin = fetch_financials(ticker)
    income = fin.get("income", pd.DataFrame())
    balance = fin.get("balance", pd.DataFrame())
    cashflow = fin.get("cashflow", pd.DataFrame())
    if income.empty:
        return {"error": f"Financial data not available for {ticker}"}

    year_col, prev_col = get_year_col(income, year)
    if year_col is None:
        return {"error": f"No fiscal data found for {year}"}

    ratios = compute_ratios_for_year(income, balance, cashflow, year_col, prev_col)
    dp = dupont_analysis(ratios)

    calc_method = ratios.get("_calc_method", "ending")
    method_note = (
        "Computed using 'average equity' ((beginning + ending) / 2), the standard "
        "financial analysis convention."
        if calc_method == "average"
        else "Computed using 'ending equity' (because previous-year data was missing; "
              "yfinance typically only has 4 comparable years). This deviates from the "
              "standard 'average equity' convention - particularly impactful for "
              "buyback-heavy companies like Apple, where ROE will appear inflated."
    )

    return {
        "ticker": ticker,
        "year": year_col.year,
        "Net Margin": round(dp["Net Margin"], 4) if dp["Net Margin"] else None,
        "Asset Turnover": round(dp["Asset Turnover"], 4) if dp["Asset Turnover"] else None,
        "Equity Multiplier": round(dp["Equity Multiplier"], 4) if dp["Equity Multiplier"] else None,
        "ROE (DuPont)": round(dp["ROE (DuPont)"], 4) if dp["ROE (DuPont)"] else None,
        "ROE (Direct)": round(dp["ROE (Direct)"], 4) if dp["ROE (Direct)"] else None,
        "calc_method": calc_method,
        "calc_method_note": method_note,
        "interpretation": (
            "ROE = Net Margin x Asset Turnover x Equity Multiplier. "
            "The three factors reflect sales profitability, asset efficiency, "
            "and financial leverage respectively."
        ),
    }


def tool_trend_analysis(company: str, num_years: int = 5,
                          target_year: int = 2024) -> Dict:
    """Multi-year trend analysis for core metrics."""
    ticker = resolve_ticker(company)
    if not ticker:
        return {"error": f"Could not find a US ticker for '{company}'"}

    fin = fetch_financials(ticker)
    if fin.get("income", pd.DataFrame()).empty:
        return {"error": f"Financial data not available for {ticker}"}

    trend_df = compute_multi_year_ratios(fin, target_year, num_years=num_years)
    if trend_df.empty:
        return {"error": "Insufficient trend data"}

    key_metrics = ["Net Margin", "ROE (Return on Equity)", "ROA (Return on Assets)",
                   "Gross Margin", "Debt to Assets", "Asset Turnover"]
    result = {"ticker": ticker, "years": [int(y) for y in trend_df.columns],
              "actual_years_count": len(trend_df.columns), "trends": {}}

    for m in key_metrics:
        if m not in trend_df.index:
            continue
        series = trend_df.loc[m]
        values = [round(v, 4) if pd.notna(v) else None for v in series.values]
        result["trends"][m] = values

    direction = {}
    for m, values in result["trends"].items():
        clean = [v for v in values if v is not None]
        if len(clean) >= 2:
            change = (clean[-1] - clean[0]) / abs(clean[0]) if clean[0] != 0 else 0
            if change > 0.1:
                direction[m] = "Clear upward trend"
            elif change > 0.02:
                direction[m] = "Slight upward trend"
            elif change < -0.1:
                direction[m] = "Clear downward trend"
            elif change < -0.02:
                direction[m] = "Slight downward trend"
            else:
                direction[m] = "Roughly flat"
    result["direction"] = direction
    return result


def tool_peer_comparison(company: str, peers: Optional[List[str]] = None,
                          year: int = 2024, auto_match: bool = False) -> Dict:
    """
    Peer comparison.
    - peers: user-specified peer companies/tickers
    - auto_match: if True or peers is empty, auto-match by industry + size
    """
    ticker = resolve_ticker(company)
    if not ticker:
        return {"error": f"Could not find a US ticker for '{company}'"}

    info = fetch_company_info(ticker)

    peer_tickers = []
    if peers:
        for p in peers:
            t = resolve_ticker(p)
            if t and t != ticker:
                peer_tickers.append(t)

    if not peer_tickers and (auto_match or not peers):
        suggestions = get_peer_suggestions_by_size(
            sector=info.get("sector", ""),
            target_market_cap=info.get("market_cap"),
            exclude=ticker, n=5,
            company_name=info.get("name", ""),
            industry=info.get("industry", ""),
            country=info.get("country", ""),
        )
        peer_tickers = [s["ticker"] for s in suggestions]

    if not peer_tickers:
        return {"error": "Could not find suitable peer companies"}

    compare_df = compare_with_peers(ticker, peer_tickers, year)
    if compare_df.empty:
        return {"error": "Failed to fetch peer data"}

    key_metrics = ["Net Margin", "ROE (Return on Equity)", "ROA (Return on Assets)",
                   "Gross Margin", "Debt to Assets"]
    result = {
        "ticker": ticker,
        "year": year,
        "peers_used": peer_tickers,
        "comparison": {},
    }
    for m in key_metrics:
        if m not in compare_df.index:
            continue
        row = compare_df.loc[m]
        result["comparison"][m] = {
            col: round(v, 4) if pd.notna(v) else None
            for col, v in row.items()
        }
    return result


def tool_generate_full_report(company: str, year: int = 2024,
                                include_peers: bool = True) -> Dict:
    """Generate a full Markdown financial analysis report."""
    ticker = resolve_ticker(company)
    if not ticker:
        return {"error": f"Could not find a US ticker for '{company}'"}

    info = fetch_company_info(ticker)
    fin = fetch_financials(ticker)
    income = fin.get("income", pd.DataFrame())
    balance = fin.get("balance", pd.DataFrame())
    cashflow = fin.get("cashflow", pd.DataFrame())

    if income.empty:
        return {"error": "Financial data not available"}

    year_col, prev_col = get_year_col(income, year)
    if year_col is None:
        return {"error": f"No fiscal data found for {year}"}

    ratios = compute_ratios_for_year(income, balance, cashflow, year_col, prev_col)
    dp = dupont_analysis(ratios)
    trend_df = compute_multi_year_ratios(fin, year, num_years=5)

    # Peers
    compare_df = pd.DataFrame()
    if include_peers:
        suggestions = get_peer_suggestions_by_size(
            sector=info.get("sector", ""),
            target_market_cap=info.get("market_cap"),
            exclude=ticker, n=5,
            company_name=info.get("name", ""),
            industry=info.get("industry", ""),
            country=info.get("country", ""),
        )
        peer_tickers = [s["ticker"] for s in suggestions]
        if peer_tickers:
            compare_df = compare_with_peers(ticker, peer_tickers, year)

    actual_year = year_col.year

    # Quick fallback Markdown (used if AI section generation fails)
    md = generate_report(info, ratios, dp, trend_df, compare_df, actual_year)

    return {
        "ticker": ticker,
        "year": actual_year,
        "report_markdown": md,
        "filename": f"{ticker}_{actual_year}_FinancialReport.md",
        # Full data pipeline (for UI; LLM does not see this)
        "_full_data": {
            "info": info,
            "ratios": ratios,
            "dupont": dp,
            "trend_df": trend_df,
            "compare_df": compare_df,
            "actual_year": actual_year,
            "ticker": ticker,
        },
    }


# ============================================================
# Tool JSON Schemas (for OpenAI function calling)
# ============================================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_company",
            "description": ("Fetch basic company info (sector, market cap, summary). "
                            "Always call this first to confirm a company exists when "
                            "the user mentions one."),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": ("Company name or ticker, e.g. 'Apple', 'AAPL', "
                                        "'Tesla', 'TSLA'. Must be US-listed.")
                    }
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_ratios",
            "description": ("Compute all financial ratios (profitability, efficiency, "
                            "solvency, cash flow) for a given company and year. "
                            "Returns value + rating for each ratio."),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name or ticker"},
                    "year": {"type": "integer", "description": "Target fiscal year, default 2024"},
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dupont_analysis",
            "description": ("DuPont analysis: decomposes ROE into Net Margin x Asset "
                            "Turnover x Equity Multiplier to identify ROE drivers."),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name or ticker"},
                    "year": {"type": "integer", "description": "Target fiscal year, default 2024"},
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trend_analysis",
            "description": ("Multi-year trend analysis showing how core metrics changed "
                            "(up/down/flat) over the past several years."),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name or ticker"},
                    "num_years": {"type": "integer", "description": "Years to look back, default 5 (actual may be fewer)"},
                    "target_year": {"type": "integer", "description": "Ending year, default 2024"},
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peer_comparison",
            "description": ("Peer comparison. When user doesn't specify peers, set "
                            "auto_match=true to auto-match by industry + size."),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Target company"},
                    "peers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User-specified peer companies/tickers, can be empty"
                    },
                    "year": {"type": "integer", "description": "Fiscal year, default 2024"},
                    "auto_match": {
                        "type": "boolean",
                        "description": "When no peers given, auto-match by industry + size. Default true.",
                        "default": True,
                    },
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_full_report",
            "description": ("Generate a complete Markdown financial analysis report "
                            "covering ratios, DuPont, trends, peers, benchmarks, and "
                            "diagnosis. Use when user asks for 'full report', "
                            "'comprehensive analysis', 'download report', etc."),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "year": {"type": "integer", "description": "Default 2024"},
                    "include_peers": {"type": "boolean", "default": True},
                },
                "required": ["company"],
            },
        },
    },
]


# ============================================================
# Tool dispatcher
# ============================================================
TOOL_FUNCTIONS = {
    "fetch_company": tool_fetch_company,
    "compute_ratios": tool_compute_ratios,
    "dupont_analysis": tool_dupont_analysis,
    "trend_analysis": tool_trend_analysis,
    "peer_comparison": tool_peer_comparison,
    "generate_full_report": tool_generate_full_report,
}


def execute_tool(name: str, args: Dict) -> Dict:
    """Execute a tool by name with given args."""
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**args)
    except Exception as e:
        return {"error": f"Tool execution failed: {type(e).__name__}: {str(e)}"}
