"""
Financial ratio computation module
Covers profitability, efficiency, solvency, growth + DuPont analysis.

All ratio names use English keys (e.g. "ROE", "Net Margin", etc.).
"""
from typing import Dict, Optional, List, Any
import pandas as pd


def _to_float(value: Any) -> Optional[float]:
    """Safely convert a single value to float. Returns None on failure."""
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_value(df: pd.DataFrame, candidates: List[str], col) -> Optional[float]:
    """
    Look up a metric value by trying multiple candidate field names.
    Handles yfinance (English fields) and FMP (camelCase fields).
    """
    if df is None or df.empty:
        return None
    if col not in df.columns:
        return None

    # Case-insensitive index lookup
    index_lookup = {str(idx).strip().lower(): idx for idx in df.index}

    for name in candidates:
        real_name = None
        if name in df.index:
            real_name = name
        else:
            real_name = index_lookup.get(str(name).strip().lower())

        if real_name is None:
            continue

        try:
            val = df.loc[real_name, col]

            if isinstance(val, pd.Series):
                vals = pd.to_numeric(val, errors="coerce").dropna()
                if vals.empty:
                    continue
                return float(vals.iloc[0])

            if isinstance(val, pd.DataFrame):
                vals = pd.to_numeric(val.stack(), errors="coerce").dropna()
                if vals.empty:
                    continue
                return float(vals.iloc[0])

            parsed = _to_float(val)
            if parsed is not None:
                return parsed
        except (KeyError, ValueError, TypeError):
            continue

    return None


# Field candidate names: yfinance + FMP (US data sources only)
FIELDS = {
    # ===== Income Statement =====
    "revenue": [
        "Total Revenue", "Operating Revenue", "Revenue",
        "revenue", "totalRevenue", "operatingRevenue",
    ],
    "cogs": [
        "Cost Of Revenue", "Cost of Revenue", "Reconciled Cost Of Revenue",
        "costOfRevenue", "costAndExpenses",
    ],
    "gross_profit": [
        "Gross Profit", "grossProfit",
    ],
    "operating_income": [
        "Operating Income", "Total Operating Income As Reported",
        "operatingIncome",
    ],
    "ebit": [
        "EBIT", "Operating Income", "ebit", "operatingIncome",
    ],
    "net_income": [
        "Net Income", "Net Income Common Stockholders",
        "Net Income Continuous Operations",
        "netIncome", "netIncomeCommonStockholders",
    ],
    "interest_expense": [
        "Interest Expense", "Interest Expense Non Operating",
        "interestExpense",
    ],
    "tax_expense": [
        "Tax Provision", "Income Tax Expense", "incomeTaxExpense",
    ],
    "pretax_income": [
        "Pretax Income", "Income Before Tax", "incomeBeforeTax",
    ],

    # ===== Balance Sheet =====
    "total_assets": [
        "Total Assets", "totalAssets",
    ],
    "current_assets": [
        "Current Assets", "Total Current Assets", "totalCurrentAssets",
    ],
    "current_liab": [
        "Current Liabilities", "Total Current Liabilities", "totalCurrentLiabilities",
    ],
    "total_liab": [
        "Total Liabilities Net Minority Interest", "Total Liab", "totalLiabilities",
        "totalLiabilitiesNetMinorityInterest",
    ],
    "total_equity": [
        "Total Equity Gross Minority Interest", "Stockholders Equity",
        "Common Stock Equity", "Total Stockholder Equity",
        "totalStockholdersEquity", "totalEquity", "totalEquityGrossMinorityInterest",
    ],
    "cash": [
        "Cash And Cash Equivalents", "Cash", "cashAndCashEquivalents",
        "cashAndShortTermInvestments",
    ],
    "inventory": ["Inventory", "inventory"],
    "receivables": [
        "Accounts Receivable", "Receivables", "netReceivables",
        "accountReceivables",
    ],
    "payables": [
        "Accounts Payable", "Payables", "accountPayables",
    ],
    "long_term_debt": [
        "Long Term Debt", "longTermDebt",
    ],
    "short_term_debt": [
        "Short Term Debt", "Current Debt", "shortTermDebt", "currentDebt",
    ],

    # ===== Cash Flow Statement =====
    "operating_cf": [
        "Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
        "operatingCashFlow", "netCashProvidedByOperatingActivities",
    ],
    "capex": [
        "Capital Expenditure", "capitalExpenditure", "capitalExpenditures",
        "Purchase Of PPE", "Purchase of Property Plant and Equipment",
    ],
    "free_cf": ["Free Cash Flow", "freeCashFlow"],
}


def safe_div(a, b) -> Optional[float]:
    """Safe division - avoids None denominator and division by zero."""
    if a is None or b is None:
        return None
    try:
        a = float(a)
        b = float(b)
        if b == 0:
            return None
        return a / b
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def avg(a, b) -> Optional[float]:
    """Two-period average. Returns the available one when only one is given."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


# All ratio keys used by the system (English)
RATIO_KEYS = {
    "GROSS_MARGIN": "Gross Margin",
    "OPERATING_MARGIN": "Operating Margin",
    "NET_MARGIN": "Net Margin",
    "ROA": "ROA (Return on Assets)",
    "ROE": "ROE (Return on Equity)",
    "ROIC": "ROIC (Return on Invested Capital)",
    "ASSET_TURNOVER": "Asset Turnover",
    "INVENTORY_TURNOVER": "Inventory Turnover",
    "RECEIVABLES_TURNOVER": "Receivables Turnover",
    "CURRENT_RATIO": "Current Ratio",
    "QUICK_RATIO": "Quick Ratio",
    "DEBT_TO_ASSETS": "Debt to Assets",
    "EQUITY_MULTIPLIER": "Equity Multiplier",
    "INTEREST_COVERAGE": "Interest Coverage",
    "OCF_TO_REVENUE": "Operating Cash Flow / Revenue",
    "FCF": "Free Cash Flow",
}


def compute_ratios_for_year(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    year_col,
    prev_year_col=None,
) -> Dict:
    """
    Compute all ratios for a single fiscal year.
    year_col: current period column (Timestamp)
    prev_year_col: previous period column, used for averaged metrics
    """
    g = lambda df, key, col: get_value(df, FIELDS[key], col)

    # Income statement items
    revenue = g(income_df, "revenue", year_col)
    cogs = g(income_df, "cogs", year_col)
    gross_profit = g(income_df, "gross_profit", year_col)
    if gross_profit is None and revenue is not None and cogs is not None:
        gross_profit = revenue - cogs

    op_income = g(income_df, "operating_income", year_col)
    net_income = g(income_df, "net_income", year_col)
    interest = g(income_df, "interest_expense", year_col)
    tax = g(income_df, "tax_expense", year_col)
    pretax = g(income_df, "pretax_income", year_col)

    # Balance sheet - current period
    assets = g(balance_df, "total_assets", year_col)
    cur_assets = g(balance_df, "current_assets", year_col)
    cur_liab = g(balance_df, "current_liab", year_col)
    total_liab = g(balance_df, "total_liab", year_col)
    equity = g(balance_df, "total_equity", year_col)
    inventory = g(balance_df, "inventory", year_col)
    receivables = g(balance_df, "receivables", year_col)
    cash = g(balance_df, "cash", year_col)
    lt_debt = g(balance_df, "long_term_debt", year_col)
    st_debt = g(balance_df, "short_term_debt", year_col)

    # Previous period for averages
    if prev_year_col is not None:
        prev_assets = g(balance_df, "total_assets", prev_year_col)
        prev_equity = g(balance_df, "total_equity", prev_year_col)
        prev_inventory = g(balance_df, "inventory", prev_year_col)
        prev_receivables = g(balance_df, "receivables", prev_year_col)
    else:
        prev_assets = prev_equity = prev_inventory = prev_receivables = None

    avg_assets = avg(assets, prev_assets)
    avg_equity = avg(equity, prev_equity)
    avg_inventory = avg(inventory, prev_inventory)
    avg_receivables = avg(receivables, prev_receivables)

    # Cash flow
    op_cf = get_value(cashflow_df, FIELDS["operating_cf"], year_col)
    capex = get_value(cashflow_df, FIELDS["capex"], year_col)
    fcf = get_value(cashflow_df, FIELDS["free_cf"], year_col)
    if fcf is None and op_cf is not None and capex is not None:
        # yfinance / FMP record capex as negative, so FCF = OCF + CapEx
        # If capex shows positive (some sources), then FCF = OCF - CapEx
        fcf = op_cf - capex if capex > 0 else op_cf + capex

    # Total debt
    total_debt = None
    if lt_debt is not None or st_debt is not None:
        total_debt = (lt_debt or 0) + (st_debt or 0)

    tax_rate = safe_div(tax, pretax)
    if tax_rate is None:
        tax_rate = 0.21
    nopat = op_income * (1 - tax_rate) if op_income is not None else None
    invested_capital = None
    if equity is not None or total_debt is not None:
        invested_capital = (equity or 0) + (total_debt or 0)

    ratios = {
        # ===== Profitability =====
        "Gross Margin": safe_div(gross_profit, revenue),
        "Operating Margin": safe_div(op_income, revenue),
        "Net Margin": safe_div(net_income, revenue),
        "ROA (Return on Assets)": safe_div(net_income, avg_assets),
        "ROE (Return on Equity)": safe_div(net_income, avg_equity),
        "ROIC (Return on Invested Capital)": safe_div(nopat, invested_capital),

        # ===== Efficiency =====
        "Asset Turnover": safe_div(revenue, avg_assets),
        "Inventory Turnover": safe_div(cogs, avg_inventory),
        "Receivables Turnover": safe_div(revenue, avg_receivables),

        # ===== Solvency =====
        "Current Ratio": safe_div(cur_assets, cur_liab),
        "Quick Ratio": safe_div(
            (cur_assets - inventory) if (cur_assets is not None and inventory is not None) else cur_assets,
            cur_liab,
        ),
        "Debt to Assets": safe_div(total_liab, assets),
        "Equity Multiplier": safe_div(avg_assets, avg_equity),
        "Interest Coverage": safe_div(op_income, abs(interest)) if interest else None,

        # ===== Cash Flow =====
        "Operating Cash Flow / Revenue": safe_div(op_cf, revenue),
        "Free Cash Flow": fcf,

        # ===== Raw values (for reports/debug) =====
        "_revenue": revenue,
        "_net_income": net_income,
        "_total_assets": assets,
        "_equity": equity,
        "_calc_method": "average" if prev_year_col is not None else "ending",
    }
    return ratios


def dupont_analysis(ratios: Dict) -> Dict:
    """
    DuPont decomposition:
    ROE = Net Margin x Asset Turnover x Equity Multiplier
    """
    nm = ratios.get("Net Margin")
    ato = ratios.get("Asset Turnover")
    em = ratios.get("Equity Multiplier")
    roe_calc = nm * ato * em if (nm is not None and ato is not None and em is not None) else None

    return {
        "Net Margin": nm,
        "Asset Turnover": ato,
        "Equity Multiplier": em,
        "ROE (DuPont)": roe_calc,
        "ROE (Direct)": ratios.get("ROE (Return on Equity)"),
    }


def compute_multi_year_ratios(
    financials: Dict[str, pd.DataFrame],
    target_year: int,
    num_years: int = 5,
) -> pd.DataFrame:
    """
    Compute ratios across multiple years for trend analysis.
    Returns DataFrame with metrics as rows, years as columns.
    """
    income_df = financials.get("income", pd.DataFrame())
    balance_df = financials.get("balance", pd.DataFrame())
    cashflow_df = financials.get("cashflow", pd.DataFrame())

    if income_df is None or income_df.empty:
        return pd.DataFrame()

    cols = [c for c in income_df.columns if hasattr(c, "year")]
    cols = sorted(cols, reverse=True)
    cols = [c for c in cols if c.year <= target_year][:num_years]
    cols = sorted(cols)  # Ascending for trend chart

    result = {}
    for i, col in enumerate(cols):
        prev_col = cols[i - 1] if i > 0 else None
        ratios = compute_ratios_for_year(income_df, balance_df, cashflow_df, col, prev_col)
        result[col.year] = ratios

    df = pd.DataFrame(result)
    if df.empty:
        return df

    # Filter out internal underscore-prefixed fields
    df = df[~df.index.astype(str).str.startswith("_")]
    return df


# Helper: identify if a ratio name should be displayed as a percentage
def is_percentage_metric(name: str) -> bool:
    """Check if a ratio name represents a percentage value (e.g. margins, returns)."""
    pct_keywords = ["Margin", "ROE", "ROA", "ROIC", "Debt to Assets",
                    "/ Revenue", "Tax Rate"]
    return any(kw in name for kw in pct_keywords)
