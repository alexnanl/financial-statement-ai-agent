"""
data_provider.py - Unified data adapter for US stocks

Design:
    - PRIMARY source:   SEC EDGAR (data.sec.gov) - free, no API key, no daily limit
                        Returns full financial history (10+ years) per company.
    - SECONDARY source: yfinance (Yahoo Finance) - free, no API key
                        Used to fill in market cap (SEC doesn't provide stock prices)
                        and as a safety net when SEC data is incomplete.
    - DORMANT source:   FMP (Financial Modeling Prep) - kept in code but disabled
                        Set USE_FMP=True (and provide FMP_API_KEY) to re-enable.
                        Disabled because FMP's "Legacy free tier" was discontinued;
                        new free accounts cannot access financial statements.

Routing:
    US stocks  -> SEC EDGAR (primary) -> yfinance (fill gaps)
    Non-US     -> rejected with a clear error (this build is US-only)

Diagnostics:
    Every call to SEC/FMP logs its outcome into _DIAG (last 20 calls).
    Call diagnose() to see what happened on the most recent fetches.

Dependencies:
    pip install yfinance      # Yahoo Finance (for market cap)
    requests                  # SEC EDGAR HTTP calls

Environment / secrets:
    SEC_USER_AGENT            # Optional override for SEC User-Agent header.
                              # Defaults to "alexnanl (alexnanl@github.com)".
                              # SEC requires a real contact in case of abuse.

    FMP_API_KEY               # OPTIONAL - only used if you flip USE_FMP=True
                              # below. Kept here in case you upgrade to a paid
                              # FMP plan in the future.
"""
import os
import time
import random
from collections import deque
from typing import Dict, Optional, List, Any
import pandas as pd
import streamlit as st


# ===========================================
# Master switches
# ===========================================
USE_FMP = False   # Set to True to re-enable FMP as primary (requires paid plan)
USE_SEC = True    # SEC EDGAR is the primary source in this build


# ===========================================
# SEC user-agent (required by SEC fair-use policy)
# ===========================================
DEFAULT_SEC_USER_AGENT = "alexnanl (alexnanl@github.com)"


def _get_sec_user_agent() -> str:
    """Get the User-Agent string sent to SEC (overridable via secrets/env)."""
    try:
        ua = st.secrets.get("SEC_USER_AGENT", None)
        if ua and isinstance(ua, str) and ua.strip():
            return ua.strip()
    except Exception:
        pass
    env_ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if env_ua:
        return env_ua
    return DEFAULT_SEC_USER_AGENT


# ===========================================
# Diagnostic log (ring buffer of last 20 API calls)
# ===========================================
_DIAG = deque(maxlen=20)


def _log(endpoint: str, status: str, detail: str = "") -> None:
    """Record an API call outcome for diagnostics."""
    _DIAG.append({
        "endpoint": endpoint,
        "status": status,
        "detail": detail,
        "ts": time.strftime("%H:%M:%S"),
    })


def diagnose() -> Dict:
    """Return current data-source diagnostics."""
    return {
        "primary_source": "SEC EDGAR" if USE_SEC else ("FMP" if USE_FMP else "yfinance"),
        "sec_user_agent": _get_sec_user_agent(),
        "fmp_enabled": USE_FMP,
        "fmp_key_present": bool(_get_fmp_key()) if USE_FMP else False,
        "fmp_key_source": _get_fmp_key_source() if USE_FMP else "disabled",
        "recent_calls": list(_DIAG),
    }


def clear_diagnostics() -> None:
    """Reset the diagnostic log."""
    _DIAG.clear()


def setup_status() -> Dict[str, Any]:
    """Tell the UI which data sources are active. Backward-compatible shape."""
    return {
        "sec_enabled": USE_SEC,
        "fmp_enabled": USE_FMP,
        "fmp_configured": bool(_get_fmp_key()) if USE_FMP else False,
        "yfinance_available": True,
        "primary_source_label": (
            "SEC EDGAR" if USE_SEC
            else ("FMP" if USE_FMP and _get_fmp_key() else "yfinance")
        ),
    }


# ===========================================
# Market detection (US-only build)
# ===========================================

def detect_market(ticker: str) -> str:
    """Detect market from ticker. US-only build."""
    if not ticker:
        return "unsupported"
    t = ticker.upper().strip()
    if any(t.endswith(suf) for suf in [".SS", ".SZ", ".BJ", ".HK", ".T", ".L",
                                          ".DE", ".PA", ".KS", ".SW", ".MI", ".AS"]):
        return "unsupported"
    base = t.split(".")[0].replace("-", "")
    if base.isalnum() and len(base) <= 6 and any(c.isalpha() for c in base):
        return "us"
    return "unsupported"


# ===========================================
# Retry wrapper
# ===========================================

def _call_with_retry(func, *args, max_retries=3, **kwargs):
    """Generic retry wrapper for rate-limited APIs."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(x in err_str for x in ["rate", "429", "too many", "timeout"]):
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue
            raise
    raise last_err


# ===========================================
# SEC EDGAR
# ===========================================

SEC_BASE = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"

# Ticker -> CIK mapping (cached 24h)
@st.cache_data(ttl=86400, show_spinner=False)
def _load_ticker_cik_map() -> Dict[str, str]:
    """
    Fetch and cache the SEC's official ticker -> CIK mapping.
    Returns dict {TICKER: zero-padded-10-digit-CIK}.
    """
    import requests
    headers = {"User-Agent": _get_sec_user_agent(), "Accept": "application/json"}
    url = f"{SEC_WWW}/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if not resp.ok:
            _log("company_tickers.json", "http_error",
                 f"HTTP {resp.status_code}")
            return {}
        data = resp.json()
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        mapping = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                mapping[ticker] = str(cik).zfill(10)
        _log("company_tickers.json", "ok", f"loaded {len(mapping)} tickers")
        return mapping
    except Exception as e:
        _log("company_tickers.json", "error",
             f"{type(e).__name__}: {str(e)[:80]}")
        return {}


def _ticker_to_cik(ticker: str) -> Optional[str]:
    """Convert ticker (e.g. 'AAPL') to zero-padded CIK (e.g. '0000320193')."""
    if not ticker:
        return None
    # Handle dotted/dashed tickers (BRK.B, BRK-B). SEC uses no separator.
    candidates = [ticker.upper(), ticker.upper().replace(".", "-"),
                  ticker.upper().replace("-", "."), ticker.upper().replace("-", "")]
    mapping = _load_ticker_cik_map()
    for c in candidates:
        if c in mapping:
            return mapping[c]
    return None


def _sec_get(url: str, label: str) -> Optional[dict]:
    """Hit a SEC JSON endpoint. Returns parsed dict, or None on failure."""
    import requests
    headers = {"User-Agent": _get_sec_user_agent(), "Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.exceptions.Timeout:
        _log(label, "error", "request timed out (>15s)")
        return None
    except requests.exceptions.ConnectionError as e:
        _log(label, "error", f"connection error: {str(e)[:80]}")
        return None
    except Exception as e:
        _log(label, "error", f"{type(e).__name__}: {str(e)[:80]}")
        return None

    if resp.status_code == 403:
        _log(label, "blocked",
             "403 - check User-Agent header (must include name+email)")
        return None
    if resp.status_code == 404:
        _log(label, "not_found", "404 - CIK/ticker not in SEC database")
        return None
    if resp.status_code == 429:
        _log(label, "rate_limit",
             "429 - exceeded 10 req/sec; will retry after backoff")
        return None
    if not resp.ok:
        _log(label, "http_error", f"HTTP {resp.status_code}")
        return None

    try:
        data = resp.json()
        _log(label, "ok", f"{len(str(data))} bytes")
        return data
    except Exception as e:
        _log(label, "parse_error", f"invalid JSON: {str(e)[:80]}")
        return None


# SIC code -> Sector mapping (rough)
# SEC uses SIC codes, not GICS sectors. This is a sensible mapping.
# Narrower (more specific) ranges win over broader ones - see _sic_to_sector.
SIC_TO_SECTOR = {
    # ===== Broad ranges (catch-all) =====
    range(100, 1000):  "Agriculture",
    range(1000, 1500): "Basic Materials",      # Mining
    range(1500, 1800): "Industrials",          # Construction
    range(2000, 2400): "Consumer Defensive",   # Food, Tobacco
    range(2400, 2700): "Basic Materials",      # Lumber, Furniture, Paper
    range(2700, 2800): "Communication Services",  # Publishing
    range(2800, 3000): "Basic Materials",      # Chemicals
    range(3000, 3200): "Consumer Cyclical",    # Rubber, Plastics, Leather
    range(3200, 3500): "Basic Materials",      # Stone, Clay, Glass, Metals
    range(3500, 3700): "Industrials",          # Industrial Machinery
    range(3700, 3800): "Industrials",          # Transportation Equipment
    range(3800, 4000): "Industrials",          # Instruments, Misc Mfg
    range(4000, 4800): "Industrials",          # Transportation
    range(4800, 4900): "Communication Services",
    range(4900, 5000): "Utilities",
    range(5000, 6000): "Consumer Cyclical",    # Wholesale + Retail
    range(6000, 7000): "Financial Services",
    range(7000, 8000): "Industrials",          # Services (general)
    range(8000, 9000): "Healthcare",           # Health Services (specific narrower below)

    # ===== Narrower overrides (more specific) =====
    # Manufacturing - specifics
    range(2200, 2400): "Consumer Cyclical",       # Textiles, Apparel
    range(2830, 2840): "Healthcare",              # Pharmaceutical Preparations (2834=Pharma)
    range(2900, 3000): "Energy",                  # Petroleum Refining (291x)
    range(3570, 3580): "Technology",              # Computer & Office Equipment
    range(3600, 3700): "Technology",              # Electronic Components
    range(3670, 3680): "Technology",              # Semiconductors (3674)
    range(3710, 3720): "Consumer Cyclical",       # Motor Vehicles (3711=Auto)
    range(3720, 3730): "Industrials",             # Aircraft
    range(3825, 3845): "Healthcare",              # Medical Instruments

    # Retail specifics
    range(5400, 5500): "Consumer Defensive",      # Food retail (groceries)
    range(5910, 5920): "Consumer Defensive",      # Drug stores
    range(5812, 5814): "Consumer Cyclical",       # Restaurants

    # Real estate
    range(6500, 6600): "Real Estate",
    range(6770, 6800): "Real Estate",             # REITs

    # Services specifics
    range(7370, 7380): "Technology",              # Computer Services (7372=Software)
    range(7372, 7373): "Technology",              # Prepackaged Software
    range(7800, 7900): "Communication Services",  # Motion Pictures
    range(8060, 8070): "Healthcare",              # Hospitals
    range(8200, 8400): "Consumer Cyclical",       # Educational Services
    range(8700, 8800): "Industrials",             # Engineering, Accounting
}


def _sic_to_sector(sic_code: Any) -> str:
    """
    Map a SIC code to a sector name. Returns 'N/A' if unknown.
    Matches narrowest (most specific) range first, so e.g. SIC 2834
    (Pharma) gets 'Healthcare', not the broader 2800-2900 'Basic Materials'.
    """
    if sic_code is None:
        return "N/A"
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return "N/A"
    # Sort ranges by width (narrowest first) so specific overrides win
    matches = [(r, sector) for r, sector in SIC_TO_SECTOR.items() if sic in r]
    if not matches:
        return "N/A"
    # Pick the narrowest range
    matches.sort(key=lambda x: x[0].stop - x[0].start)
    return matches[0][1]


def sec_get_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company info from SEC EDGAR submissions endpoint."""
    cik = _ticker_to_cik(ticker)
    if not cik:
        _log(f"submissions/{ticker}", "not_found",
             f"ticker {ticker} not in SEC database")
        return None

    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    data = _sec_get(url, f"submissions/{ticker}")
    if not data:
        return None

    sic = data.get("sic")
    sector = _sic_to_sector(sic)
    industry = data.get("sicDescription", "N/A")
    name = data.get("name", ticker.upper())

    return {
        "ticker": ticker.upper(),
        "name": name,
        "sector": sector,
        "industry": industry,
        "country": "US",
        "currency": "USD",
        "market_cap": None,   # SEC doesn't provide this - filled in later from yfinance
        "summary": data.get("description", ""),
        "exchange": (data.get("exchanges") or ["N/A"])[0] if data.get("exchanges") else "N/A",
        "website": data.get("website", ""),
        "cik": cik,
        "sic": sic,
        "fiscal_year_end": data.get("fiscalYearEnd"),
        "_source": "sec",
    }


# ===========================================
# SEC XBRL -> standard field mapping
# ===========================================
#
# SEC uses XBRL tags (us-gaap:...). Different companies use slightly different
# tags for the same concept. We try each candidate in order; first one found wins.
#
SEC_XBRL_TAGS = {
    # Income statement
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostsAndExpenses",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
    ],
    "tax_expense": [
        "IncomeTaxExpenseBenefit",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],

    # Balance sheet
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liab": ["LiabilitiesCurrent"],
    "total_liab": ["Liabilities"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "inventory": ["InventoryNet"],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "payables": [
        "AccountsPayableCurrent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent",
    ],

    # Cash flow statement
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}

# What we expose to ratios.py. The ratios module looks these up by display name,
# so the column names in the resulting DataFrame match what ratios.py expects.
INTERNAL_TO_DISPLAY = {
    "revenue":          "Total Revenue",
    "cogs":             "Cost Of Revenue",
    "gross_profit":     "Gross Profit",
    "operating_income": "Operating Income",
    "net_income":       "Net Income",
    "interest_expense": "Interest Expense",
    "tax_expense":      "Tax Provision",
    "pretax_income":    "Pretax Income",
    "total_assets":     "Total Assets",
    "current_assets":   "Current Assets",
    "current_liab":     "Current Liabilities",
    "total_liab":       "Total Liab",
    "total_equity":     "Stockholders Equity",
    "cash":             "Cash And Cash Equivalents",
    "inventory":        "Inventory",
    "receivables":      "Accounts Receivable",
    "payables":         "Accounts Payable",
    "long_term_debt":   "Long Term Debt",
    "short_term_debt":  "Short Term Debt",
    "operating_cf":     "Operating Cash Flow",
    "capex":            "Capital Expenditure",
}

# Which internal keys belong in which statement
INCOME_KEYS = ["revenue", "cogs", "gross_profit", "operating_income", "net_income",
               "interest_expense", "tax_expense", "pretax_income"]
BALANCE_KEYS = ["total_assets", "current_assets", "current_liab", "total_liab",
                "total_equity", "cash", "inventory", "receivables", "payables",
                "long_term_debt", "short_term_debt"]
CASHFLOW_KEYS = ["operating_cf", "capex"]


def _pick_annual_facts(fact_data: dict, num_years: int = 10) -> Dict[pd.Timestamp, float]:
    """
    From a SEC XBRL "fact" dict, extract one annual value per fiscal year.
    Returns {fiscal_year_end_date: value}.

    SEC's XBRL facts contain many filings per year (10-Q, 10-K, amendments).
    We want one annual value per FY, preferring 10-K reports (full year).
    """
    if not fact_data or "units" not in fact_data:
        return {}

    # Find the right units container. Most amounts are in USD.
    units = fact_data["units"]
    chosen_unit_list = units.get("USD") or units.get("USD/shares") or \
                       next(iter(units.values()), [])

    # Group by fiscal year, prefer FY (annual) frames
    annual = {}
    for entry in chosen_unit_list:
        # We want annual numbers. SEC marks 'fp':'FY' for full-year filings.
        # Some balance-sheet items don't have fp, just an 'end' date.
        fp = entry.get("fp", "")
        form = entry.get("form", "")
        fy = entry.get("fy")
        end = entry.get("end")
        val = entry.get("val")

        if val is None or end is None or fy is None:
            continue

        # For income statement / cashflow: keep only FY (annual) values
        # For balance sheet: any filing's "end" balance is a point-in-time number
        # We detect statement type by whether 'start' is present (only flow items have start).
        is_flow_item = "start" in entry

        if is_flow_item:
            # Income/cashflow item - must be full year
            if fp != "FY":
                continue
            # Confirm it spans roughly a year
            try:
                start = pd.Timestamp(entry["start"])
                end_ts = pd.Timestamp(end)
                if (end_ts - start).days < 350:  # not a full year
                    continue
            except Exception:
                continue
        else:
            # Balance sheet item - just keep year-end (Q4) snapshots
            # SEC's 10-K filings have fp='FY'; 10-Q have fp='Q1/Q2/Q3'
            # For balance items in 10-K, fp is 'FY'
            if fp not in ("FY", ""):
                continue

        # Use fiscal year as key, prefer 10-K filings
        try:
            end_ts = pd.Timestamp(end)
        except Exception:
            continue

        # Keep the latest filing for this FY (10-K/A amendments overwrite 10-K)
        existing = annual.get(end_ts)
        if existing is None:
            annual[end_ts] = val
        # If we already have a value, prefer 10-K over 10-Q
        elif form.startswith("10-K"):
            annual[end_ts] = val

    # Keep only most recent N years
    sorted_dates = sorted(annual.keys(), reverse=True)[:num_years]
    return {d: annual[d] for d in sorted_dates}


def sec_get_financials(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch financial statements from SEC EDGAR's XBRL Company Facts API.
    Returns three DataFrames (income / balance / cashflow) with the same
    shape that downstream code expects.
    """
    cik = _ticker_to_cik(ticker)
    if not cik:
        _log(f"companyfacts/{ticker}", "not_found", f"no CIK for {ticker}")
        return None

    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    data = _sec_get(url, f"companyfacts/{ticker}")
    if not data:
        return None

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        _log(f"companyfacts/{ticker}", "empty", "no us-gaap facts in response")
        return None

    # For each internal key, find the first XBRL tag that has data
    # and extract its annual values.
    extracted = {}  # internal_key -> {date: value}
    for internal_key, candidate_tags in SEC_XBRL_TAGS.items():
        for tag in candidate_tags:
            if tag in us_gaap:
                annual = _pick_annual_facts(us_gaap[tag])
                if annual:
                    extracted[internal_key] = annual
                    break

    if not extracted:
        _log(f"companyfacts/{ticker}", "empty",
             "no recognized XBRL tags - unusual filing format")
        return None

    # Build the three statement DataFrames.
    # Columns = report dates (descending), Index = display field names.
    def build_statement_df(keys: List[str]) -> pd.DataFrame:
        # Collect all dates present in any key
        all_dates = set()
        for k in keys:
            if k in extracted:
                all_dates.update(extracted[k].keys())
        if not all_dates:
            return pd.DataFrame()
        sorted_dates = sorted(all_dates, reverse=True)

        rows = {}
        for k in keys:
            display_name = INTERNAL_TO_DISPLAY[k]
            if k not in extracted:
                rows[display_name] = [None] * len(sorted_dates)
            else:
                values = extracted[k]
                rows[display_name] = [values.get(d) for d in sorted_dates]

        df = pd.DataFrame(rows, index=sorted_dates).T  # transpose so dates = columns
        df.columns = sorted_dates
        return df

    income_df = build_statement_df(INCOME_KEYS)
    balance_df = build_statement_df(BALANCE_KEYS)
    cashflow_df = build_statement_df(CASHFLOW_KEYS)

    return {
        "income": income_df,
        "balance": balance_df,
        "cashflow": cashflow_df,
        "_source": "sec",
    }


# ===========================================
# yfinance (used for market cap, and as last-resort fallback)
# ===========================================

def yf_get_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company info via yfinance (used as last-resort fallback)."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            return tk.info or {}

        info = _call_with_retry(_do_fetch)
        if not info or len(info) < 3:
            return None
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "US"),
            "currency": info.get("financialCurrency", info.get("currency", "USD")),
            "market_cap": info.get("marketCap"),
            "summary": info.get("longBusinessSummary", ""),
            "exchange": info.get("exchange", "N/A"),
            "website": info.get("website", ""),
            "_source": "yfinance",
        }
    except Exception:
        return None


def yf_get_market_cap(ticker: str) -> Optional[float]:
    """Lightweight market cap fetch via yfinance (SEC doesn't provide this)."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            return info.get("marketCap")

        result = _call_with_retry(_do_fetch, max_retries=2)
        if result:
            _log(f"yf_market_cap/{ticker}", "ok", f"${result/1e9:.1f}B")
        return result
    except Exception as e:
        _log(f"yf_market_cap/{ticker}", "error", f"{type(e).__name__}")
        return None


def yf_get_summary(ticker: str) -> Optional[str]:
    """Fetch business summary from yfinance (SEC's is often empty)."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            return info.get("longBusinessSummary", "")

        return _call_with_retry(_do_fetch, max_retries=2)
    except Exception:
        return None


def yf_get_financials(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch three financial statements via yfinance (used only if SEC fails)."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            return {
                "income": tk.financials if tk.financials is not None else pd.DataFrame(),
                "balance": tk.balance_sheet if tk.balance_sheet is not None else pd.DataFrame(),
                "cashflow": tk.cashflow if tk.cashflow is not None else pd.DataFrame(),
                "_source": "yfinance",
            }

        return _call_with_retry(_do_fetch)
    except Exception:
        return None


# ===========================================
# FMP (DORMANT - kept for future re-enable)
# ===========================================

FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _get_fmp_key() -> Optional[str]:
    """Get FMP API key (only used if USE_FMP=True)."""
    try:
        key = st.secrets.get("FMP_API_KEY", None)
        if key and isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    env_key = os.environ.get("FMP_API_KEY", "").strip()
    if env_key:
        return env_key
    return None


def _get_fmp_key_source() -> str:
    try:
        key = st.secrets.get("FMP_API_KEY", None)
        if key and isinstance(key, str) and key.strip():
            return "streamlit_secrets"
    except Exception:
        pass
    if os.environ.get("FMP_API_KEY", "").strip():
        return "env_var"
    return "not_found"


def _fmp_get(endpoint: str, params: Optional[Dict] = None) -> Optional[list]:
    """Hit FMP REST endpoint (only used if USE_FMP=True)."""
    import requests
    key = _get_fmp_key()
    if not key:
        _log(endpoint, "skip", "FMP_API_KEY not configured")
        return None
    params = params or {}
    params["apikey"] = key
    try:
        resp = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=10)
    except Exception as e:
        _log(endpoint, "error", f"{type(e).__name__}: {str(e)[:80]}")
        return None

    if resp.status_code == 403:
        _log(endpoint, "paywall", "403 - endpoint requires paid plan")
        return None
    if resp.status_code == 429:
        _log(endpoint, "rate_limit", "429 - daily quota exceeded")
        return None
    if not resp.ok:
        _log(endpoint, "http_error", f"HTTP {resp.status_code}")
        return None

    try:
        data = resp.json()
    except Exception:
        _log(endpoint, "parse_error", "invalid JSON")
        return None

    if isinstance(data, dict):
        if "Error Message" in data:
            _log(endpoint, "fmp_error", str(data["Error Message"])[:120])
            return None

    if isinstance(data, list) and len(data) > 0:
        _log(endpoint, "ok", f"received {len(data)} record(s)")
        return data

    _log(endpoint, "empty", "no data")
    return None


def fmp_get_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company profile from FMP (DORMANT - only used if USE_FMP=True)."""
    data = _fmp_get(f"profile/{ticker.upper()}")
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    p = data[0]
    return {
        "ticker": ticker.upper(),
        "name": p.get("companyName") or ticker.upper(),
        "sector": p.get("sector", "N/A"),
        "industry": p.get("industry", "N/A"),
        "country": p.get("country", "US"),
        "currency": p.get("currency", "USD"),
        "market_cap": p.get("mktCap"),
        "summary": p.get("description", ""),
        "exchange": p.get("exchangeShortName", "N/A"),
        "website": p.get("website", ""),
        "_source": "fmp",
    }


def fmp_get_financials(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch financials from FMP (DORMANT - only used if USE_FMP=True)."""
    tk = ticker.upper()
    income_raw = _fmp_get(f"income-statement/{tk}", {"limit": 10})
    balance_raw = _fmp_get(f"balance-sheet-statement/{tk}", {"limit": 10})
    cashflow_raw = _fmp_get(f"cash-flow-statement/{tk}", {"limit": 10})

    if not income_raw and not balance_raw and not cashflow_raw:
        return None

    def to_df(rows):
        if not rows:
            return pd.DataFrame()
        exclude = {"date", "symbol", "reportedCurrency", "cik", "fillingDate",
                   "acceptedDate", "calendarYear", "period", "link", "finalLink"}
        data = {}
        for row in rows:
            date_str = row.get("date")
            if not date_str:
                continue
            try:
                col = pd.Timestamp(date_str)
            except Exception:
                continue
            col_data = {}
            for k, v in row.items():
                if k in exclude:
                    continue
                if isinstance(v, (int, float)):
                    col_data[k] = v
            data[col] = col_data
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df.reindex(sorted(df.columns, reverse=True), axis=1)

    return {
        "income": to_df(income_raw or []),
        "balance": to_df(balance_raw or []),
        "cashflow": to_df(cashflow_raw or []),
        "_source": "fmp",
    }


# ===========================================
# Smart routing entry points (what data_fetcher.py calls)
# ===========================================

def get_info_smart(ticker: str) -> Dict:
    """
    Get company info with smart routing.
    Path: SEC EDGAR (primary) -> yfinance (fill market cap & summary) -> fallback to yfinance only
    """
    market = detect_market(ticker)
    if market != "us":
        return {
            "ticker": ticker.upper(),
            "name": ticker.upper(),
            "error": f"Ticker '{ticker}' is not a US stock. This app only supports US-listed companies.",
            "_source": "none",
        }

    # PRIMARY: SEC EDGAR
    if USE_SEC:
        result = sec_get_company_info(ticker)
        if result:
            # SEC doesn't have market cap or business summary - fill from yfinance
            if not result.get("market_cap"):
                mcap = yf_get_market_cap(ticker)
                if mcap:
                    result["market_cap"] = mcap
            if not result.get("summary"):
                summary = yf_get_summary(ticker)
                if summary:
                    result["summary"] = summary
            return result

    # SECONDARY: FMP (only if enabled)
    if USE_FMP and _get_fmp_key():
        result = fmp_get_company_info(ticker)
        if result:
            return result

    # FALLBACK: yfinance
    result = yf_get_company_info(ticker)
    if result:
        return result

    return {
        "ticker": ticker.upper(),
        "name": ticker.upper(),
        "error": f"Could not fetch info for {ticker} from any source.",
        "_source": "none",
    }


def get_financials_smart(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Get financials with smart routing.
    Path: SEC EDGAR (primary) -> yfinance (fill gaps) -> yfinance (full fallback)
    """
    market = detect_market(ticker)
    if market != "us":
        return {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "error": f"Ticker '{ticker}' is not a US stock.",
            "_source": "none",
        }

    sec_result = None
    fmp_result = None
    yf_result = None

    # PRIMARY: SEC EDGAR
    if USE_SEC:
        sec_result = sec_get_financials(ticker)
        if sec_result and not sec_result["income"].empty \
                and not sec_result["balance"].empty \
                and not sec_result["cashflow"].empty:
            return sec_result

    # SECONDARY: FMP (only if enabled)
    if USE_FMP and _get_fmp_key():
        fmp_result = fmp_get_financials(ticker)
        if fmp_result and not fmp_result["income"].empty \
                and not fmp_result["balance"].empty \
                and not fmp_result["cashflow"].empty:
            return fmp_result

    # FALLBACK: yfinance
    yf_result = yf_get_financials(ticker)

    # Merge: prefer SEC/FMP for any non-empty statements, fill rest from yfinance
    if (sec_result or fmp_result) and yf_result:
        primary = sec_result or fmp_result
        merged = {
            "income": primary["income"] if not primary["income"].empty
                      else yf_result["income"],
            "balance": primary["balance"] if not primary["balance"].empty
                       else yf_result["balance"],
            "cashflow": primary["cashflow"] if not primary["cashflow"].empty
                        else yf_result["cashflow"],
            "_source": f"{primary['_source']}+yfinance_merged",
        }
        if not merged["income"].empty:
            return merged

    # Last resort: SEC partial, no yfinance
    if sec_result and not sec_result["income"].empty:
        return sec_result

    # Last resort: yfinance only
    if yf_result and not yf_result["income"].empty:
        return yf_result

    return {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "error": f"Could not fetch financials for {ticker}.",
        "_source": "none",
    }
