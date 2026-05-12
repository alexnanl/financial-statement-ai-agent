"""
data_provider.py - Unified data adapter for US stocks

Design:
    - PRIMARY source:  FMP (Financial Modeling Prep) free tier
                       Used whenever FMP_API_KEY is configured.
                       Faster, broader coverage, fewer rate limits.
    - FALLBACK source: yfinance (Yahoo Finance)
                       Free, no API key required. Used when:
                         * FMP_API_KEY is not configured, OR
                         * FMP returns an error / partial result
                       In the partial case, missing pieces are merge-filled from yfinance.

    All sources return a unified dict shape so downstream code is agnostic.

Routing:
    US stocks  -> FMP (primary) -> yfinance (fallback)
    Non-US     -> rejected with a clear error (this build is US-only)

Diagnostics:
    Every call to FMP logs its outcome into _FMP_DIAG (last 20 calls).
    Call diagnose() to see what happened on the most recent fetches.
    The Streamlit UI surfaces this via a "Data source diagnostics" expander.

Dependencies:
    pip install yfinance      # Yahoo Finance
    requests                  # FMP HTTP calls

Environment variables (optional but strongly recommended):
    FMP_API_KEY               # Get a free key at
                              # https://site.financialmodelingprep.com
                              # Free tier: 250 calls/day
"""
import os
import time
import random
from collections import deque
from typing import Dict, Optional
import pandas as pd
import streamlit as st


# ===========================================
# Diagnostic log (ring buffer of last 20 FMP calls)
# ===========================================
_FMP_DIAG = deque(maxlen=20)


def _log(endpoint: str, status: str, detail: str = "") -> None:
    """Record an FMP call outcome for diagnostics."""
    _FMP_DIAG.append({
        "endpoint": endpoint,
        "status": status,
        "detail": detail,
        "ts": time.strftime("%H:%M:%S"),
    })


def diagnose() -> Dict:
    """
    Return current data-source diagnostics. The UI can use this to show the user
    what's actually happening with FMP and why a request might have fallen back
    to yfinance.
    """
    key = _get_fmp_key()
    return {
        "fmp_key_present": bool(key),
        "fmp_key_length": len(key) if key else 0,
        "fmp_key_source": _get_fmp_key_source(),
        "recent_calls": list(_FMP_DIAG),
    }


def clear_diagnostics() -> None:
    """Reset the diagnostic log."""
    _FMP_DIAG.clear()


# ===========================================
# Market detection (US-only build)
# ===========================================

def detect_market(ticker: str) -> str:
    """
    Detect market from ticker.
    This build only supports US stocks; everything else returns 'unsupported'.
    """
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
# FMP (Financial Modeling Prep)
# ===========================================

# Note: We use FMP API v3 endpoints throughout, which are the most
# compatible with the free tier.
FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _get_fmp_key() -> Optional[str]:
    """Get FMP API key from Streamlit secrets or environment variable."""
    # 1) Streamlit secrets
    try:
        key = st.secrets.get("FMP_API_KEY", None)
        if key and isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    # 2) Environment variable
    env_key = os.environ.get("FMP_API_KEY", "").strip()
    if env_key:
        return env_key
    return None


def _get_fmp_key_source() -> str:
    """Tell us WHERE the FMP key was found (for diagnostics)."""
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
    """
    Hit FMP REST endpoint. Returns parsed JSON list, or None on failure.
    All outcomes (success/failure) are logged to _FMP_DIAG.
    """
    import requests
    key = _get_fmp_key()
    if not key:
        _log(endpoint, "skip", "FMP_API_KEY not configured")
        return None

    params = params or {}
    params["apikey"] = key

    try:
        resp = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=10)
    except requests.exceptions.Timeout:
        _log(endpoint, "error", "request timed out (>10s)")
        return None
    except requests.exceptions.ConnectionError as e:
        _log(endpoint, "error", f"connection error: {str(e)[:80]}")
        return None
    except Exception as e:
        _log(endpoint, "error", f"{type(e).__name__}: {str(e)[:80]}")
        return None

    # HTTP-level errors
    if resp.status_code == 401:
        _log(endpoint, "auth_error", "401 - invalid or missing API key")
        return None
    if resp.status_code == 403:
        _log(endpoint, "paywall", "403 - endpoint requires paid plan")
        return None
    if resp.status_code == 429:
        _log(endpoint, "rate_limit", "429 - daily quota exceeded (250/day on free tier)")
        return None
    if not resp.ok:
        _log(endpoint, "http_error", f"HTTP {resp.status_code}: {resp.text[:100]}")
        return None

    # Parse JSON
    try:
        data = resp.json()
    except Exception as e:
        _log(endpoint, "parse_error", f"invalid JSON: {str(e)[:80]}")
        return None

    # FMP-level error messages
    if isinstance(data, dict):
        if "Error Message" in data:
            _log(endpoint, "fmp_error", str(data["Error Message"])[:120])
            return None
        if "message" in data and not data.get("symbol"):
            _log(endpoint, "fmp_error", str(data["message"])[:120])
            return None

    if isinstance(data, list):
        if len(data) == 0:
            _log(endpoint, "empty", "API returned empty list")
            return None
        _log(endpoint, "ok", f"received {len(data)} record(s)")
        return data

    _log(endpoint, "unexpected", f"unexpected response type: {type(data).__name__}")
    return None


def fmp_get_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company profile from FMP. Returns None if not available."""
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


def _fmp_statement_to_df(rows: list, value_keys_exclude: list = None) -> pd.DataFrame:
    """
    Convert FMP statement rows (list of dicts) to a DataFrame.
    Columns = report dates (Timestamps), Index = field names.
    """
    if not rows:
        return pd.DataFrame()

    exclude = set(value_keys_exclude or []) | {
        "date", "symbol", "reportedCurrency", "cik", "fillingDate",
        "acceptedDate", "calendarYear", "period", "link", "finalLink"
    }

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
    # Sort columns descending (most recent first), matching yfinance convention
    df = df.reindex(sorted(df.columns, reverse=True), axis=1)
    return df


def fmp_get_financials(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch the three core financial statements from FMP. Returns None on failure."""
    tk = ticker.upper()
    income_raw = _fmp_get(f"income-statement/{tk}", {"limit": 10})
    balance_raw = _fmp_get(f"balance-sheet-statement/{tk}", {"limit": 10})
    cashflow_raw = _fmp_get(f"cash-flow-statement/{tk}", {"limit": 10})

    if not income_raw and not balance_raw and not cashflow_raw:
        return None

    return {
        "income": _fmp_statement_to_df(income_raw or []),
        "balance": _fmp_statement_to_df(balance_raw or []),
        "cashflow": _fmp_statement_to_df(cashflow_raw or []),
        "_source": "fmp",
    }


# ===========================================
# yfinance (fallback)
# ===========================================

def yf_get_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company info via yfinance."""
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


def yf_get_financials(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch three financial statements via yfinance."""
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
# Smart routing entry points
#
# Strategy:
#   Primary:  FMP (when FMP_API_KEY is set)  - faster, broader, no rate-limit
#   Fallback: yfinance                       - free, no key required
#
# Both sources return the same dict shape so downstream code is agnostic.
# When FMP returns partial data (e.g. balance sheet empty), the missing
# pieces are filled in from yfinance ("merge fallback") rather than
# discarding the FMP data outright.
# ===========================================

def setup_status() -> Dict[str, bool]:
    """Tell the UI whether FMP is configured (primary) or only yfinance is available."""
    return {
        "fmp_configured": bool(_get_fmp_key()),
        "yfinance_available": True,
    }


def get_info_smart(ticker: str) -> Dict:
    """
    Get company info with smart routing.
    Primary: FMP. Fallback: yfinance.
    """
    market = detect_market(ticker)
    if market != "us":
        return {
            "ticker": ticker.upper(),
            "name": ticker.upper(),
            "error": f"Ticker '{ticker}' is not a US stock. This app only supports US-listed companies.",
            "_source": "none",
        }

    # PRIMARY: FMP
    if _get_fmp_key():
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
    Primary: FMP. Fallback: yfinance.

    Merge-fallback: if FMP returns a partial result (e.g. cashflow empty),
    the missing statements are filled in from yfinance.
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

    fmp_result = None
    yf_result = None

    # PRIMARY: FMP
    if _get_fmp_key():
        fmp_result = fmp_get_financials(ticker)
        if fmp_result and not fmp_result["income"].empty \
                and not fmp_result["balance"].empty \
                and not fmp_result["cashflow"].empty:
            return fmp_result

    # FALLBACK: yfinance (also used to fill FMP gaps)
    yf_result = yf_get_financials(ticker)

    # Case 1: FMP partial, yfinance available -> merge
    if fmp_result and yf_result:
        merged = {
            "income": fmp_result["income"] if not fmp_result["income"].empty
                      else yf_result["income"],
            "balance": fmp_result["balance"] if not fmp_result["balance"].empty
                       else yf_result["balance"],
            "cashflow": fmp_result["cashflow"] if not fmp_result["cashflow"].empty
                        else yf_result["cashflow"],
            "_source": "fmp+yfinance_merged",
        }
        if not merged["income"].empty:
            return merged

    # Case 2: FMP succeeded fully (rare since we'd have returned above) - return it
    if fmp_result and not fmp_result["income"].empty:
        return fmp_result

    # Case 3: only yfinance worked
    if yf_result and not yf_result["income"].empty:
        return yf_result

    return {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "error": f"Could not fetch financials for {ticker}.",
        "_source": "none",
    }
