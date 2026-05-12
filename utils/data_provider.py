"""
data_provider.py - Unified data adapter for US stocks

Updated version: uses FMP Stable API endpoints first.

Design:
    - PRIMARY source:  FMP (Financial Modeling Prep) Stable API
                       Used whenever FMP_API_KEY is configured and the selected
                       endpoints are available under the user's FMP plan.
    - FALLBACK source: yfinance (Yahoo Finance)
                       Used when:
                         * FMP_API_KEY is not configured, OR
                         * FMP returns 403 / paywall / empty / partial data, OR
                         * FMP has a temporary API/network issue.
                       In the partial case, missing pieces are merge-filled from yfinance.

Routing:
    US stocks  -> FMP Stable API (primary) -> yfinance (fallback)
    Non-US     -> rejected with a clear error (this build is US-only)

Streamlit Secrets / Environment variables:
    FMP_API_KEY
"""
import os
import time
import random
from typing import Dict, Optional, Any
import pandas as pd
import streamlit as st


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
    # US tickers are plain alphanumeric (with optional - or .) and no foreign suffix
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
# FMP Stable API
# ===========================================

# Newer FMP API style:
#   https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&limit=10&apikey=...
FMP_BASE = "https://financialmodelingprep.com/stable"


def _get_fmp_key() -> Optional[str]:
    """Get FMP API key from Streamlit secrets or environment variable."""
    try:
        key = st.secrets.get("FMP_API_KEY", None)
        if key:
            return str(key).strip()
    except Exception:
        pass
    key = os.environ.get("FMP_API_KEY")
    return key.strip() if key else None


def _log_fmp_call(endpoint: str, symbol: str, status: str, detail: str = "") -> None:
    """
    Store a small FMP call log in Streamlit session_state for debugging.
    Safe to ignore outside Streamlit.
    """
    try:
        if "fmp_call_log" not in st.session_state:
            st.session_state.fmp_call_log = []
        st.session_state.fmp_call_log.insert(0, {
            "time": time.strftime("%H:%M:%S"),
            "endpoint": endpoint,
            "symbol": symbol,
            "status": status,
            "detail": detail,
        })
        st.session_state.fmp_call_log = st.session_state.fmp_call_log[:20]
    except Exception:
        pass


def _extract_error_message(data: Any) -> str:
    """Try to extract a readable error message from FMP JSON."""
    if isinstance(data, dict):
        for key in ["Error Message", "error", "message", "Message"]:
            if key in data and data[key]:
                return str(data[key])
    return ""


def _fmp_get(endpoint: str, params: Optional[Dict] = None) -> Optional[list]:
    """
    Hit FMP Stable REST endpoint.

    Returns parsed JSON list on success, or None on failure / paywall / empty response.
    Example:
        _fmp_get("income-statement", {"symbol": "AAPL", "limit": 10})
    """
    import requests

    key = _get_fmp_key()
    symbol = ""
    if params and params.get("symbol"):
        symbol = str(params.get("symbol"))

    if not key:
        _log_fmp_call(endpoint, symbol, "skipped", "missing FMP_API_KEY")
        return None

    params = dict(params or {})
    params["apikey"] = key

    try:
        resp = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=12)

        if resp.status_code == 403:
            _log_fmp_call(endpoint, symbol, "paywall", "403 - endpoint requires paid plan or is not available for this key")
            return None
        if resp.status_code == 401:
            _log_fmp_call(endpoint, symbol, "auth_error", "401 - invalid or unauthorized API key")
            return None
        if resp.status_code == 429:
            _log_fmp_call(endpoint, symbol, "rate_limit", "429 - rate limit reached")
            return None
        if not resp.ok:
            _log_fmp_call(endpoint, symbol, "http_error", f"HTTP {resp.status_code}")
            return None

        data = resp.json()
        err_msg = _extract_error_message(data)
        if err_msg:
            status = "paywall" if "paid" in err_msg.lower() or "premium" in err_msg.lower() else "api_error"
            _log_fmp_call(endpoint, symbol, status, err_msg[:160])
            return None

        if isinstance(data, list):
            if len(data) == 0:
                _log_fmp_call(endpoint, symbol, "empty", "empty list returned")
                return None
            _log_fmp_call(endpoint, symbol, "success", f"{len(data)} rows")
            return data

        # Some FMP endpoints may return a single dict; normalize to list.
        if isinstance(data, dict) and data:
            _log_fmp_call(endpoint, symbol, "success", "single object")
            return [data]

        _log_fmp_call(endpoint, symbol, "empty", "unexpected empty response")
        return None
    except Exception as e:
        _log_fmp_call(endpoint, symbol, "exception", f"{type(e).__name__}: {str(e)[:120]}")
        return None


def _first_nonempty(row: Dict, keys, default=None):
    """Return first non-empty value from a dict for several possible FMP field names."""
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "None"):
            return v
    return default


def fmp_get_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company profile from FMP Stable API. Returns None if not available."""
    tk = ticker.upper().strip()
    data = _fmp_get("profile", {"symbol": tk})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    p = data[0]
    return {
        "ticker": tk,
        "name": _first_nonempty(p, ["companyName", "company_name", "name"], tk),
        "sector": _first_nonempty(p, ["sector"], "N/A"),
        "industry": _first_nonempty(p, ["industry"], "N/A"),
        "country": _first_nonempty(p, ["country"], "US"),
        "currency": _first_nonempty(p, ["currency", "reportedCurrency"], "USD"),
        "market_cap": _first_nonempty(p, ["mktCap", "marketCap", "market_cap"], None),
        "summary": _first_nonempty(p, ["description", "companyDescription", "longBusinessSummary"], ""),
        "exchange": _first_nonempty(p, ["exchangeShortName", "exchange", "exchangeFullName"], "N/A"),
        "website": _first_nonempty(p, ["website", "site"], ""),
        "_source": "fmp_stable",
    }


def _fmp_statement_to_df(rows: list, value_keys_exclude: list = None) -> pd.DataFrame:
    """
    Convert FMP statement rows (list of dicts) to a DataFrame.
    Columns = report dates (Timestamps), Index = field names.
    """
    if not rows:
        return pd.DataFrame()

    exclude = set(value_keys_exclude or []) | {
        "date", "symbol", "reportedCurrency", "cik",
        "fillingDate", "filingDate", "acceptedDate",
        "calendarYear", "fiscalYear", "period",
        "link", "finalLink", "source", "url",
    }

    data = {}
    for row in rows:
        date_str = row.get("date") or row.get("filingDate") or row.get("fillingDate")
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
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                col_data[k] = v
        if col_data:
            data[col] = col_data

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Sort columns descending (most recent first), matching yfinance convention
    df = df.reindex(sorted(df.columns, reverse=True), axis=1)
    return df


def fmp_get_financials(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Fetch the three core financial statements from FMP Stable API."""
    tk = ticker.upper().strip()

    # Stable API uses symbol as query parameter instead of path parameter.
    common = {"symbol": tk, "limit": 10, "period": "annual"}
    income_raw = _fmp_get("income-statement", common)
    balance_raw = _fmp_get("balance-sheet-statement", common)
    cashflow_raw = _fmp_get("cash-flow-statement", common)

    if not income_raw and not balance_raw and not cashflow_raw:
        return None

    return {
        "income": _fmp_statement_to_df(income_raw or []),
        "balance": _fmp_statement_to_df(balance_raw or []),
        "cashflow": _fmp_statement_to_df(cashflow_raw or []),
        "_source": "fmp_stable",
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
# ===========================================

def setup_status() -> Dict[str, bool]:
    """Tell the UI whether FMP is configured (primary) or only yfinance is available."""
    return {
        "fmp_configured": bool(_get_fmp_key()),
        "fmp_base": FMP_BASE,
        "yfinance_available": True,
    }


def get_info_smart(ticker: str) -> Dict:
    """
    Get company info with smart routing.
    Primary: FMP Stable API. Fallback: yfinance.
    """
    market = detect_market(ticker)
    if market != "us":
        return {
            "ticker": ticker.upper(),
            "name": ticker.upper(),
            "error": f"Ticker '{ticker}' is not a US stock. This app only supports US-listed companies.",
            "_source": "none",
        }

    # PRIMARY: FMP Stable API
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
    Primary: FMP Stable API. Fallback: yfinance.

    Merge-fallback: if FMP returns a partial result, the missing statements are
    filled from yfinance instead of discarding the usable FMP parts.
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

    # PRIMARY: FMP Stable API
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
                      else yf_result.get("income", pd.DataFrame()),
            "balance": fmp_result["balance"] if not fmp_result["balance"].empty
                       else yf_result.get("balance", pd.DataFrame()),
            "cashflow": fmp_result["cashflow"] if not fmp_result["cashflow"].empty
                        else yf_result.get("cashflow", pd.DataFrame()),
            "_source": "fmp_stable+yfinance_merged",
        }
        if not merged["income"].empty:
            return merged

    # Case 2: FMP succeeded at least for income statement
    if fmp_result and not fmp_result["income"].empty:
        return fmp_result

    # Case 3: only yfinance worked
    if yf_result and not yf_result.get("income", pd.DataFrame()).empty:
        return yf_result

    return {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "error": f"Could not fetch financials for {ticker}.",
        "_source": "none",
    }
