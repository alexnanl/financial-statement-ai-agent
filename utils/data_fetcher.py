"""
Data fetcher module - US stocks only
Uses smart routing (FMP -> yfinance) via data_provider.
"""
import pandas as pd
import time
import random
from typing import Optional, Dict, List
import streamlit as st


def _yf_call_with_retry(func, *args, max_retries=3, **kwargs):
    """
    yfinance call wrapper with auto-retry + exponential backoff.
    Used to work around Yahoo Finance rate limits.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(x in err_str for x in ["rate", "429", "too many"]):
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue
            raise
    raise last_err


# US industry peer pools - used as fallback when LLM peer suggestion fails
INDUSTRY_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ORCL", "ADBE",
                   "CRM", "AVGO", "AMD", "INTC", "CSCO", "QCOM", "IBM", "TXN",
                   "INTU", "NOW", "PLTR", "SNOW", "DDOG"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW",
                           "TJX", "BKNG", "CMG", "ABNB", "F", "GM", "MAR",
                           "WHR", "DPZ", "YUM"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
                            "SCHW", "PYPL", "V", "MA", "BRK-B"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
                   "DHR", "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "VZ", "T", "CMCSA",
                                "TMUS", "CHTR", "EA", "TTWO"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "MDLZ", "PM", "CL",
                            "KMB", "GIS", "K", "STZ"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "VLO"],
    "Industrials": ["BA", "CAT", "HON", "UPS", "GE", "RTX", "LMT", "DE", "MMM",
                    "EMR", "ETN", "ITW", "CSX", "NSC", "FDX"],
    "Basic Materials": ["LIN", "SHW", "FCX", "NEM", "ECL", "APD", "DD", "DOW", "PPG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR"],
    "Utilities": ["NEE", "SO", "DUK", "AEP", "EXC", "SRE", "D", "PCG", "XEL"],
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_info(ticker: str) -> Dict:
    """
    Fetch company info with smart routing:
       US stocks -> FMP (if API key) -> yfinance fallback
    """
    try:
        from utils.data_provider import get_info_smart
        return get_info_smart(ticker)
    except Exception as e:
        return _fetch_company_info_yf_only(ticker, error=str(e))


def _fetch_company_info_yf_only(ticker: str, error: str = "") -> Dict:
    """yfinance-only fallback implementation."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            return tk.info or {}

        info = _yf_call_with_retry(_do_fetch)
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
            "_source": "yfinance_legacy",
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "name": ticker.upper(),
                "error": error or str(e), "_source": "none"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch annual financial statements with smart routing.
    Returns: {'income': IncomeStatement, 'balance': BalanceSheet, 'cashflow': CashFlow}
    """
    try:
        from utils.data_provider import get_financials_smart
        result = get_financials_smart(ticker)
        # Normalize: ensure all three keys exist
        for key in ["income", "balance", "cashflow"]:
            if key not in result:
                result[key] = pd.DataFrame()
        return result
    except Exception as e:
        return _fetch_financials_yf_only(ticker, error=str(e))


def _fetch_financials_yf_only(ticker: str, error: str = "") -> Dict[str, pd.DataFrame]:
    """yfinance-only fallback implementation."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            return {
                "income": tk.financials if tk.financials is not None else pd.DataFrame(),
                "balance": tk.balance_sheet if tk.balance_sheet is not None else pd.DataFrame(),
                "cashflow": tk.cashflow if tk.cashflow is not None else pd.DataFrame(),
                "_source": "yfinance_legacy",
            }

        return _yf_call_with_retry(_do_fetch)
    except Exception as e:
        return {
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "error": error or str(e),
            "_source": "none",
        }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_price(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical stock price (with rate-limit retry)."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            return tk.history(period=period)

        return _yf_call_with_retry(_do_fetch)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _get_market_cap_for_ticker(ticker: str) -> Optional[float]:
    """Lightweight market cap lookup for peer size matching."""
    try:
        import yfinance as yf

        def _do_fetch():
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            return info.get("marketCap")

        return _yf_call_with_retry(_do_fetch, max_retries=2)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _ai_suggest_peers(company_name: str, ticker: str, sector: str,
                       industry: str, country: str,
                       market_cap: Optional[float], n: int = 5) -> List[str]:
    """
    Use LLM to intelligently suggest peer companies.
    Returns list of US tickers like ['MSFT', 'GOOGL', ...]
    """
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []

    try:
        from openai import OpenAI
        import json as _json

        client = OpenAI(api_key=api_key)

        market_cap_desc = ""
        if market_cap:
            if market_cap > 1e11:
                market_cap_desc = f"Market cap ~${market_cap/1e9:.0f}B (mega cap)"
            elif market_cap > 1e10:
                market_cap_desc = f"Market cap ~${market_cap/1e9:.0f}B (large cap)"
            else:
                market_cap_desc = f"Market cap ~${market_cap/1e9:.1f}B (mid/small cap)"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial industry analyst specializing in US-listed companies. "
                        "Given a company, recommend its most direct US-listed competitors / "
                        "comparable peers.\n\n"
                        "Core principles:\n"
                        "1. Sub-industry first - e.g. for a payment processor, recommend Visa, "
                        "Mastercard, PayPal, not generic 'financial services' names\n"
                        "2. Similar scale - prefer peers in 1/3x to 3x market cap range\n"
                        "3. Real comparability - must have similar business model, not just same sector\n"
                        "4. US-listed only - only return tickers on NYSE/NASDAQ\n\n"
                        "Ticker format: just the symbol, e.g. AAPL, MSFT, BRK-B. "
                        "Do NOT include suffixes like .HK, .SS - only US tickers.\n\n"
                        "Return JSON: {\"peers\": [\"ticker1\", \"ticker2\", ...]}\n"
                        "Only tickers, no company names, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target company: {company_name} ({ticker})\n"
                        f"Sector: {sector} / {industry}\n"
                        f"Country: {country}\n"
                        f"{market_cap_desc}\n\n"
                        f"Please recommend {n} most direct US-listed peer competitors."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        result = _json.loads(resp.choices[0].message.content or "{}")
        peers = result.get("peers", [])
        # Filter: only US-style tickers (no foreign suffixes)
        peers = [p.upper().strip() for p in peers
                 if p and isinstance(p, str) and len(p) <= 8
                 and not any(p.upper().endswith(s) for s in
                             [".HK", ".SS", ".SZ", ".T", ".L", ".DE", ".PA", ".KS"])]
        # Exclude the target itself
        peers = [p for p in peers if p != ticker.upper()]
        return peers[:n]
    except Exception:
        return []


def get_peer_suggestions_by_size(sector: str, target_market_cap: Optional[float],
                                    exclude: str = "", n: int = 5,
                                    size_tolerance: float = 3.0,
                                    company_name: str = "",
                                    industry: str = "",
                                    country: str = "") -> List[Dict]:
    """
    Suggest peers by "industry + size" - LLM first, fallback to hardcoded pool.

    Returns: [{'ticker': 'MSFT', 'market_cap': 3.5e12, 'size_ratio': 1.05}, ...]
    """
    import math

    # ===== Priority 1: LLM smart suggestion =====
    ai_peers = []
    if company_name and (industry or sector):
        ai_peers = _ai_suggest_peers(
            company_name=company_name,
            ticker=exclude or "",
            sector=sector or "",
            industry=industry or "",
            country=country or "",
            market_cap=target_market_cap,
            n=n + 2,
        )

    ai_peers_clean = []
    seen_ai = set()
    for p in ai_peers:
        pu = p.upper()
        if pu in seen_ai or pu == (exclude or "").upper():
            continue
        seen_ai.add(pu)
        ai_peers_clean.append(p)

    # ===== Priority 2: Industry pool (used only when LLM gives too few) =====
    pool_peers = []
    seen_pool = set(seen_ai) | {(exclude or "").upper()}
    for p in INDUSTRY_PEERS.get(sector, []):
        if p.upper() in seen_pool:
            continue
        seen_pool.add(p.upper())
        pool_peers.append(p)

    # If no target market cap, return AI-priority list
    if not target_market_cap:
        all_picked = ai_peers_clean[:n]
        if len(all_picked) < n:
            all_picked.extend(pool_peers[: n - len(all_picked)])
        return [{"ticker": c, "market_cap": None, "size_ratio": None}
                for c in all_picked[:n]]

    # Score AI peers (keep all - AI already vetted relevance)
    ai_scored = []
    for c in ai_peers_clean:
        cap = _get_market_cap_for_ticker(c)
        if cap is None or cap <= 0:
            ai_scored.append({"ticker": c, "market_cap": None, "size_ratio": None,
                                "_sort_key": 999})
            continue
        ratio = cap / target_market_cap
        ai_scored.append({"ticker": c, "market_cap": cap, "size_ratio": ratio,
                            "_sort_key": abs(math.log(ratio))})

    ai_scored.sort(key=lambda x: x["_sort_key"])

    if len(ai_scored) >= n:
        return [{"ticker": s["ticker"], "market_cap": s["market_cap"],
                  "size_ratio": s["size_ratio"]} for s in ai_scored[:n]]

    # Score pool peers with size tolerance
    pool_scored = []
    for c in pool_peers:
        cap = _get_market_cap_for_ticker(c)
        if cap is None or cap <= 0:
            continue
        ratio = cap / target_market_cap
        if (1.0 / size_tolerance) <= ratio <= size_tolerance:
            pool_scored.append({"ticker": c, "market_cap": cap, "size_ratio": ratio,
                                  "_sort_key": abs(math.log(ratio))})
    pool_scored.sort(key=lambda x: x["_sort_key"])

    # If still not enough, relax size constraint
    if len(ai_scored) + len(pool_scored) < n:
        already = {s["ticker"] for s in pool_scored}
        backup = []
        for c in pool_peers:
            if c in already:
                continue
            cap = _get_market_cap_for_ticker(c)
            if cap is None:
                continue
            ratio = cap / target_market_cap
            backup.append({"ticker": c, "market_cap": cap, "size_ratio": ratio,
                            "_sort_key": abs(math.log(ratio))})
        backup.sort(key=lambda x: x["_sort_key"])
        pool_scored.extend(backup[: n - len(ai_scored) - len(pool_scored)])

    # Merge: AI first, pool as backfill
    final = []
    for s in ai_scored + pool_scored:
        final.append({"ticker": s["ticker"],
                       "market_cap": s["market_cap"],
                       "size_ratio": s["size_ratio"]})
        if len(final) >= n:
            break

    return final[:n]


# US company name -> ticker lookup table (covers major large-caps)
COMMON_NAME_MAP = {
    # ===== Tech =====
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN",
    "facebook": "META", "meta": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "snowflake": "SNOW",
    "palantir": "PLTR",
    "shopify": "SHOP",
    "spotify": "SPOT",
    "zoom": "ZM",
    "servicenow": "NOW",
    "datadog": "DDOG",
    "texas instruments": "TXN",
    "applied materials": "AMAT",
    "micron": "MU",

    # ===== Consumer =====
    "walmart": "WMT",
    "coca cola": "KO", "coca-cola": "KO", "coke": "KO",
    "pepsi": "PEP", "pepsico": "PEP",
    "mcdonalds": "MCD", "mcdonald's": "MCD",
    "starbucks": "SBUX",
    "disney": "DIS",
    "nike": "NKE",
    "procter & gamble": "PG", "procter and gamble": "PG", "p&g": "PG",
    "costco": "COST",
    "target": "TGT",
    "home depot": "HD",
    "lowes": "LOW", "lowe's": "LOW",
    "kraft heinz": "KHC",
    "general mills": "GIS",
    "kellogg": "K",
    "philip morris": "PM",
    "altria": "MO",
    "estee lauder": "EL",
    "mondelez": "MDLZ",
    "domino's": "DPZ", "dominos": "DPZ",
    "yum brands": "YUM",
    "chipotle": "CMG",
    "booking": "BKNG", "booking.com": "BKNG",
    "whirlpool": "WHR",
    "marriott": "MAR",

    # ===== Auto =====
    "ford": "F",
    "general motors": "GM", "gm": "GM",
    "stellantis": "STLA",
    "rivian": "RIVN",
    "lucid": "LCID",

    # ===== Financial =====
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman sachs": "GS", "goldman": "GS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C", "citi": "C",
    "morgan stanley": "MS",
    "american express": "AXP", "amex": "AXP",
    "blackrock": "BLK",
    "visa": "V",
    "mastercard": "MA",
    "berkshire hathaway": "BRK-B", "berkshire": "BRK-B",
    "charles schwab": "SCHW",

    # ===== Healthcare =====
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "merck": "MRK",
    "eli lilly": "LLY", "lilly": "LLY",
    "abbott": "ABT",
    "abbvie": "ABBV",
    "unitedhealth": "UNH", "united health": "UNH",
    "thermo fisher": "TMO",
    "danaher": "DHR",
    "bristol myers": "BMY", "bristol-myers": "BMY",
    "amgen": "AMGN",
    "gilead": "GILD",
    "cvs": "CVS",
    "medtronic": "MDT",
    "intuitive surgical": "ISRG",

    # ===== Energy =====
    "chevron": "CVX",
    "exxon": "XOM", "exxonmobil": "XOM",
    "conocophillips": "COP",
    "schlumberger": "SLB",
    "occidental": "OXY",

    # ===== Industrial / Aerospace =====
    "boeing": "BA",
    "caterpillar": "CAT",
    "general electric": "GE", "ge": "GE",
    "fedex": "FDX",
    "ups": "UPS",
    "delta airlines": "DAL", "delta": "DAL",
    "united airlines": "UAL",
    "american airlines": "AAL",
    "honeywell": "HON",
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "deere": "DE", "john deere": "DE",
    "3m": "MMM",

    # ===== Semiconductors =====
    "tsmc": "TSM",  # ADR listed on NYSE
}


def looks_like_ticker(s: str) -> bool:
    """
    Check if input looks like a stock ticker:
    - Length 1-8, no spaces
    - Only ASCII letters/digits/dots/dashes
    - Either all uppercase or contains digits
    """
    s = (s or "").strip()
    if not s or len(s) > 8 or " " in s:
        return False
    if not s.isascii():
        return False
    if not all(c.isalnum() or c in ".-" for c in s):
        return False
    has_digit = any(c.isdigit() for c in s)
    has_lower = any(c.islower() for c in s)
    if has_digit:
        return True
    return not has_lower


def _ai_identify_ticker(query: str) -> Optional[Dict]:
    """
    Last-resort: ask GPT to identify a US ticker from a company name.
    Returns {"ticker": "F", "name": "Ford Motor Company"} or None.
    """
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        import json as _json

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a US stock ticker lookup expert. The user gives you a "
                        "company name (English), and you return its US-listed ticker "
                        "symbol on NYSE or NASDAQ.\n"
                        "Rules:\n"
                        "- Return only US-listed tickers (NYSE/NASDAQ), e.g. AAPL, F, BRK-B\n"
                        "- Do NOT return foreign tickers (no .HK, .SS, .T etc.)\n"
                        "- If the company has only a foreign listing, try to find its US ADR "
                        "(e.g. TSM for Taiwan Semiconductor, BABA for Alibaba ADR)\n"
                        "- If no US ticker exists, return ticker=null\n"
                        "Return JSON: {\"ticker\": \"SYMBOL\", \"name\": \"Full Company Name\"}"
                    ),
                },
                {"role": "user", "content": f"Company: {query}"},
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        result = _json.loads(resp.choices[0].message.content or "{}")
        ticker = result.get("ticker")
        if not ticker or ticker.lower() in ("null", "none", ""):
            return None
        return {
            "ticker": ticker.upper(),
            "name": result.get("name", query),
        }
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def search_ticker_by_name(query: str) -> List[Dict]:
    """
    Search ticker by company name.
    1) Local quick table  2) Yahoo Finance search  3) GPT identification (US only)
    Returns: [{'ticker': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NMS', 'type': 'EQUITY'}, ...]
    """
    q = (query or "").strip()
    if not q:
        return []

    results = []
    seen_tickers = set()

    # 1) Local table - exact match
    q_lower = q.lower()
    if q_lower in COMMON_NAME_MAP:
        ticker = COMMON_NAME_MAP[q_lower]
        if ticker not in seen_tickers:
            results.append({
                "ticker": ticker,
                "name": q,
                "exchange": "NYSE/NASDAQ",
                "type": "EQUITY",
            })
            seen_tickers.add(ticker)

    # 2) Local table - fuzzy match
    if len(q_lower) >= 2:
        for key, ticker in COMMON_NAME_MAP.items():
            if ticker in seen_tickers:
                continue
            if q_lower in key.lower() or key.lower() in q_lower:
                results.append({
                    "ticker": ticker,
                    "name": key,
                    "exchange": "NYSE/NASDAQ",
                    "type": "EQUITY",
                })
                seen_tickers.add(ticker)
                if len(results) >= 5:
                    break

    # 3) Yahoo Finance search API
    try:
        import requests
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(url, params={"q": q, "quotesCount": 6, "newsCount": 0},
                             headers=headers, timeout=8)
        if resp.ok:
            data = resp.json()
            for item in data.get("quotes", []):
                if item.get("quoteType") not in ("EQUITY",):
                    continue
                ticker = item.get("symbol", "")
                if not ticker or ticker in seen_tickers:
                    continue
                # Filter out non-US tickers
                if any(ticker.upper().endswith(s) for s in
                       [".HK", ".SS", ".SZ", ".T", ".L", ".DE", ".PA", ".KS",
                        ".SW", ".MI", ".AS", ".TO", ".AX"]):
                    continue
                results.append({
                    "ticker": ticker,
                    "name": item.get("longname") or item.get("shortname") or ticker,
                    "exchange": item.get("exchDisp") or item.get("exchange", ""),
                    "type": item.get("quoteType", ""),
                })
                seen_tickers.add(ticker)
    except Exception:
        pass

    # 4) GPT identification (last resort)
    if not results:
        ai_result = _ai_identify_ticker(q)
        if ai_result and ai_result.get("ticker"):
            ticker = ai_result["ticker"]
            results.append({
                "ticker": ticker,
                "name": ai_result.get("name", q),
                "exchange": "NYSE/NASDAQ",
                "type": "EQUITY",
                "source": "ai",
            })

    return results[:8]
