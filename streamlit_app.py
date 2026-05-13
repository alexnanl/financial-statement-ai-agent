"""
Financial Analysis AI Agent - Streamlit main entry point.
US stocks only. English-only UI.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_fetcher import (
    fetch_company_info, fetch_financials,
    get_peer_suggestions_by_size, search_ticker_by_name, looks_like_ticker
)
from utils.ratios import (
    compute_ratios_for_year, compute_multi_year_ratios, dupont_analysis,
    is_percentage_metric,
)
from utils.benchmark import compare_with_peers, benchmark_analysis
from utils.charts import (
    plot_trend, plot_peer_comparison, plot_dupont_decomposition,
    plot_dupont_waterfall, plot_radar
)
from utils.report import generate_report
from utils.data_provider import setup_status


# ===== Page config =====
st.set_page_config(
    page_title="Financial Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Custom styles =====
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1300px; }
    h1 { color: #1f4e79; font-weight: 700; }
    h2 { color: #1f4e79; border-bottom: 2px solid #c00000; padding-bottom: 0.3rem; }
    h3 { color: #2c5282; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; color: #1f4e79; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f5f7fa;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
    .stAlert { border-left: 4px solid #1f4e79; }
</style>
""", unsafe_allow_html=True)


# ===== Mode selection =====
if "mode" not in st.session_state:
    st.session_state.mode = "ai"  # Default: AI chat

with st.sidebar:
    mode_choice = st.radio(
        "Mode",
        options=["🤖 AI Chat", "📊 Classic"],
        index=0 if st.session_state.mode == "ai" else 1,
        key="mode_radio",
    )
    st.session_state.mode = "ai" if "AI" in mode_choice else "classic"

    # Data source status indicator
    status = setup_status()
    if status["fmp_configured"]:
        st.caption("📡 Data: **FMP** (primary) + yfinance (fallback)")
    else:
        st.caption("📡 Data: **yfinance** only "
                    "(add `FMP_API_KEY` for better reliability)")

    st.markdown("---")

MODE = st.session_state.mode


# ===== Title =====
st.title("📊 Financial Analysis AI Agent")
st.caption("Enter a US company name or ticker to generate ratio analysis, "
            "DuPont decomposition, trends, peer comparison, and a downloadable report.")


# ===== AI chat mode - render chat page and exit =====
if MODE == "ai":
    from utils.chat_page import render_chat_page
    render_chat_page()
    st.stop()


# ===== Classic analysis mode =====


# ===== Sidebar - input =====
with st.sidebar:
    st.header("⚙️ Analysis Parameters")

    query = st.text_input(
        "Company Name / Ticker",
        value="Apple",
        placeholder="e.g. Apple, AAPL, Tesla, Microsoft",
        help="Accepts company names or US stock tickers (NYSE / NASDAQ)",
    ).strip()

    ticker = None
    if query:
        if looks_like_ticker(query):
            ticker = query.upper()
            st.caption(f"Using ticker: **{ticker}**")
        else:
            with st.spinner("Searching..."):
                matches = search_ticker_by_name(query)
            if not matches:
                st.error(f"No match found for '{query}'. "
                          "Try the full company name or ticker.")
            elif len(matches) == 1:
                m = matches[0]
                ticker = m["ticker"]
                st.success(f"{m['name']} ({ticker})")
            else:
                options = [f"{m['ticker']} - {m['name']} ({m.get('exchange','')})"
                           for m in matches]
                choice = st.selectbox("Multiple matches - please select:",
                                       options, index=0)
                ticker = choice.split(" - ")[0].strip()

    target_year = st.number_input("Target Year",
                                    min_value=2010, max_value=2025, value=2024)
    num_years = st.slider("Years of Trend Data", min_value=2, max_value=8, value=5)
    st.caption("yfinance typically provides ~4 years of annual data; "
                "FMP free tier provides ~5 years. "
                "Actual range may be shorter than your setting.")

    st.markdown("---")
    st.subheader("Peer Companies")
    peer_input = st.text_area(
        "Peer company names (one per line)",
        value="",
        height=120,
        placeholder="e.g.\nMicrosoft\nGoogle\nAmazon",
    )
    auto_peers = st.checkbox("Auto-suggest peers (by industry + size)", value=True)

    st.markdown("---")
    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    if run:
        # Persist the Classic analysis state across Streamlit reruns.
        # Download buttons trigger a rerun by default, and without this flag
        # the app would return to the initial welcome screen.
        st.session_state["classic_analysis_has_run"] = True

    with st.expander("ℹ️ Input Examples"):
        st.markdown("""
        **Company names**:
        - Apple → AAPL
        - Microsoft → MSFT
        - Tesla → TSLA
        - Berkshire Hathaway → BRK-B

        **Ticker formats**:
        - Just the symbol: `AAPL`, `TSLA`, `MSFT`, `BRK-B`
        - This app only supports US stocks listed on NYSE / NASDAQ.
        """)


# ===== Main area - welcome screen =====
# Keep showing the most recent Classic analysis after non-input reruns
# such as downloading Markdown / HTML / Word / PDF reports.
has_classic_analysis = st.session_state.get("classic_analysis_has_run", False)
if not run and not has_classic_analysis:
    st.info("👈 Enter a company name (e.g. Apple, Tesla) or ticker on the left, "
             "then click **Run Analysis**.")

    with st.expander("📖 What This Does", expanded=True):
        status = setup_status()
        if status["fmp_configured"]:
            data_status = ("✅ **FMP is configured** as the primary data source "
                            "on this deployment.")
        else:
            data_status = ("⚠️ **FMP is not configured.** Running on yfinance only. "
                            "Adding `FMP_API_KEY` to Streamlit Secrets significantly "
                            "improves data coverage and avoids Yahoo rate limits.")

        st.markdown(f"""
        ### Features

        1. **Ratio Analysis** - Profitability, efficiency, solvency, cash flow (15+ metrics)
        2. **DuPont Analysis** - ROE decomposition (Net Margin × Asset Turnover × Equity Multiplier)
        3. **Trend Analysis** - Multi-year trends with interactive charts
        4. **Peer Comparison** - Auto-matched by industry + market cap
        5. **Benchmark Analysis** - Excellent / Good / Fair / Weak ratings vs. generic thresholds
        6. **AI Report** - Auto-generated downloadable report (Markdown / HTML / Word / PDF)

        ### Data Sources

        ```text
        US Stocks (NYSE / NASDAQ)
         ├─ Primary:  FMP (Financial Modeling Prep) - 5 yrs of data, fewer rate limits
         └─ Fallback: yfinance (Yahoo Finance) - 4 yrs, free, no key needed
        ```

        {data_status}
        """)
    st.stop()

if not ticker:
    st.error("❌ Please enter a valid company name or US ticker first.")
    st.stop()


# ===== Run analysis =====
with st.spinner(f"📡 Fetching data for {ticker}..."):
    info = fetch_company_info(ticker)
    financials = fetch_financials(ticker)

if "error" in info or financials.get("income", pd.DataFrame()).empty:
    st.error(f"❌ Could not fetch data for {ticker}. "
              "Make sure it's a valid US-listed ticker (NYSE / NASDAQ).")
    if "error" in info:
        st.code(info.get("error"))
    st.stop()

# Data source label (shown small under the company header)
src = info.get("_source") or financials.get("_source", "unknown")
SRC_LABELS = {
    "fmp": "FMP",
    "yfinance": "yfinance",
    "yfinance_legacy": "yfinance",
    "fmp+yfinance_merged": "FMP + yfinance (merged)",
}
src_label = SRC_LABELS.get(src, src)


# ===== Company card =====
st.markdown(f"## {info['name']} ({info['ticker']})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sector", info.get("sector", "N/A"))
c2.metric("Industry", (info.get("industry") or "N/A")[:25])
c3.metric("Country", info.get("country", "N/A"))
mcap = info.get("market_cap")
mcap_str = f"${mcap/1e9:.1f}B {info.get('currency', 'USD')}" if mcap else "N/A"
c4.metric("Market Cap", mcap_str)

st.caption(f"📡 Data source: **{src_label}**")

if info.get("summary"):
    with st.expander("📝 Business Summary"):
        st.write(info["summary"])

# ===== Pick target year =====
income_df = financials["income"]
balance_df = financials["balance"]
cashflow_df = financials["cashflow"]

cols = sorted(income_df.columns, reverse=True)
cols_filtered = [c for c in cols if c.year <= target_year]
if not cols_filtered:
    avail = str([c.year for c in cols])
    st.error(f"⚠️ No data found for {target_year} or earlier. Available: {avail}")
    st.stop()

year_col = cols_filtered[0]
prev_col = cols_filtered[1] if len(cols_filtered) > 1 else None
actual_year = year_col.year

if actual_year != target_year:
    st.warning(f"⚠️ {target_year} fiscal data not yet available; using {actual_year}.")


# ===== Compute =====
ratios = compute_ratios_for_year(income_df, balance_df, cashflow_df, year_col, prev_col)
dupont = dupont_analysis(ratios)
trend_df = compute_multi_year_ratios(financials, target_year, num_years=num_years)


# ===== Helper: format value for display =====
def fmt_display(v, metric_name):
    if v is None or pd.isna(v):
        return "N/A"
    if is_percentage_metric(metric_name):
        return f"{v*100:.2f}%"
    if abs(v) > 1e6:
        return f"{v/1e6:.1f}M"
    return f"{v:.3f}"


# ===== Tab layout =====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🔻 DuPont",
    "📉 Trends",
    "🆚 Peers",
    "🎯 Benchmark",
    "📄 Report",
])


# ===== Tab 1: Overview =====
with tab1:
    st.subheader(f"Key Metrics - FY{actual_year}")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("ROE", f"{(ratios.get('ROE (Return on Equity)') or 0)*100:.2f}%")
    cc2.metric("ROA", f"{(ratios.get('ROA (Return on Assets)') or 0)*100:.2f}%")
    cc3.metric("Net Margin", f"{(ratios.get('Net Margin') or 0)*100:.2f}%")
    cc4.metric("Gross Margin", f"{(ratios.get('Gross Margin') or 0)*100:.2f}%")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Current Ratio", f"{ratios.get('Current Ratio') or 0:.2f}")
    cc2.metric("Quick Ratio", f"{ratios.get('Quick Ratio') or 0:.2f}")
    cc3.metric("Debt/Assets", f"{(ratios.get('Debt to Assets') or 0)*100:.2f}%")
    cc4.metric("Asset Turnover", f"{ratios.get('Asset Turnover') or 0:.2f}")

    st.markdown("---")
    st.markdown("### All Ratios")
    rows = []
    for k, v in ratios.items():
        if k.startswith("_"):
            continue
        rows.append({"Metric": k, "Value": fmt_display(v, k)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===== Tab 2: DuPont =====
with tab2:
    st.subheader("DuPont: ROE Decomposition")
    st.latex(r"ROE = \text{Net Margin} \times \text{Asset Turnover} \times \text{Equity Multiplier}")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Net Margin", f"{(dupont.get('Net Margin') or 0)*100:.2f}%")
    cc2.metric("Asset Turnover", f"{dupont.get('Asset Turnover') or 0:.2f}")
    cc3.metric("Equity Multiplier", f"{dupont.get('Equity Multiplier') or 0:.2f}")
    cc4.metric("ROE (DuPont)", f"{(dupont.get('ROE (DuPont)') or 0)*100:.2f}%")

    st.plotly_chart(plot_dupont_waterfall(dupont, actual_year), use_container_width=True)

    if trend_df is not None and not trend_df.empty:
        dupont_history = {}
        for col in trend_df.columns:
            yr_ratios = trend_df[col].to_dict()
            dupont_history[col] = {
                "Net Margin": yr_ratios.get("Net Margin"),
                "Asset Turnover": yr_ratios.get("Asset Turnover"),
                "Equity Multiplier": yr_ratios.get("Equity Multiplier"),
                "ROE (DuPont)": yr_ratios.get("ROE (Return on Equity)"),
            }
        st.plotly_chart(plot_dupont_decomposition(dupont_history),
                        use_container_width=True)


# ===== Tab 3: Trends =====
with tab3:
    n = len(trend_df.columns) if not trend_df.empty else 0
    st.subheader(f"Last {n} Years Trends")

    if n < num_years:
        st.info(f"You requested {num_years} years, but the data source only "
                 f"provides {n} years of annual reports for {ticker}. "
                 "This is a data-source limitation.")

    if trend_df.empty:
        st.warning("Insufficient trend data")
    else:
        st.plotly_chart(plot_trend(trend_df,
            ["Gross Margin", "Operating Margin", "Net Margin"],
            title="Profitability Trends"), use_container_width=True)
        st.plotly_chart(plot_trend(trend_df,
            ["ROE (Return on Equity)", "ROA (Return on Assets)"],
            title="Return Trends"), use_container_width=True)
        st.plotly_chart(plot_trend(trend_df,
            ["Current Ratio", "Quick Ratio", "Debt to Assets"],
            title="Solvency Trends"), use_container_width=True)

        with st.expander("📋 Trend Data Details"):
            display_df = trend_df.copy().astype(object)
            for idx in display_df.index:
                for col in display_df.columns:
                    display_df.at[idx, col] = fmt_display(trend_df.loc[idx, col], idx)
            st.dataframe(display_df, use_container_width=True)


# ===== Tab 4: Peer comparison =====
compare_df = pd.DataFrame()
with tab4:
    st.subheader("Peer Comparison (matched by industry + size)")

    raw_peers = [p.strip() for p in peer_input.split("\n") if p.strip()]
    peer_list = []
    failed_peers = []
    for p in raw_peers:
        if looks_like_ticker(p):
            peer_list.append(p.upper())
        else:
            matches = search_ticker_by_name(p)
            if matches:
                peer_list.append(matches[0]["ticker"])
            else:
                failed_peers.append(p)
    if failed_peers:
        st.warning(f"⚠️ Couldn't recognize: {', '.join(failed_peers)}")

    if not peer_list and auto_peers:
        with st.spinner("Matching peers by industry & size..."):
            suggested = get_peer_suggestions_by_size(
                sector=info.get("sector", ""),
                target_market_cap=info.get("market_cap"),
                exclude=ticker,
                n=4,
                company_name=info.get("name", ""),
                industry=info.get("industry", ""),
                country=info.get("country", ""),
            )
        peer_list = [s["ticker"] for s in suggested]

        if peer_list:
            peers_with_size = []
            for s in suggested:
                tk = s["ticker"]
                cap = s.get("market_cap")
                if cap:
                    if cap > 1e12:
                        size_str = f"${cap/1e12:.2f}T"
                    elif cap > 1e9:
                        size_str = f"${cap/1e9:.1f}B"
                    else:
                        size_str = f"${cap/1e6:.0f}M"
                    peers_with_size.append(f"{tk} ({size_str})")
                else:
                    peers_with_size.append(tk)
            st.info(f"🤖 Auto-selected peers in **{info.get('sector', 'N/A')}** "
                     f"with similar market cap: {', '.join(peers_with_size)}")

    if not peer_list:
        st.warning("Add at least one peer company in the sidebar, "
                    "or check 'Auto-suggest'.")
    else:
        with st.spinner(f"Fetching peer data ({len(peer_list)} companies)..."):
            compare_df = compare_with_peers(ticker, peer_list, target_year)

        if compare_df.empty or len(compare_df.columns) < 2:
            st.error("Failed to fetch peer data. Check the names.")
        else:
            key_metrics_for_chart = [
                "Net Margin", "ROE (Return on Equity)",
                "ROA (Return on Assets)", "Debt to Assets"
            ]
            for m in key_metrics_for_chart:
                if m in compare_df.index:
                    st.plotly_chart(plot_peer_comparison(compare_df, m),
                                    use_container_width=True)

            radar_metrics = ["Net Margin", "ROE (Return on Equity)",
                             "ROA (Return on Assets)", "Asset Turnover",
                             "Gross Margin", "Current Ratio"]
            st.plotly_chart(plot_radar(compare_df, radar_metrics),
                            use_container_width=True)

            with st.expander("📋 Peer Comparison Details"):
                display = compare_df.copy().astype(object)
                for idx in display.index:
                    for col in display.columns:
                        display.at[idx, col] = fmt_display(compare_df.loc[idx, col], idx)
                st.dataframe(display, use_container_width=True)

            st.session_state["compare_df"] = compare_df


# ===== Tab 5: Benchmark =====
with tab5:
    st.subheader("Benchmark Analysis (rule-of-thumb thresholds)")
    st.caption("⚠️ Generic thresholds - varies by industry. "
                "Use alongside peer comparison.")

    bench_df = benchmark_analysis(ratios)
    if bench_df.empty:
        st.warning("No benchmark data available")
    else:
        display_bench = bench_df.copy().astype(object)
        for idx, row in bench_df.iterrows():
            metric = row["Metric"]
            for col in ["Company", "Excellent", "Good", "Fair"]:
                v = row[col]
                if pd.isna(v) or v is None:
                    display_bench.at[idx, col] = "-"
                elif is_percentage_metric(metric):
                    display_bench.at[idx, col] = f"{v*100:.2f}%"
                else:
                    display_bench.at[idx, col] = f"{v:.2f}"
        st.dataframe(display_bench, use_container_width=True, hide_index=True)


# ===== Tab 6: Report =====
with tab6:
    st.subheader("📄 Analysis Report")

    compare_df = st.session_state.get("compare_df", pd.DataFrame())

    # Check API key
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        with st.expander("OpenAI API Key (enter to enable AI-driven analysis)"):
            api_key = st.text_input("API Key", type="password", key="report_api_key")

    cache_key = f"{ticker}_{actual_year}_{len(compare_df.columns) if not compare_df.empty else 0}"

    if "report_cache" not in st.session_state:
        st.session_state.report_cache = {}
    cached = st.session_state.report_cache.get(cache_key)

    from utils.report_builder import (
        collect_all_charts, generate_section_analyses,
        build_html_report, build_docx_report, build_pdf_report
    )

    # Phase 1: collect charts
    if cached is None or "charts" not in cached:
        with st.spinner("Collecting charts..."):
            charts = collect_all_charts(ratios, dupont, trend_df, compare_df, actual_year)
        if cached is None:
            cached = {}
        cached["charts"] = charts
        st.session_state.report_cache[cache_key] = cached

    # Phase 2: LLM-generated section analyses
    if "sections" not in cached:
        if not api_key:
            st.warning("No OpenAI API Key configured - using basic rule-based report "
                        "(no AI-driven analysis). Add `OPENAI_API_KEY` to Streamlit "
                        "Cloud Settings → Secrets.")
            fallback = generate_report(info, ratios, dupont, trend_df, compare_df, actual_year)
            cached["sections"] = {
                "overview": "See the company info table above.",
                "profitability": fallback,
                "operating": "", "solvency": "", "dupont": "",
                "trend": "", "peer": "", "diagnosis": "",
            }
            cached["is_llm"] = False
        else:
            with st.spinner("AI generating in-depth section analyses (~15-40s)..."):
                try:
                    sections = generate_section_analyses(
                        info, ratios, dupont, trend_df, compare_df,
                        actual_year, api_key,
                    )
                    cached["sections"] = sections
                    cached["is_llm"] = True
                except Exception as e:
                    st.error(f"AI analysis failed: {type(e).__name__}: {str(e)[:200]}")
                    fallback = generate_report(info, ratios, dupont, trend_df, compare_df, actual_year)
                    cached["sections"] = {
                        "overview": fallback, "profitability": "",
                        "operating": "", "solvency": "", "dupont": "",
                        "trend": "", "peer": "", "diagnosis": "",
                    }
                    cached["is_llm"] = False
        st.session_state.report_cache[cache_key] = cached

    sections = cached["sections"]
    charts = cached["charts"]

    # === Header ===
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if cached.get("is_llm"):
            st.success("✅ AI-driven analysis report ready")
        else:
            st.info("ℹ️ Rule-based report (no AI)")
    with col_b:
        if st.button("🔄 Regenerate", use_container_width=True, key="regen_report"):
            del st.session_state.report_cache[cache_key]
            st.rerun()

    st.markdown("---")

    # ========== Inline rendering: per section "data/chart -> AI analysis" ==========

    def _render_anchor_charts(anchor):
        for ch in charts:
            if ch.get("section_anchor") == anchor:
                st.plotly_chart(ch["fig"], use_container_width=True,
                                key=f"sec_{ch['id']}")

    def _render_subset_table(keys):
        rows = []
        for k in keys:
            v = ratios.get(k)
            if v is None and k not in ratios:
                continue
            rows.append({"Metric": k, "Value": fmt_display(v, k)})
        if rows:
            st.table(pd.DataFrame(rows).set_index("Metric"))

    profit_keys = ["Gross Margin", "Operating Margin", "Net Margin",
                   "ROA (Return on Assets)", "ROE (Return on Equity)"]
    operating_keys = ["Asset Turnover", "Inventory Turnover", "Receivables Turnover"]
    solvency_keys = ["Current Ratio", "Quick Ratio", "Debt to Assets",
                     "Equity Multiplier", "Interest Coverage"]

    # ===== 1. Company Overview =====
    st.markdown("## 1. Company Overview")
    info_rows = [
        ("Company Name", info.get("name", "-")),
        ("Ticker", info.get("ticker", "-")),
        ("Sector / Industry", f"{info.get('sector', '-')} / {info.get('industry', '-')}"),
        ("Country", info.get("country", "-")),
        ("Market Cap", f"${info.get('market_cap', 0)/1e9:.1f}B {info.get('currency', 'USD')}"
                  if info.get("market_cap") else "-"),
    ]
    st.table(pd.DataFrame(info_rows, columns=["Item", "Value"]).set_index("Item"))
    if sections.get("overview"):
        st.markdown(sections["overview"])

    # ===== 2. Profitability =====
    st.markdown("## 2. Profitability Analysis")
    _render_anchor_charts("profitability")
    st.markdown("**Key Profitability Metrics**")
    _render_subset_table(profit_keys)
    if sections.get("profitability"):
        st.markdown(sections["profitability"])

    # ===== 3. Operating Efficiency =====
    st.markdown("## 3. Operating Efficiency")
    st.markdown("**Efficiency Metrics**")
    _render_subset_table(operating_keys)
    if sections.get("operating"):
        st.markdown(sections["operating"])

    # ===== 4. Solvency =====
    st.markdown("## 4. Solvency and Capital Structure")
    _render_anchor_charts("solvency")
    st.markdown("**Solvency Metrics**")
    _render_subset_table(solvency_keys)
    if sections.get("solvency"):
        st.markdown(sections["solvency"])

    # ===== 5. DuPont =====
    st.markdown("## 5. DuPont Analysis: ROE Drivers")
    _render_anchor_charts("dupont")
    if sections.get("dupont"):
        st.markdown(sections["dupont"])

    # ===== 6. Trends =====
    st.markdown("## 6. Multi-Year Trends")
    _render_anchor_charts("trend")
    if sections.get("trend"):
        st.markdown(sections["trend"])

    # ===== 7. Peer Comparison =====
    st.markdown("## 7. Peer Comparison")
    if not compare_df.empty:
        st.markdown("**Peer Comparison Table**")
        compare_display = compare_df.copy().astype(object)
        for idx in compare_display.index:
            for col in compare_display.columns:
                v = compare_df.loc[idx, col]
                compare_display.loc[idx, col] = fmt_display(v, idx)
        st.dataframe(compare_display, use_container_width=True)
        _render_anchor_charts("peer")
    if sections.get("peer"):
        st.markdown(sections["peer"])

    # ===== 8. Diagnosis =====
    st.markdown("## 8. Overall Diagnosis and Investor Perspective")
    if sections.get("diagnosis"):
        st.markdown(sections["diagnosis"])

    # === Download options ===
    st.markdown("---")
    st.markdown("### 📥 Download Report")

    fmt_col1, fmt_col2, fmt_col3, fmt_col4 = st.columns(4)

    # Concatenate Markdown version
    titles_md = [
        ("1. Company Overview", "overview"),
        ("2. Profitability Analysis", "profitability"),
        ("3. Operating Efficiency", "operating"),
        ("4. Solvency and Capital Structure", "solvency"),
        ("5. DuPont Analysis: ROE Drivers", "dupont"),
        ("6. Multi-Year Trends", "trend"),
        ("7. Peer Comparison", "peer"),
        ("8. Overall Diagnosis and Investor Perspective", "diagnosis"),
    ]
    md_content = f"# {info['name']} ({ticker}) FY{actual_year} Financial Analysis Report\n\n"
    for title, key in titles_md:
        body = sections.get(key, "").strip()
        md_content += f"## {title}\n\n{body if body else '(No analysis for this section.)'}\n\n"

    with fmt_col1:
        st.download_button(
            "📄 Markdown",
            data=md_content,
            file_name=f"{ticker}_{actual_year}_FinancialAnalysis.md",
            mime="text/markdown",
            use_container_width=True,
            on_click="ignore",
        )

    with fmt_col2:
        try:
            html = build_html_report(info, actual_year, sections,
                                       charts, ratios, compare_df)
            st.download_button(
                "🌐 HTML",
                data=html.encode("utf-8"),
                file_name=f"{ticker}_{actual_year}_FinancialAnalysis.html",
                mime="text/html",
                use_container_width=True,
                on_click="ignore",
            )
        except Exception as e:
            st.button("HTML (failed)", disabled=True,
                       help=f"{type(e).__name__}: {str(e)[:100]}",
                       use_container_width=True)

    with fmt_col3:
        try:
            docx_bytes = build_docx_report(info, actual_year, sections,
                                             charts, ratios, compare_df)
            st.download_button(
                "📝 Word",
                data=docx_bytes,
                file_name=f"{ticker}_{actual_year}_FinancialAnalysis.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                on_click="ignore",
            )
        except Exception as e:
            st.button("Word (failed)", disabled=True,
                       help=f"{type(e).__name__}: {str(e)[:100]}",
                       use_container_width=True)

    with fmt_col4:
        try:
            html_for_pdf = build_html_report(info, actual_year, sections,
                                               charts, ratios, compare_df)
            pdf_bytes = build_pdf_report(html_for_pdf)
            if pdf_bytes:
                st.download_button(
                    "📕 PDF",
                    data=pdf_bytes,
                    file_name=f"{ticker}_{actual_year}_FinancialAnalysis.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    on_click="ignore",
                )
            else:
                st.button("PDF (unavailable)", disabled=True,
                           help="Server missing weasyprint - use HTML and print to PDF in browser",
                           use_container_width=True)
        except Exception as e:
            st.button("PDF (failed)", disabled=True,
                       help=f"{type(e).__name__}: {str(e)[:100]}",
                       use_container_width=True)

    st.caption("💡 Tip: HTML files can be printed to PDF via Ctrl+P in your browser.")
