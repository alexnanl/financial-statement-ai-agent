"""
AI Chat mode page - embedded into streamlit_app.py
"""
import os
import streamlit as st
import pandas as pd
from utils.ai_agent import chat_with_tools
from utils.charts import (
    plot_trend, plot_peer_comparison, plot_dupont_waterfall,
)
from utils.data_fetcher import fetch_financials
from utils.ratios import compute_multi_year_ratios
from utils.benchmark import compare_with_peers


# Anti-abuse: max messages per session
MAX_MESSAGES_PER_SESSION = 60
MAX_INPUT_LENGTH = 1000


def _render_report_download_buttons(ev: dict, key_prefix: str, api_key: str = None):
    """
    Render 4 download format buttons (Markdown / HTML / Word / PDF).
    Uses the same build_*_report functions as the classic mode for consistency.

    ev = {"type": "report", "filename", "markdown", "ticker", "year", "full_data"}
    full_data = {"info", "ratios", "dupont", "trend_df", "compare_df", "actual_year", "ticker"}
    """
    fname_md = ev.get("filename") or "report.md"
    md_content = ev.get("markdown", "")
    full_data = ev.get("full_data")
    ticker = ev.get("ticker", "report")
    year = ev.get("year", 2024)
    base_name = f"{ticker}_{year}_FinancialAnalysis"

    # No full_data (old history msg) - fallback to simple Markdown only
    if not full_data:
        st.download_button(
            f"Download Markdown: {fname_md}",
            data=md_content if md_content else "(Empty report - please regenerate)",
            file_name=fname_md,
            mime="text/markdown",
            key=f"{key_prefix}_md_only",
            disabled=not bool(md_content),
            on_click="ignore",
        )
        if not md_content:
            st.warning("Report content is empty. Please send a new request "
                       "(e.g. 'give me a full report on Tesla').")
        return

    # Full data available - generate all 4 formats
    info = full_data["info"]
    ratios = full_data["ratios"]
    dupont = full_data["dupont"]
    trend_df = full_data["trend_df"]
    compare_df = full_data["compare_df"]
    actual_year = full_data["actual_year"]

    # Cache sections + charts so we don't regenerate on every render
    cache_key = f"chat_report_{ticker}_{actual_year}"
    if "report_cache" not in st.session_state:
        st.session_state.report_cache = {}

    cached = st.session_state.report_cache.get(cache_key)

    from utils.report_builder import (
        collect_all_charts, generate_section_analyses,
        build_html_report, build_docx_report, build_pdf_report
    )

    # Phase 1: collect charts
    if cached is None or "charts" not in cached:
        with st.spinner("Preparing charts..."):
            charts = collect_all_charts(ratios, dupont, trend_df, compare_df, actual_year)
        if cached is None:
            cached = {}
        cached["charts"] = charts
        st.session_state.report_cache[cache_key] = cached

    # Phase 2: LLM section-by-section analysis
    if "sections" not in cached:
        if not api_key:
            # Fallback to rule-based report
            cached["sections"] = {
                "overview": md_content or "See company info above.",
                "profitability": "", "operating": "", "solvency": "", "dupont": "",
                "trend": "", "peer": "", "diagnosis": "",
            }
            cached["is_llm"] = False
        else:
            with st.spinner("AI generating in-depth section analyses (~15-40 seconds)..."):
                try:
                    sections = generate_section_analyses(
                        info, ratios, dupont, trend_df, compare_df,
                        actual_year, api_key,
                    )
                    cached["sections"] = sections
                    cached["is_llm"] = True
                except Exception as e:
                    st.warning(f"AI section generation failed ({type(e).__name__}); "
                               "falling back to rule-based report")
                    cached["sections"] = {
                        "overview": md_content or "See company info above.",
                        "profitability": "", "operating": "", "solvency": "", "dupont": "",
                        "trend": "", "peer": "", "diagnosis": "",
                    }
                    cached["is_llm"] = False
        st.session_state.report_cache[cache_key] = cached

    sections = cached["sections"]
    charts = cached["charts"]

    if cached.get("is_llm"):
        st.success("AI-driven analysis report ready - choose a download format:")
    else:
        st.info("Rule-based report (no API key or AI call failed)")

    # ===== 4 download buttons =====
    fmt_col1, fmt_col2, fmt_col3, fmt_col4 = st.columns(4)

    with fmt_col1:
        # Markdown: concatenate all sections
        full_md_parts = [f"# {info.get('name', ticker)} ({ticker}) - FY{actual_year} Financial Analysis\n"]
        section_titles = [
            ("overview", "Company Overview"),
            ("profitability", "Profitability"),
            ("operating", "Operating Efficiency"),
            ("solvency", "Solvency"),
            ("dupont", "DuPont Analysis"),
            ("trend", "Multi-Year Trends"),
            ("peer", "Peer Comparison"),
            ("diagnosis", "Overall Diagnosis"),
        ]
        for skey, stitle in section_titles:
            content = sections.get(skey, "")
            if content:
                full_md_parts.append(f"\n## {stitle}\n\n{content}\n")
        full_md = "\n".join(full_md_parts)

        st.download_button(
            "Markdown",
            data=full_md,
            file_name=f"{base_name}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{key_prefix}_md",
            on_click="ignore",
        )

    with fmt_col2:
        try:
            html = build_html_report(info, actual_year, sections,
                                       charts, ratios, compare_df)
            st.download_button(
                "HTML",
                data=html.encode("utf-8"),
                file_name=f"{base_name}.html",
                mime="text/html",
                use_container_width=True,
                key=f"{key_prefix}_html",
                on_click="ignore",
            )
        except Exception as e:
            st.button("HTML (failed)", disabled=True,
                       help=f"{type(e).__name__}: {str(e)[:100]}",
                       use_container_width=True,
                       key=f"{key_prefix}_html_fail")

    with fmt_col3:
        try:
            docx_bytes = build_docx_report(info, actual_year, sections,
                                             charts, ratios, compare_df)
            st.download_button(
                "Word",
                data=docx_bytes,
                file_name=f"{base_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key=f"{key_prefix}_docx",
                on_click="ignore",
            )
        except Exception as e:
            st.button("Word (failed)", disabled=True,
                       help=f"{type(e).__name__}: {str(e)[:100]}",
                       use_container_width=True,
                       key=f"{key_prefix}_docx_fail")

    with fmt_col4:
        try:
            html_for_pdf = build_html_report(info, actual_year, sections,
                                               charts, ratios, compare_df)
            pdf_bytes = build_pdf_report(html_for_pdf)
            if pdf_bytes:
                st.download_button(
                    "PDF",
                    data=pdf_bytes,
                    file_name=f"{base_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"{key_prefix}_pdf",
                    on_click="ignore",
                )
            else:
                st.button("PDF (unavailable)", disabled=True,
                           help="Server missing weasyprint - use HTML and print to PDF in browser",
                           use_container_width=True,
                           key=f"{key_prefix}_pdf_unavail")
        except Exception as e:
            st.button("PDF (failed)", disabled=True,
                       help=f"{type(e).__name__}: {str(e)[:100]}",
                       use_container_width=True,
                       key=f"{key_prefix}_pdf_fail")


def render_tool_visual(tool_name: str, args: dict, result: dict):
    """Render visualizations for tool call results."""
    if "error" in result:
        return

    year = args.get("year", 2024)

    try:
        if tool_name == "dupont_analysis":
            ticker = result.get("ticker")
            if not ticker:
                return
            dp = {
                "Net Margin": result.get("Net Margin"),
                "Asset Turnover": result.get("Asset Turnover"),
                "Equity Multiplier": result.get("Equity Multiplier"),
                "ROE (DuPont)": result.get("ROE (DuPont)"),
            }
            st.plotly_chart(plot_dupont_waterfall(dp, result.get("year", year)),
                            use_container_width=True, key=f"dupont_{ticker}_{year}")

        elif tool_name == "trend_analysis":
            ticker = result.get("ticker")
            if not ticker:
                return
            num_years = args.get("num_years", 5)
            target_year = args.get("target_year", 2024)
            fin = fetch_financials(ticker)
            trend_df = compute_multi_year_ratios(fin, target_year, num_years=num_years)
            if not trend_df.empty:
                st.plotly_chart(
                    plot_trend(trend_df,
                                ["Gross Margin", "Operating Margin", "Net Margin"],
                                title="Profitability Trends"),
                    use_container_width=True,
                    key=f"trend_prof_{ticker}_{num_years}"
                )
                st.plotly_chart(
                    plot_trend(trend_df,
                                ["ROE (Return on Equity)", "ROA (Return on Assets)"],
                                title="Return Trends"),
                    use_container_width=True,
                    key=f"trend_ret_{ticker}_{num_years}"
                )

        elif tool_name == "peer_comparison":
            ticker = result.get("ticker")
            peers = result.get("peers_used", [])
            if ticker and peers:
                compare_df = compare_with_peers(ticker, peers, year)
                if not compare_df.empty:
                    for m in ["Net Margin", "ROE (Return on Equity)"]:
                        if m in compare_df.index:
                            st.plotly_chart(
                                plot_peer_comparison(compare_df, m),
                                use_container_width=True,
                                key=f"peer_{m}_{ticker}_{year}"
                            )

        elif tool_name == "compute_ratios":
            ratios_data = result.get("ratios", {})
            if ratios_data:
                key_metrics_pct = [
                    ("ROE", "ROE (Return on Equity)"),
                    ("ROA", "ROA (Return on Assets)"),
                    ("Net Margin", "Net Margin"),
                    ("Gross Margin", "Gross Margin"),
                ]
                cols = st.columns(4)
                for col, (label, key) in zip(cols, key_metrics_pct):
                    val = ratios_data.get(key, {}).get("value")
                    if val is not None:
                        col.metric(label, f"{val*100:.2f}%")
    except Exception as e:
        st.caption(f"_(Chart rendering skipped: {e})_")


def render_chat_page():
    """Render the AI chat page."""

    # ===== Get API key =====
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        with st.expander("OpenAI API Key (developer fallback)"):
            api_key = st.text_input(
                "OpenAI API Key", type="password",
                help="If the deployed key is missing, enter your own. Session-only."
            )

    if not api_key:
        st.error("OpenAI API Key is not configured. Add `OPENAI_API_KEY` to Streamlit "
                 "Secrets, or enter one above to enable AI features.")
        st.markdown("""
        **Admin setup**:
        1. Go to Settings -> Secrets in your Streamlit Cloud app
        2. Add a line: `OPENAI_API_KEY = "sk-..."`
        3. Save - the app will restart automatically
        """)
        return

    # ===== Initialize chat history =====
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "generated_reports" not in st.session_state:
        st.session_state.generated_reports = {}

    # ===== Toolbar =====
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("""
        **Try these examples**:
        - "How is Apple doing in 2024?" - quick assessment
        - "Run a DuPont analysis on Microsoft" - deep dive
        - "Compare Tesla with Ford" - peer comparison
        - "Generate a full report for NVIDIA" - downloadable report
        """)
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.generated_reports = {}
            st.rerun()

    st.markdown("---")

    # ===== Render history =====
    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                events = msg.get("events", [])
                for ev in events:
                    if ev["type"] == "tool_call":
                        with st.expander(f"Tool call: `{ev['tool']}`", expanded=False):
                            st.json(ev.get("args", {}))
                    elif ev["type"] == "tool_visual":
                        render_tool_visual(ev["tool"], ev["args"], ev["result"])
                    elif ev["type"] == "report":
                        _render_report_download_buttons(
                            ev,
                            key_prefix=f"hist_{i}",
                            api_key=api_key,
                        )

                if msg.get("content"):
                    st.markdown(msg["content"])

    # ===== Input box =====
    if len(st.session_state.chat_messages) >= MAX_MESSAGES_PER_SESSION:
        st.warning(f"Session limit reached ({MAX_MESSAGES_PER_SESSION} messages). "
                   "Clear chat to continue.")
        return

    user_input = st.chat_input("Ask anything about US-listed companies' financials...")

    if not user_input:
        return

    if len(user_input) > MAX_INPUT_LENGTH:
        st.error(f"Input too long (>{MAX_INPUT_LENGTH} chars). Please shorten.")
        return

    # Show user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build LLM history (only required fields, no events)
    llm_messages = []
    for m in st.session_state.chat_messages:
        if m["role"] == "user":
            llm_messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("content"):
            llm_messages.append({"role": "assistant", "content": m["content"]})

    # ===== Stream Agent response =====
    with st.chat_message("assistant"):
        events_log = []
        final_content = ""
        status_placeholder = st.empty()

        try:
            for event in chat_with_tools(llm_messages, api_key=api_key):
                if event["type"] == "tool_call":
                    status_placeholder.info(f"Calling `{event['tool']}`...")
                    events_log.append(event)
                    with st.expander(f"Tool call: `{event['tool']}`", expanded=False):
                        st.json(event.get("args", {}))

                elif event["type"] == "tool_result":
                    if events_log and events_log[-1].get("type") == "tool_call":
                        call_event = events_log[-1]
                        visual_event = {
                            "type": "tool_visual",
                            "tool": call_event["tool"],
                            "args": call_event["args"],
                            "result": event["result"],
                        }
                        events_log.append(visual_event)
                        render_tool_visual(call_event["tool"], call_event["args"],
                                            event["result"])

                elif event["type"] == "report":
                    events_log.append(event)
                    _render_report_download_buttons(
                        event,
                        key_prefix=f"new_{len(st.session_state.chat_messages)}",
                        api_key=api_key,
                    )

                elif event["type"] == "assistant":
                    final_content = event["content"]
                    status_placeholder.empty()
                    st.markdown(final_content)

                elif event["type"] == "error":
                    status_placeholder.empty()
                    st.error(event["message"])
                    final_content = f"Error: {event['message']}"

        except Exception as e:
            status_placeholder.empty()
            st.error(f"Chat error: {type(e).__name__}: {str(e)[:200]}")
            final_content = f"Error: {str(e)[:100]}"

    # Save to history
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": final_content,
        "events": events_log,
    })
