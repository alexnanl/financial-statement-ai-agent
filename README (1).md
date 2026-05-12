# Financial Analysis AI Agent

A Streamlit-based AI agent for analyzing US-listed companies. Generates financial ratios, DuPont decomposition, multi-year trends, peer comparison, and downloadable reports (Markdown / HTML / Word / PDF).

## Features

- **Ratio Analysis** - 15+ metrics across profitability, efficiency, solvency, and cash flow
- **DuPont Analysis** - ROE decomposition (Net Margin x Asset Turnover x Equity Multiplier)
- **Trend Analysis** - Multi-year visualization of key metrics
- **Peer Comparison** - Auto-matched peers by industry + market cap
- **Benchmark Analysis** - Excellent / Good / Fair / Weak ratings
- **AI Chat Mode** - Conversational interface powered by GPT function calling
- **Multi-format Reports** - Markdown, HTML, Word (.docx), PDF

## Data Sources

US stocks only (NYSE / NASDAQ). The app uses two free data sources with smart routing:

```
US Stocks
 ├─ Primary:  FMP (Financial Modeling Prep) - 5 yrs of data, fewer rate limits
 └─ Fallback: yfinance (Yahoo Finance) - 4 yrs, free, no key needed
```

**FMP is preferred** because it's faster, has broader coverage, and avoids Yahoo's rate limits. When FMP returns a partial result (e.g. cashflow statement empty), the missing pieces are merge-filled from yfinance automatically.

If `FMP_API_KEY` is not set, the app falls back entirely to yfinance.

## Setup

### Streamlit Cloud

1. Push this repository to GitHub
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io)
3. In **Settings -> Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."     # Required for AI features
   FMP_API_KEY = "..."           # Optional - improves data reliability
   ```
4. The app will auto-deploy

### Local Development

```bash
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-..."
export FMP_API_KEY="..."        # optional

streamlit run streamlit_app.py
```

## Free API Keys

- **OpenAI**: [platform.openai.com](https://platform.openai.com) - needed for AI features
- **FMP** (free tier): [site.financialmodelingprep.com](https://site.financialmodelingprep.com/developer/docs) - 250 calls/day free, improves data coverage

## Project Structure

```
finance_agent/
├── streamlit_app.py          # Main entry point (English UI inlined)
├── requirements.txt
├── packages.txt              # System dependencies (for weasyprint)
└── utils/
    ├── __init__.py
    ├── data_provider.py      # Smart routing: FMP (primary) -> yfinance (fallback)
    ├── data_fetcher.py       # Company info, financials, peer search
    ├── ratios.py             # All financial ratio calculations
    ├── benchmark.py          # Peer comparison + benchmark rating
    ├── charts.py             # Plotly visualizations
    ├── report.py             # Rule-based Markdown report
    ├── report_builder.py     # LLM-driven multi-format reports
    ├── tools.py              # Function-calling tools for AI agent
    ├── ai_agent.py           # OpenAI tool-calling main loop
    └── chat_page.py          # AI chat mode UI
```

## Usage Examples

**Classic Mode**: Enter "Apple" or "AAPL" in the sidebar, click Run Analysis. Browse tabs for ratios, DuPont, trends, peers, benchmark, and report.

**AI Chat Mode**:
- "How is Apple doing in 2024?"
- "Run a DuPont analysis on Microsoft"
- "Compare Tesla with Ford"
- "Generate a full report for NVIDIA"

## Disclaimer

This tool is for informational and academic purposes only and does not constitute investment advice. Financial data comes from public sources (Yahoo Finance, FMP) and may contain errors or delays.

## License

MIT
