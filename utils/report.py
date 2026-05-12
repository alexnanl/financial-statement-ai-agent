"""
English analysis report generation (rule-based, no LLM required).
Used as a fallback when AI section generation is not available.
"""
import pandas as pd
from typing import Dict
from utils.benchmark import GENERIC_BENCHMARKS, LOWER_IS_BETTER, evaluate_against_benchmark
from utils.ratios import is_percentage_metric


def fmt_pct(v):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v * 100:.2f}%"


def fmt_num(v, decimals=2):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def trend_direction(series: pd.Series) -> str:
    """Assess trend direction over a series of values."""
    s = series.dropna()
    if len(s) < 2:
        return "Insufficient data"
    first, last = s.iloc[0], s.iloc[-1]
    if first == 0:
        return "Flat"
    change = (last - first) / abs(first)
    if change > 0.10:
        return f"Clear upward trend (cumulative +{change*100:.1f}%)"
    elif change > 0.02:
        return f"Slight upward trend (+{change*100:.1f}%)"
    elif change < -0.10:
        return f"Clear downward trend ({change*100:.1f}%)"
    elif change < -0.02:
        return f"Slight downward trend ({change*100:.1f}%)"
    else:
        return "Roughly flat"


def generate_report(company_info: Dict, ratios: Dict, dupont: Dict,
                    trend_df: pd.DataFrame, compare_df: pd.DataFrame,
                    target_year: int) -> str:
    """Generate a full English Markdown analysis report."""

    name = company_info.get("name", "Company")
    ticker = company_info.get("ticker", "")
    sector = company_info.get("sector", "N/A")
    industry = company_info.get("industry", "N/A")
    currency = company_info.get("currency", "USD")

    md = []
    md.append(f"# {name} ({ticker}) - FY{target_year} Financial Analysis Report\n")
    md.append(f"**Sector**: {sector} / {industry}  ")
    md.append(f"**Reporting Currency**: {currency}\n")
    md.append("---\n")

    # ===== 1. Executive Summary =====
    md.append("## 1. Executive Summary\n")
    roe = ratios.get("ROE (Return on Equity)")
    nm = ratios.get("Net Margin")
    da = ratios.get("Debt to Assets")
    cr = ratios.get("Current Ratio")

    summary_points = []
    if roe is not None:
        rating = evaluate_against_benchmark("ROE (Return on Equity)", roe)
        summary_points.append(f"- **Profitability**: ROE = {fmt_pct(roe)}, rated {rating}")
    if nm is not None:
        rating = evaluate_against_benchmark("Net Margin", nm)
        summary_points.append(f"- **Net Margin**: {fmt_pct(nm)}, rated {rating}")
    if da is not None:
        rating = evaluate_against_benchmark("Debt to Assets", da)
        summary_points.append(f"- **Leverage**: Debt to Assets = {fmt_pct(da)}, rated {rating}")
    if cr is not None:
        rating = evaluate_against_benchmark("Current Ratio", cr)
        summary_points.append(f"- **Short-term Solvency**: Current Ratio = {fmt_num(cr)}, rated {rating}")

    md.append("\n".join(summary_points))
    md.append("\n")

    # ===== 2. Ratio Analysis =====
    md.append("\n## 2. Ratio Analysis\n")
    md.append("### 2.1 Profitability\n")
    md.append(f"- Gross Margin: **{fmt_pct(ratios.get('Gross Margin'))}**  ")
    md.append(f"- Operating Margin: **{fmt_pct(ratios.get('Operating Margin'))}**  ")
    md.append(f"- Net Margin: **{fmt_pct(nm)}**  ")
    md.append(f"- ROA: **{fmt_pct(ratios.get('ROA (Return on Assets)'))}**  ")
    md.append(f"- ROE: **{fmt_pct(roe)}**  \n")

    md.append("### 2.2 Efficiency\n")
    md.append(f"- Asset Turnover: **{fmt_num(ratios.get('Asset Turnover'))}x**  ")
    md.append(f"- Inventory Turnover: **{fmt_num(ratios.get('Inventory Turnover'))}x**  ")
    md.append(f"- Receivables Turnover: **{fmt_num(ratios.get('Receivables Turnover'))}x**  \n")

    md.append("### 2.3 Solvency\n")
    md.append(f"- Current Ratio: **{fmt_num(cr)}**  ")
    md.append(f"- Quick Ratio: **{fmt_num(ratios.get('Quick Ratio'))}**  ")
    md.append(f"- Debt to Assets: **{fmt_pct(da)}**  ")
    md.append(f"- Interest Coverage: **{fmt_num(ratios.get('Interest Coverage'))}x**  \n")

    # ===== 3. DuPont =====
    md.append("\n## 3. DuPont Analysis\n")
    md.append("> **ROE = Net Margin x Asset Turnover x Equity Multiplier**\n")
    md.append(f"- Net Margin (sales profitability): **{fmt_pct(dupont.get('Net Margin'))}**  ")
    md.append(f"- Asset Turnover (asset efficiency): **{fmt_num(dupont.get('Asset Turnover'))}**  ")
    md.append(f"- Equity Multiplier (financial leverage): **{fmt_num(dupont.get('Equity Multiplier'))}**  ")
    md.append(f"- **ROE (DuPont)**: {fmt_pct(dupont.get('ROE (DuPont)'))}  ")
    md.append(f"- **ROE (Direct)**: {fmt_pct(dupont.get('ROE (Direct)'))}  \n")

    nm_v = dupont.get("Net Margin") or 0
    ato_v = dupont.get("Asset Turnover") or 0
    em_v = dupont.get("Equity Multiplier") or 0
    factors = [("Net Margin", nm_v, 0.10), ("Asset Turnover", ato_v, 0.6),
               ("Equity Multiplier", em_v, 2.0)]
    drivers = []
    for fname, val, threshold in factors:
        if val > threshold * 1.2:
            drivers.append(f"{fname} high ({fmt_num(val) if fname != 'Net Margin' else fmt_pct(val)})")

    if drivers:
        md.append(f"**Primary ROE drivers**: {', '.join(drivers)}.")
    md.append("")

    # ===== 4. Trend Analysis =====
    md.append("\n## 4. Trend Analysis\n")
    if trend_df is not None and not trend_df.empty:
        years_str = " -> ".join(str(c) for c in trend_df.columns)
        md.append(f"Period covered: {years_str}\n")

        key_metrics = ["Net Margin", "ROE (Return on Equity)", "ROA (Return on Assets)",
                       "Debt to Assets", "Current Ratio", "Asset Turnover"]
        for m in key_metrics:
            if m in trend_df.index:
                direction = trend_direction(trend_df.loc[m])
                md.append(f"- **{m}**: {direction}")
        md.append("")
    else:
        md.append("*Insufficient trend data*\n")

    # ===== 5. Peer Comparison =====
    md.append("\n## 5. Peer Comparison\n")
    if compare_df is not None and not compare_df.empty and len(compare_df.columns) > 1:
        target_col = compare_df.columns[0]
        peers = compare_df.columns[1:]
        md.append(f"Peers: {', '.join(peers)}\n")

        key_metrics = ["Net Margin", "ROE (Return on Equity)", "ROA (Return on Assets)",
                       "Asset Turnover", "Debt to Assets", "Current Ratio"]
        for m in key_metrics:
            if m not in compare_df.index:
                continue
            target_v = compare_df.loc[m, target_col]
            peer_vals = compare_df.loc[m].drop(target_col).dropna()
            if pd.isna(target_v) or len(peer_vals) == 0:
                continue
            peer_mean = peer_vals.mean()
            is_pct = is_percentage_metric(m)

            lower_better = m in LOWER_IS_BETTER
            if lower_better:
                comparison = "better than" if target_v < peer_mean else "above"
            else:
                comparison = "better than" if target_v > peer_mean else "below"

            t_str = fmt_pct(target_v) if is_pct else fmt_num(target_v)
            p_str = fmt_pct(peer_mean) if is_pct else fmt_num(peer_mean)
            md.append(f"- **{m}**: Company {t_str}, Peer Avg {p_str} -> {comparison} peers")
        md.append("")
    else:
        md.append("*No peer data provided or insufficient peer data*\n")

    # ===== 6. Benchmark Rating Summary =====
    md.append("\n## 6. Benchmark Rating Summary\n")
    rating_table = []
    for ratio_name in GENERIC_BENCHMARKS.keys():
        if ratio_name in ratios:
            v = ratios[ratio_name]
            rating = evaluate_against_benchmark(ratio_name, v)
            is_pct = is_percentage_metric(ratio_name)
            display_v = fmt_pct(v) if is_pct else fmt_num(v)
            rating_table.append(f"| {ratio_name} | {display_v} | {rating} |")

    if rating_table:
        md.append("| Metric | Value | Rating |")
        md.append("|--------|-------|--------|")
        md.extend(rating_table)
    md.append("")

    # ===== 7. Overall Assessment =====
    md.append("\n## 7. Overall Assessment\n")
    pros, cons = [], []

    if roe is not None:
        if roe > 0.15:
            pros.append(f"ROE of {fmt_pct(roe)} indicates strong shareholder returns")
        elif roe < 0.05:
            cons.append(f"ROE of just {fmt_pct(roe)} is below average")

    if nm is not None:
        if nm > 0.15:
            pros.append(f"Net Margin of {fmt_pct(nm)} is high")
        elif nm < 0.03:
            cons.append(f"Net Margin of only {fmt_pct(nm)} - profitability is squeezed")

    if da is not None:
        if da > 0.70:
            cons.append(f"Debt to Assets at {fmt_pct(da)} - elevated leverage; watch solvency")
        elif da < 0.30:
            pros.append(f"Debt to Assets at {fmt_pct(da)} - conservative capital structure")

    if cr is not None:
        if cr < 1.0:
            cons.append(f"Current Ratio of only {fmt_num(cr)} - short-term liquidity is tight")

    ato = ratios.get("Asset Turnover")
    if ato is not None and ato > 1.0:
        pros.append(f"Asset Turnover of {fmt_num(ato)}x - efficient use of assets")

    if pros:
        md.append("**Strengths**:")
        for p in pros:
            md.append(f"- {p}")
        md.append("")
    if cons:
        md.append("**Risks / Weaknesses**:")
        for c in cons:
            md.append(f"- {c}")
        md.append("")
    if not pros and not cons:
        md.append("*Metrics indicate broadly neutral performance.*\n")

    md.append("\n---\n")
    md.append("*This report is auto-generated from public financial data and is for "
              "informational purposes only. It does not constitute investment advice.*")

    return "\n".join(md)
