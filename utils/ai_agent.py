"""
AI Agent - OpenAI Function Calling main loop.
"""
import json
import os
from typing import List, Dict, Generator, Optional
from utils.tools import TOOL_SCHEMAS, execute_tool


SYSTEM_PROMPT = """You are a professional financial analysis AI Agent. You can call tools to fetch financial data for US-listed companies and perform various analyses.

# Your capabilities
- Fetch data and run various analyses through 6 specialized tools
- Reply in clear, professional, accessible English
- Provide **insights** in your analysis, not just numbers

# Working principles
1. **Understand intent first**: When a user mentions a company, call `fetch_company` first to confirm it exists, then decide next steps
2. **Call tools as needed**: Don't call a bunch of tools at once - call the specific one matching the user's question
3. **Reuse known info**: Data already retrieved in the same conversation should be reused, not re-fetched
4. **Data first, interpretation follows**: After showing key numbers, provide **insightful** interpretation (why this number matters, what's good/bad about it, how it compares to benchmarks)
5. **Acknowledge uncertainty**: When data is missing or anomalous, say so and explain possible reasons (yfinance limits, company-specific situations, etc.)

# Data validity checks (CRITICAL)
- If the tool returns key ratios (ROE/Net Margin/Asset Turnover etc.) **all 0 or None** -> data wasn't fetched for that year; **do not** show charts/analysis for that year
- Explain in text: "Data for YYYY was unavailable - possibly due to yfinance limits (typically only covers ~4 years)"
- Don't call chart visualization on "all-zero data" - it will confuse the user

# Calculation conventions (IMPORTANT)
- The DuPont analysis tool returns a "calc_method" field:
  - "average": uses average equity (standard convention) -> trust directly
  - "ending": uses ending equity (fallback when previous year missing) -> **must mention this to the user**
- When ROE is abnormally high (e.g. Apple > 150%), proactively explain why: heavy stock buybacks deflate ending equity, not actually anomalous profitability
- When mixing both calc methods across years for the same company, **clearly tell the user these numbers are not directly comparable**

# When to call generate_full_report
- When user explicitly says "full report", "comprehensive analysis", "download report", "give me detailed analysis"
- Don't generate a full report for a simple question

# Multi-year analysis best practices
- When user asks about "past N years":
  - Use `trend_analysis` once to get multi-year data
  - Don't call dupont_analysis in a loop
  - `trend_analysis` will tell you the actual available year range

# Response style
- Concise and professional, avoid filler
- Use **bold** to highlight key numbers
- Show ratios as percentages (e.g. ROE 25.30%), turnover as multipliers (e.g. 1.05x)
- Large amounts: use B (billion) / T (trillion) suffixes
- Use symbols sparingly (-, +) to flag key findings

# Notes
- You see structured JSON tool results - **do not** paste raw JSON to the user; digest it and express in natural language
- yfinance typically provides only 4 years of annual reports - not a bug, just inform the user
- Users are global English speakers - keep tone professional and accessible
- This app only supports US-listed stocks (NYSE / NASDAQ). If a user asks about a foreign company, suggest searching for its US ADR if one exists."""


def get_openai_client(api_key: Optional[str] = None):
    """Get an OpenAI client; lazy import to avoid startup overhead."""
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OpenAI API Key")
    return OpenAI(api_key=key)


def chat_with_tools(messages: List[Dict], api_key: Optional[str] = None,
                     model: str = "gpt-4o-mini",
                     max_iterations: int = 6) -> Generator[Dict, None, None]:
    """
    Multi-turn tool-calling loop, streaming events.
    Event types yielded:
      {"type": "tool_call",   "tool": "...", "args": {...}}
      {"type": "tool_result", "tool": "...", "result": {...}}
      {"type": "assistant",   "content": "..."}                  # final AI text
      {"type": "report",      "filename": "...", "markdown": "..."}  # report payload
      {"type": "error",       "message": "..."}
    """
    client = get_openai_client(api_key)

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=full_messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as e:
            yield {"type": "error",
                   "message": f"OpenAI API call failed: {type(e).__name__}: {str(e)[:200]}"}
            return

        msg = resp.choices[0].message

        # Append assistant message to history
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        full_messages.append(assistant_entry)

        # No tool calls -> this is the final answer
        if not msg.tool_calls:
            yield {"type": "assistant", "content": msg.content or ""}
            return

        # Execute each tool call
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            yield {"type": "tool_call", "tool": tool_name, "args": args}

            result = execute_tool(tool_name, args)
            yield {"type": "tool_result", "tool": tool_name, "result": result}

            # If a report was generated, emit a special event for the UI
            if tool_name == "generate_full_report" and "report_markdown" in result:
                yield {
                    "type": "report",
                    "filename": result.get("filename", "report.md"),
                    "markdown": result["report_markdown"],
                    "ticker": result.get("ticker"),
                    "year": result.get("year"),
                    "full_data": result.get("_full_data"),
                }
                # Don't include full markdown or _full_data in what the LLM sees
                # (too big + has non-JSON-serializable objects like DataFrames)
                result_for_llm = {
                    "ticker": result.get("ticker"),
                    "year": result.get("year"),
                    "status": "Report generated and shown to user (with download buttons)",
                    "report_length_chars": len(result["report_markdown"]),
                }
            else:
                result_for_llm = result

            # Append tool result to history
            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_for_llm, ensure_ascii=False),
            })

    # Exceeded max iterations
    yield {"type": "error",
           "message": f"Tool iteration exceeded {max_iterations} - possibly stuck in a loop"}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    if not text:
        return 0
    return len(text) // 4
