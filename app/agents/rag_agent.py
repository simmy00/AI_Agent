from __future__ import annotations
import json
import re
import uuid
from typing import AsyncIterator, Callable, Awaitable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.prompts.templates import RAG_CONTEXT_TEMPLATE, SYSTEM_PROMPT
from app.rag.hybrid_retriever import HybridRetriever
from app.tools.tools import CalculatorTool, WebSearchTool, _search_sync, _safe_eval

import ast as _ast

settings = get_settings()

# ── Patterns the model might use to indicate tool intent ─────────────────────
_SEARCH_PATTERNS = re.compile(
    r"(what.{0,10}news|who is .{2,30}(president|pm|ceo|minister)|"
    r"weather|stock.?price|current|latest|today.{0,10}(news|score|price)|"
    r"yesterday|this week|right now|happening|recent|2024|2025|2026)",
    re.IGNORECASE,
)

_CALC_PATTERNS = re.compile(
    r"(calculate|compute|what is \d|how much is|how much.{0,20}cost|convert|multiply|"
    r"total|rate.{0,15}\d|per\s*(square|sq|unit|yard|meter|foot|acre)|"
    r"\d+\s*[\+\-\*\/\%\^]\s*\d|\d+%\s*(of|on|tip)|sqrt|log\(|sin\(|cos\()",
    re.IGNORECASE,
)

# Token artifacts from Llama that should never reach the user
_JUNK_TOKENS = re.compile(
    r"<\|[^>]+\|>|<\|end_header_id\|>|<\|start_header_id\|>|"
    r"<function=\w+>|</function>|<\|eot_id\|>|<\|python_tag\|>"
)


def _clean(text: str) -> str:
    """Strip leaked Llama special tokens from output, but preserve normal whitespace."""
    text = _JUNK_TOKENS.sub("", text)
    # Remove tool name leaks like "web_search\n\n" only at the start
    text = re.sub(r"^(web_search|calculator)\s*", "", text, flags=re.IGNORECASE)
    return text


class RAGAgent:
    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

        self._llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.4,
            streaming=True,
        )

        self._sessions: dict[str, list] = {}

    # ── Public ───────────────────────────────────────────────────────────────

    async def stream(
        self,
        session_id: str,
        user_message: str,
        on_searching: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        if user_message.strip() == "__clear__":
            self._sessions.pop(session_id, None)
            return

        history = self._sessions.setdefault(session_id, [])
        human = self._build_human_message(user_message)
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [human]

        # ── Step 1: Check if we should call a tool BEFORE the LLM ─────────
        # Build context from recent history for follow-up queries like "multiply it"
        recent_context = self._get_recent_context(history)
        tool_result = await self._maybe_call_tool(user_message, recent_context, on_searching)

        if tool_result:
            # Inject tool result into context and let LLM format the answer
            tool_context = f"Here are the results from a live search/calculation:\n\n{tool_result}\n\nUse this information to answer the user's question. Include source URLs when available."
            messages.append(HumanMessage(content=tool_context))

        # ── Step 2: Stream the LLM response ───────────────────────────────
        text_chunks: list[str] = []

        async for chunk in self._llm.astream(messages):
            if chunk.content:
                cleaned = _clean(chunk.content)
                if cleaned:
                    text_chunks.append(cleaned)
                    yield cleaned

        full_response = "".join(text_chunks)
        self._save(session_id, history, user_message, full_response)

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── Private ───────────────────────────────────────────────────────────

    def _get_recent_context(self, history: list, max_messages: int = 6) -> str:
        """Get text from recent messages for context-aware tool routing."""
        recent = history[-max_messages:] if len(history) > max_messages else history
        parts = []
        for msg in recent:
            if hasattr(msg, "content") and msg.content:
                parts.append(msg.content)
        return " ".join(parts)

    async def _maybe_call_tool(
        self, query: str, recent_context: str, on_searching: Callable | None = None
    ) -> str | None:
        """Decide if a tool should be called based on query + conversation context."""
        q = query.strip()
        # Combine current query + recent history for context awareness
        full_context = f"{recent_context} {q}"

        # Check for calculator intent — look at both current message AND history
        is_calc_query = _CALC_PATTERNS.search(q)
        is_calc_followup = bool(re.search(
            r"(calculate|multiply|compute|do it|show me|solve|what is the (answer|result|total|cost))",
            q, re.IGNORECASE
        )) and re.search(r"\d", recent_context or "")

        if is_calc_query or is_calc_followup:
            # Try extracting math from current message first, then from full context
            expr = await self._extract_math(q)
            if not expr:
                expr = await self._extract_math(full_context)
            if expr:
                try:
                    result = _safe_eval(_ast.parse(expr, mode="eval"))
                    return f"Calculation: {expr} = {result:,.10g}"
                except Exception:
                    pass

        # Check for web search intent
        if _SEARCH_PATTERNS.search(q):
            if on_searching:
                await on_searching(q)
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None, _search_sync, q
            )
            if result and "not configured" not in result.lower():
                return result

        return None

    async def _extract_math(self, query: str) -> str | None:
        """Try to extract a math expression from a natural language query."""
        # Simple patterns: "calculate 15% of 2400" → "2400 * 0.15"
        m = re.search(r"(\d+)\s*%\s*(?:of|on|tip)\s*[₹$€£]?\s*([\d,]+)", query, re.IGNORECASE)
        if m:
            pct, amount = float(m.group(1)), float(m.group(2).replace(",", ""))
            return f"{amount} * {pct / 100}"

        # Direct expression: "calculate 100 * 1.15"
        m = re.search(r"([\d\.\s\+\-\*\/\%\(\)\^]+)", query)
        if m:
            expr = m.group(1).strip().replace("^", "**")
            if any(c in expr for c in "+-*/"):
                return expr

        # Two numbers in the text → likely wants multiplication (e.g., "118 yards at 73000 per yard")
        numbers = re.findall(r"[\d,]+\.?\d*", query)
        numbers = [n.replace(",", "") for n in numbers if float(n.replace(",", "")) > 0]
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            return f"{a} * {b}"

        return None

    def _build_human_message(self, query: str) -> HumanMessage:
        """Inject RAG context if documents exist and retrieval finds hits."""
        if not self._retriever.has_documents:
            return HumanMessage(content=query)
        hits = self._retriever.retrieve(query)
        if not hits:
            return HumanMessage(content=query)
        context = "\n\n---\n\n".join(
            f"[{h['metadata'].get('filename', 'doc')}, "
            f"page {h['metadata'].get('page_number', '?')}]\n{h['content']}"
            for h in hits
        )
        return HumanMessage(
            content=RAG_CONTEXT_TEMPLATE.format(context=context, question=query)
        )

    def _save(self, session_id: str, history: list, user_msg: str, ai_content: str) -> None:
        self._sessions[session_id] = (
            history
            + [HumanMessage(content=user_msg)]
            + [AIMessage(content=ai_content)]
        )