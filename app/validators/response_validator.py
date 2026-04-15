"""
ResponseValidator — checks AI responses for accuracy and completeness.
Uses heuristic checks only (no LLM call) to keep things fast and free.
"""

from app.models.schemas import ConfidenceLevel

_WEB_TRIGGERS = [
    "news", "today", "weather", "president", "prime minister", "ceo", "who is",
    "current", "latest", "now", "price", "stock", "score", "election",
    "2024", "2025", "2026", "this week", "yesterday", "just",
]


class ResponseValidator:
    """Validates AI responses using heuristics — no LLM call needed."""

    async def validate(self, question: str, response: str) -> dict:
        """
        Returns:
          {
            confidence: "high"|"medium"|"low"|"unverified",
            issues: [...],
            should_search_web: bool,
            verdict: "..."
          }
        """
        stale_phrases = [
            "as of my last update", "as of my knowledge", "i don't have access",
            "i recommend checking", "i cannot browse", "my training data",
        ]
        response_lower = response.lower()
        has_stale = any(p in response_lower for p in stale_phrases)
        needs_web = any(kw in question.lower() for kw in _WEB_TRIGGERS)

        if has_stale and needs_web:
            return {
                "confidence": ConfidenceLevel.UNVERIFIED,
                "issues": ["Response uses training data for a real-time question"],
                "should_search_web": True,
                "verdict": "This answer may be outdated. Web search was not used.",
            }

        # Basic length / quality heuristics
        issues = []
        if len(response.strip()) < 20:
            issues.append("Response is very short")
        if has_stale:
            issues.append("Response contains stale-knowledge phrases")

        if issues:
            return {
                "confidence": ConfidenceLevel.MEDIUM,
                "issues": issues,
                "should_search_web": needs_web,
                "verdict": "Response may need improvement.",
            }

        return {
            "confidence": ConfidenceLevel.HIGH,
            "issues": [],
            "should_search_web": False,
            "verdict": "Response looks good.",
        }
