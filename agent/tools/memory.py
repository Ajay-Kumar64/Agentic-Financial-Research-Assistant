# File: agent/tools/memory.py
import re
from typing import List, Dict, Any, Optional
from agent.llm_provider import call_llm_sync


class ConversationMemoryTool:
    """
    Manages conversation state with:
    - Sliding window (last 5 turns)
    - LLM-based coreference resolution
    - History summarization for long contexts
    - Regex fallback
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def resolve_query(
        self,
        current_query: str,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        if not conversation_history or len(conversation_history) == 0:
            return current_query

        # COMPRESS HISTORY for LLM context window
        history_text = self._summarize_history(conversation_history)

        prompt = f"""Given this conversation history:

{history_text}

The user now asks: "{current_query}"

Rewrite the user's query to be fully self-contained, replacing all pronouns (they, it, that, those, this) and references with their actual referents.
If the query is already self-contained, return it unchanged.
Return ONLY the rewritten query, nothing else."""

        try:
            resolved, _ = call_llm_sync(
                prompt=prompt,
                system_instruction="You are a query rewriter. Return ONLY the rewritten query string. No explanations, no markdown.",
                temperature=0.0
            )
            resolved = resolved.strip().strip('"').strip("'")
            if resolved and len(resolved) > 5:
                return resolved
        except Exception as e:
            print(f"[Memory] LLM resolution failed: {e}. Falling back to regex.")

        return self._regex_fallback(current_query, conversation_history)

    def _summarize_history(self, history: List[Dict[str, Any]]) -> str:
        """
        Compress conversation history to fit LLM context window.
        Strategy: Summarize older turns, keep last 2 verbatim.
        """
        if len(history) <= 3:
            lines = []
            for turn in history:
                q = turn.get("query", "")[:100]
                r = turn.get("response", "")[:100]
                lines.append(f"Turn {turn['turn']}: User: {q} | Agent: {r}")
            return "\n".join(lines)

        # Summarize older turns into topics
        summary_lines = [f"[Summary of Turns 1-{len(history)-2}]"]
        topics = set()
        for turn in history[:-2]:
            q = turn.get("query", "").lower()
            if "repo" in q: topics.add("repo rate")
            if "gdp" in q: topics.add("GDP")
            if "inflation" in q: topics.add("inflation")
            if "npa" in q: topics.add("NPA")
            if "forex" in q or "foreign exchange" in q: topics.add("forex reserves")
            if "credit" in q: topics.add("credit growth")
            if "digital" in q or "upi" in q: topics.add("digital payments")
        summary_lines.append(f"Topics covered: {', '.join(topics) if topics else 'general financial queries'}")

        # Keep last 2 turns verbatim
        for turn in history[-2:]:
            q = turn.get("query", "")[:120]
            r = turn.get("response", "")[:120]
            summary_lines.append(f"Turn {turn['turn']}: User: {q} | Agent: {r}")

        return "\n".join(summary_lines)

    def _regex_fallback(self, query: str, history: List[Dict[str, Any]]) -> str:
        """Lightweight regex-based coreference resolution."""
        resolved = query
        last_turn = history[-1] if history else {}
        last_query = last_turn.get("query", "")
        last_response = last_turn.get("response", "")
        lower_q = f" {query.lower()} "

        # they → RBI
        if " they " in lower_q or query.lower().startswith("how did they"):
            if "rbi" in last_query.lower():
                resolved = re.sub(r'\bthey\b', 'RBI', resolved, flags=re.IGNORECASE)

        # it → last topic
        if " it " in lower_q or query.lower().endswith(" it") or " it?" in lower_q:
            if "inflation" in last_query.lower():
                resolved = re.sub(r'\bit\b', 'inflation', resolved, flags=re.IGNORECASE)
            elif "credit growth" in last_query.lower():
                resolved = re.sub(r'\bit\b', 'credit growth', resolved, flags=re.IGNORECASE)
            elif "repo rate" in last_query.lower():
                resolved = re.sub(r'\bit\b', 'the repo rate', resolved, flags=re.IGNORECASE)
            elif "forex" in last_query.lower() or "foreign exchange" in last_query.lower():
                resolved = re.sub(r'\bit\b', 'the foreign exchange reserve', resolved, flags=re.IGNORECASE)

        # that → last topic
        if " that " in lower_q or query.lower().startswith("compare that"):
            if "npa" in last_query.lower() or "banking sector" in last_query.lower():
                resolved = re.sub(r'\bthat\b', 'the NPA situation', resolved, flags=re.IGNORECASE)
            elif "forex" in last_query.lower() or "foreign exchange" in last_query.lower():
                resolved = re.sub(r'\bthat\b', 'the foreign exchange reserve position', resolved, flags=re.IGNORECASE)
            elif "inflation" in last_query.lower():
                resolved = re.sub(r'\bthat\b', 'inflation', resolved, flags=re.IGNORECASE)

        # previous year / last year
        if "previous year" in query.lower() or "last year" in query.lower():
            ym = re.search(r'FY(\d{4})[-]?(\d{2})', last_query)
            if ym:
                year_start = int(ym.group(1))
                prev_year_str = f"FY{year_start-1}-{str(year_start-1)[2:]}"
                resolved = re.sub(r'previous year|last year', prev_year_str, resolved, flags=re.IGNORECASE)

        # those two / between those
        if "those two" in query.lower() or "between those" in query.lower() or "between them" in query.lower():
            nums = re.findall(r'\d+\.\d+', last_response)
            if len(nums) >= 2:
                resolved = f"percentage increase from {nums[-2]} to {nums[-1]}"
            else:
                nums = re.findall(r'\d+\.\d+', last_query)
                if len(nums) >= 2:
                    resolved = f"percentage increase from {nums[-2]} to {nums[-1]}"

        if resolved != query:
            print(f"[Memory] Regex fallback resolved: '{query}' → '{resolved}'")
        return resolved

    def update_history(
        self,
        history: List[Dict[str, Any]],
        query: str,
        response: str,
        tools_used: List[str]
    ) -> List[Dict[str, Any]]:
        """Append current turn and trim to sliding window."""
        history.append({
            "turn": len(history) + 1,
            "query": query,
            "response": response[:300],
            "tools_used": tools_used,
            "timestamp": None
        })
        # SLIDING WINDOW: keep only last N turns
        if len(history) > self.window_size:
            history = history[-self.window_size:]
        return history


# Global instance
memory_tool = ConversationMemoryTool(window_size=5)