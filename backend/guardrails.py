"""
Lightweight, dependency-free guardrails for the RAG pipeline.

These are deliberately simple, regex/heuristic-based checks -- enough to
demonstrate the pattern (validate input before it reaches the LLM, validate
output before it reaches the user) without pulling in a heavy framework.
For a production system, consider swapping this module's internals for a
library like `guardrails-ai` or `nemo-guardrails` while keeping the same
check_input / check_output interface so main.py doesn't need to change.
"""

import re
from dataclasses import dataclass

from config import settings

# --- Prompt injection heuristics -----------------------------------------
# Not exhaustive -- these catch the common, obvious patterns. Treat this as
# a first line of defense, not a guarantee.
_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) instructions",
    r"disregard (all |the )?(previous|above|prior) (instructions|rules)",
    r"you are now (a|an) ",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (if you (are|were)|a) (jailbroken|unrestricted|dan)",
    r"pretend (you have no|there are no) (restrictions|rules|filters)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# --- Sensitive / off-topic denylist ---------------------------------------
# Simple keyword screen. Extend as needed for your use case.
_SENSITIVE_PATTERNS = [
    r"\bhow (do|to) (i |you )?(make|build|synthesi[sz]e) (a bomb|explosives?|nerve gas)",
    r"\bhow (do|to) (i |you )?hack (into|someone)",
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)

# --- Patterns that should never appear in an answer to the user -----------
_SECRET_LEAK_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")


@dataclass
class GuardrailResult:
    passed: bool
    reason: str | None = None


def check_input(question: str) -> GuardrailResult:
    """Validate a user question before it's sent to retrieval/the LLM."""
    if not question or not question.strip():
        return GuardrailResult(False, "Question cannot be empty.")

    if len(question) > settings.max_question_chars:
        return GuardrailResult(
            False,
            f"Question is too long ({len(question)} chars). "
            f"Keep it under {settings.max_question_chars} characters.",
        )

    if settings.block_suspected_injection and _INJECTION_RE.search(question):
        return GuardrailResult(
            False,
            "This question looks like it's trying to override the assistant's "
            "instructions, which isn't allowed. Please rephrase.",
        )

    if _SENSITIVE_RE.search(question):
        return GuardrailResult(
            False, "This question is outside what this assistant can help with."
        )

    return GuardrailResult(True)


def check_output(answer: str, context_chunks: list[dict]) -> GuardrailResult:
    """Sanity-check the LLM's answer before it's returned to the user.

    This does NOT block the response (the answer is still returned) -- it
    attaches a notice so the frontend can show a warning banner. Swap this
    for a hard block in check_output's caller if you'd rather refuse
    ungrounded answers outright.
    """
    if _SECRET_LEAK_RE.search(answer):
        return GuardrailResult(False, "Response withheld: it appeared to contain an API key.")

    if not answer or not answer.strip():
        return GuardrailResult(False, "The model returned an empty response.")

    if not context_chunks:
        return GuardrailResult(
            False, "No supporting context was retrieved -- treat this answer with caution."
        )

    return GuardrailResult(True)
