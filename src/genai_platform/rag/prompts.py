"""STEP 5 - prompt templates. The prompt IS the anti-hallucination mechanism, so it lives in one file."""
from __future__ import annotations

import re
from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate

REFUSAL = "I don't know based on the provided documents."

RAG_SYSTEM = f"""You are a careful assistant that answers ONLY from the numbered context passages.
Rules:
1. Use nothing but the context. Do not use outside knowledge, even if you are sure.
2. Cite every factual sentence with the passage number in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the answer, reply exactly: {REFUSAL}
4. Never cite a number that is not in the context. Never invent sources.
5. The conversation history is only for resolving references such as "it" or "that"; it is not evidence.Be concise and direct.
6. Write a complete sentence that answers the question, then the citation, e.g. "The warranty is 12 months [1]." Never reply with only a citation.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM),
        ("human", "Conversation so far:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"),
    ]
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You rewrite search queries for a document retrieval system. Given a user question, write {n} "
                   "alternative queries that would find the same information using different wording, synonyms, or "
                   "a more specific/more general angle. Output ONE query per line, no numbering, no commentary."),
        ("human", "{question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You compress conversations. Merge the previous summary with the new turns into ONE dense summary "
                   "(max 120 words). Keep: names, numbers, decisions, user preferences, open questions, document names. "
                   "Drop: greetings and filler. Output the summary only."),
        ("human", "Previous summary:\n{summary}\n\nNew turns:\n{turns}\n\nUpdated summary:"),
    ]
)


def format_context(hits: Sequence) -> str:
    if not hits:
        return "(no passages retrieved)"
    header = f"There are {len(hits)} passage(s), numbered [1] to [{len(hits)}]. Cite only these numbers.\n\n"
    return header + "\n\n".join(f"[{i}] (source: {h.cite})\n{h.text}" for i, h in enumerate(hits, start=1))


_CITE = re.compile(r"\[(\d+)\]")


def validate_citations(answer: str, n_passages: int) -> dict:
    """Post-hoc guard: every [n] must point at a real passage; a non-refusal answer must cite something."""
    cited = sorted({int(x) for x in _CITE.findall(answer)})
    invalid = [c for c in cited if not 1 <= c <= n_passages]
    refused = REFUSAL.lower() in answer.lower()
    return {"cited": [c for c in cited if c not in invalid], "invalid": invalid, "refused": refused,
            "grounded": refused or (bool(cited) and not invalid)}
