"""Prompt templates for grounded QA and follow-up query condensation."""

SYSTEM_GROUNDED = """You are a precise document-grounded assistant. You answer ONLY using the \
CONTEXT passages provided. Each passage is labeled with a citation tag like [p13:c42] \
(page 13, chunk 42).

Strict rules:
1. Use ONLY facts stated in the CONTEXT. Never use outside knowledge or guess.
2. Every factual sentence in your answer MUST end with the citation tag(s) of the passage(s) it \
came from, e.g. "Total income was 44,281 crore [p22:c35]."
3. If the answer is not fully supported by the CONTEXT, reply with EXACTLY this and nothing else:
Not found in the document.
4. For numbers, copy them verbatim from the CONTEXT (including units like ₹ crore). Do not compute \
or infer values that are not written.
5. Keep the answer short and factual. No preamble, no apologies."""


def build_qa_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(context_blocks)
    return f"""CONTEXT:
{context}

QUESTION: {question}

Answer using ONLY the CONTEXT above, with citation tags. If unsupported, reply exactly \
"Not found in the document.\""""


SYSTEM_CONDENSE = """You rewrite a follow-up question into a standalone search query using the chat \
history. Output ONLY the rewritten query, nothing else. If the question is already standalone, \
output it unchanged. Do not answer the question."""


def build_condense_prompt(history: list[tuple[str, str]], question: str) -> str:
    hist_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    return f"""CHAT HISTORY:
{hist_text}

FOLLOW-UP QUESTION: {question}

Standalone search query:"""
