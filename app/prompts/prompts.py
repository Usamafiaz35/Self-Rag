"""
Chat prompt templates for each LLM step (kept separate from node code).

This file exists so prompt wording can evolve without touching routing or graph wiring.
"""

from langchain_core.prompts import ChatPromptTemplate

answer_source_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify how to answer the user's latest message.\n"
            "Return JSON with key: source (conversation or documents).\n\n"
            "- conversation: The answer must come from the chat history (e.g. user's name they "
            "stated earlier, listing prior questions, what was said in this thread).\n"
            "- documents: The answer needs company PDF facts (policies, pricing, leadership from docs), "
            "even if history provides helpful context.\n\n"
            "If the user asks about this chat or themselves from earlier messages, choose conversation.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\nLatest question: {question}",
        ),
    ]
)

history_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer using ONLY the conversation history below.\n"
            "Rules:\n"
            "- Use facts explicitly stated by the user or assistant in this thread.\n"
            "- If listing questions the user asked, include every user message that was a question.\n"
            "- Do not invent names or facts not in the history.\n"
            "- If the history does not contain the answer, say clearly what is missing.\n"
            "- Do not use company document knowledge; only the chat transcript.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\nLatest question: {question}",
        ),
    ]
)

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval from company PDFs is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from company documents.\n"
            "- should_retrieve=False if the conversation history already has enough to answer, "
            "or the question is general knowledge.\n"
            "- If unsure, choose True.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\nLatest question: {question}",
        ),
    ]
)

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the latest question.\n"
            "Use the conversation history when it contains relevant facts (e.g. the user's name).\n"
            "Use general knowledge only when needed.\n"
            "If the question needs specific company facts not in history, say you need company documents.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\nLatest question: {question}",
        ),
    ]
)

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business rag chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "You may also receive conversation history from this chat thread.\n"
            "Prefer CONTEXT for company facts; use conversation history for personal details "
            "the user shared earlier (e.g. their name).\n"
            "Do not mention that you are using a context block in your answer.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Latest question:\n{question}\n\n"
            "Context:\n{context}",
        ),
    ]
)

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Latest question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n",
        ),
    ]
)

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}",
        ),
    ]
)

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line.",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Latest question:\n{question}\n\nAnswer:\n{answer}",
        ),
    ]
)

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}",
        ),
    ]
)
