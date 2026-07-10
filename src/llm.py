import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL_NAME = "llama-3.3-70b-versatile"

ASK_SYSTEM_PROMPT = """You are a document Q&A assistant. You must answer ONLY \
using the provided context chunks. Do not use outside knowledge, even if you \
know the answer. If the context does not contain enough information to answer \
the question, respond with exactly: "I cannot answer this based on the provided documents."

When you do answer, cite which chunk(s) you used by their [chunk_id] tags.
Respond in valid JSON with this exact schema:
{"answer": "...", "used_chunk_ids": ["...", "..."], "covered": true/false}
"""

CONTRADICT_SYSTEM_PROMPT = """You are analyzing whether two documents conflict \
on a topic. You will be given chunks from Document A and Document B. Determine \
if they present conflicting claims, recommendations, or conclusions. Two \
documents that simply cover different topics are NOT a conflict -- only flag \
a genuine disagreement on the same question.
Respond in valid JSON with this exact schema:
{"conflict": true/false, "reasoning": "...", "topic": "..."}
"""


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        _client = Groq(api_key=api_key)
    return _client


def answer_from_context(question: str, hits: list) -> dict:
    context_blocks = []
    for h in hits:
        chunk_id = f"{h['source_file']}#chunk{h['chunk_index']}"
        context_blocks.append(f"[{chunk_id}] ({h['section_title']}):\n{h['text']}")
    context_str = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"CONTEXT CHUNKS:\n\n{context_str}\n\nQUESTION: {question}"

    client = _get_client()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    return json.loads(raw)


def check_contradiction(topic_hint: str, doc_a_chunks: list, doc_b_chunks: list) -> dict:
    doc_a_text = "\n".join(c["text"] for c in doc_a_chunks)
    doc_b_text = "\n".join(c["text"] for c in doc_b_chunks)

    user_prompt = (
        f"TOPIC OF INTEREST: {topic_hint}\n\n"
        f"DOCUMENT A:\n{doc_a_text}\n\n---\n\nDOCUMENT B:\n{doc_b_text}"
    )

    client = _get_client()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CONTRADICT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    return json.loads(raw)


def translate_text(text: str, target_language: str) -> str:
    client = _get_client()
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"Translate the user's text into {target_language}. "
                                           f"Return ONLY the translation, nothing else."},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content.strip()
