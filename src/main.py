from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retrieve import retrieve, retrieve_by_doc_id
from src.guardrail import passes_guardrail
from src.llm import answer_from_context, check_contradiction
from src.translate import detect_language, to_english, from_english

app = FastAPI(title="Potens RAG Q&A", version="1.0")


class AskRequest(BaseModel):
    query: str
    top_k: int = 4


class AskResponse(BaseModel):
    answer: str
    covered: bool
    language: str
    citations: list


class ContradictRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str
    topic: str


class ContradictResponse(BaseModel):
    conflict: bool
    reasoning: str
    topic: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    lang_code = detect_language(req.query)
    query_en = to_english(req.query, lang_code)

    hits = retrieve(query_en, top_k=req.top_k)

    if not passes_guardrail(hits):
        message = "I cannot answer this based on the provided documents."
        translated = from_english(message, lang_code)
        return AskResponse(answer=translated, covered=False, language=lang_code, citations=[])

    result = answer_from_context(query_en, hits)

    if not result.get("covered", True):
        translated = from_english(result["answer"], lang_code)
        return AskResponse(answer=translated, covered=False, language=lang_code, citations=[])

    # build citations only for chunks the model actually says it used
    used_ids = set(result.get("used_chunk_ids", []))
    citations = []
    for h in hits:
        chunk_id = f"{h['source_file']}#chunk{h['chunk_index']}"
        if chunk_id in used_ids:
            citations.append({
                "source_file": h["source_file"],
                "chunk_reference": chunk_id,
                "section": h["section_title"],
                "snippet": h["text"][:280],
            })

    # fallback: if the model didn't name chunk ids properly, cite the top hit
    if not citations and hits:
        h = hits[0]
        citations.append({
            "source_file": h["source_file"],
            "chunk_reference": f"{h['source_file']}#chunk{h['chunk_index']}",
            "section": h["section_title"],
            "snippet": h["text"][:280],
        })

    translated_answer = from_english(result["answer"], lang_code)
    return AskResponse(answer=translated_answer, covered=True, language=lang_code, citations=citations)


@app.post("/contradict", response_model=ContradictResponse)
def contradict(req: ContradictRequest):
    chunks_a = retrieve_by_doc_id(req.doc_id_1)
    chunks_b = retrieve_by_doc_id(req.doc_id_2)

    if not chunks_a:
        raise HTTPException(status_code=404, detail=f"doc_id '{req.doc_id_1}' not found")
    if not chunks_b:
        raise HTTPException(status_code=404, detail=f"doc_id '{req.doc_id_2}' not found")

    result = check_contradiction(req.topic, chunks_a, chunks_b)
    return ContradictResponse(**result)
