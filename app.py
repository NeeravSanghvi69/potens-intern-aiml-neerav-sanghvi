import streamlit as st
from src.retrieve import retrieve, retrieve_by_doc_id
from src.guardrail import passes_guardrail
from src.llm import answer_from_context, check_contradiction
from src.translate import detect_language, to_english, from_english

st.set_page_config(page_title="Potens Document Q&A", page_icon="📄", layout="wide")
st.title("📄 Document Q&A with Citations")
st.caption("RAG over 5 policy/technical documents · multilingual · no silent hallucination")

tab1, tab2 = st.tabs(["Ask a question", "Check for contradictions"])

with tab1:
    query = st.text_input("Ask something (any language):", placeholder="e.g. Does the EU AI Act ban open-weight models?")
    top_k = st.slider("Chunks to retrieve", 2, 8, 4)

    if st.button("Ask", type="primary") and query:
        with st.spinner("Retrieving and reasoning..."):
            lang_code = detect_language(query)
            query_en = to_english(query, lang_code)
            hits = retrieve(query_en, top_k=top_k)

            if not passes_guardrail(hits):
                st.warning(from_english("I cannot answer this based on the provided documents.", lang_code))
            else:
                result = answer_from_context(query_en, hits)
                if not result.get("covered", True):
                    st.warning(from_english(result["answer"], lang_code))
                else:
                    st.success(from_english(result["answer"], lang_code))

                    used_ids = set(result.get("used_chunk_ids", []))
                    st.subheader("Citations")
                    shown = False
                    for h in hits:
                        chunk_id = f"{h['source_file']}#chunk{h['chunk_index']}"
                        if chunk_id in used_ids:
                            shown = True
                            with st.expander(f"📎 {h['source_file']} — {h['section_title']}"):
                                st.write(h["text"])
                                st.caption(f"similarity: {h['similarity']}")
                    if not shown and hits:
                        h = hits[0]
                        with st.expander(f"📎 {h['source_file']} — {h['section_title']}"):
                            st.write(h["text"])

                st.subheader("Retrieved chunks (debug view)")
                for h in hits:
                    st.caption(f"{h['source_file']} · {h['section_title']} · sim={h['similarity']}")

with tab2:
    st.write("Check whether two documents conflict on a topic.")
    col1, col2 = st.columns(2)
    with col1:
        doc_a = st.text_input("Document A id", value="doc4_openweight_case_for")
    with col2:
        doc_b = st.text_input("Document B id", value="doc5_openweight_case_against")
    topic = st.text_input("Topic to compare", value="cost and difficulty of removing safety fine-tuning")

    if st.button("Check contradiction"):
        with st.spinner("Comparing documents..."):
            chunks_a = retrieve_by_doc_id(doc_a)
            chunks_b = retrieve_by_doc_id(doc_b)
            if not chunks_a or not chunks_b:
                st.error("One or both doc_ids not found. Check the filename without .txt extension.")
            else:
                result = check_contradiction(topic, chunks_a, chunks_b)
                if result["conflict"]:
                    st.error(f"⚠️ Conflict found on: {result['topic']}")
                else:
                    st.success(f"✅ No conflict found on: {result['topic']}")
                st.write(result["reasoning"])
