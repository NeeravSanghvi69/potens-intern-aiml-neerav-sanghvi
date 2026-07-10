"""
Chunking strategy
------------------
Documents in data/ are plain-text with markdown-style '## Section' headers.
We chunk by section first (structural chunking) rather than blind fixed-size
windows, because cutting mid-sentence destroys the meaning a citation is
supposed to preserve. If a section is still too long, we fall back to a
sliding window with overlap so no single chunk exceeds ~500 words.

Each chunk keeps metadata: source_file, section_title, chunk_index.
This metadata is what makes citations possible downstream.
"""

import re
import os

MAX_WORDS_PER_CHUNK = 220
OVERLAP_WORDS = 40


def _split_into_sections(text: str):
    """Split on markdown '##' headers, keeping the header with its body."""
    pattern = r"(?=^## )"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        header_match = re.match(r"^##\s*(.+)", part)
        title = header_match.group(1).strip() if header_match else "Untitled"
        sections.append({"title": title, "body": part})
    return sections


def _sliding_window(words, max_words=MAX_WORDS_PER_CHUNK, overlap=OVERLAP_WORDS):
    if len(words) <= max_words:
        return [words]
    windows = []
    start = 0
    while start < len(words):
        end = start + max_words
        windows.append(words[start:end])
        if end >= len(words):
            break
        start = end - overlap
    return windows


def chunk_document(filepath: str):
    """
    Returns a list of dicts: {text, source_file, section_title, chunk_index}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    filename = os.path.basename(filepath)
    sections = _split_into_sections(raw_text)
    chunks = []
    chunk_idx = 0

    for section in sections:
        words = section["body"].split()
        if len(words) < 8:
            # skip trivial fragments like a lone document title line
            continue
        windows = _sliding_window(words)
        for w in windows:
            chunk_text = " ".join(w)
            chunks.append({
                "text": chunk_text,
                "source_file": filename,
                "section_title": section["title"],
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1

    return chunks
