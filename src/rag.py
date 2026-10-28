import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = "knowledge"


def _parse_file(text):
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]
    return meta, text


def load_chunks(base_dir=KNOWLEDGE_DIR):
    chunks = []
    if not os.path.isdir(base_dir):
        return chunks

    for fname in sorted(os.listdir(base_dir)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(base_dir, fname), encoding="utf-8") as f:
            meta, body = _parse_file(f.read())

        
        for part in re.split(r"\n(?=## )", body):
            part = part.strip()
            if part:
                chunks.append({
                    "label": meta.get("label", "GENERAL"),
                    "source": meta.get("source", "غير محدد"),
                    "text": part,
                })
    return chunks


def retrieve(label, query, top_k=4, base_dir=KNOWLEDGE_DIR):
    """يرجّع أقرب المقاطع للنتيجة (label) ولأعراض المريض (query)."""
    chunks = [c for c in load_chunks(base_dir) if c["label"] in (label, "GENERAL")]
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    try:
        vec = TfidfVectorizer().fit(texts + [query])
        sims = cosine_similarity(vec.transform([query]), vec.transform(texts))[0]
        top_idx = sims.argsort()[::-1][:top_k]
    except ValueError:
        
        top_idx = range(min(top_k, len(chunks)))

    return [f"{chunks[i]['text']}\n(المصدر: {chunks[i]['source']})" for i in top_idx]