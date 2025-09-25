import os, json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_DIR = "/content/segmented_json"
INDEX_PATH = "data/vector_store/legal.index"
DOCS_PATH = "data/vector_store/docs.json"
os.makedirs("data/vector_store", exist_ok=True)

embedder = SentenceTransformer("law-ai/InLegalBERT")
texts, embeddings, doc_map = [], [], []

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".json"):
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        combined_text = " ".join(sum(data["sections"].values(), []))
        emb = embedder.encode(combined_text, convert_to_numpy=True)
        embeddings.append(emb)
        doc_map.append({"file": fname, "text": combined_text})

emb_matrix = np.vstack(embeddings)
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

faiss.write_index(index, INDEX_PATH)
with open(DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(doc_map, f)

print(f"✅ FAISS index built with {len(doc_map)} docs")