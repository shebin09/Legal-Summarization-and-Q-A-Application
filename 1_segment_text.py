import os, json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

INPUT_DIR = r"D:\project_law_1\data\raw_text"
OUTPUT_DIR = r"D:\project_law_1\data\segmented_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Working dir:", os.getcwd())
print("Absolute INPUT_DIR:", os.path.abspath(INPUT_DIR))
print("Absolute OUTPUT_DIR:", os.path.abspath(OUTPUT_DIR))

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"❌ Input dir not found: {os.path.abspath(INPUT_DIR)}")

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
if not files:
    raise RuntimeError(f"⚠️ No .txt files found in {os.path.abspath(INPUT_DIR)}")

# Load InLegalBERT (force failure if unavailable)
try:
    embedder = SentenceTransformer("law-ai/InLegalBERT")
except Exception as e:
    raise RuntimeError("❌ Failed to load model 'law-ai/InLegalBERT'. "
                       "Check internet connection or HuggingFace access.") from e

SECTION_LABELS = ["facts", "arguments", "statutes", "decision"]

def segment_document(text, n_clusters=4):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
    if not paragraphs:
        return {l: [] for l in SECTION_LABELS}

    embeddings = embedder.encode(paragraphs, convert_to_numpy=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    clustered = {l: [] for l in SECTION_LABELS}

    for i, p in enumerate(paragraphs):
        label = SECTION_LABELS[kmeans.labels_[i]]
        clustered[label].append(p)

    return clustered

for fname in files:
    with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
        text = f.read()
    sections = segment_document(text)
    base, _ = os.path.splitext(fname)
    outpath = os.path.join(OUTPUT_DIR, base + ".json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({"sections": sections}, f, indent=2)
    print(f"✅ Wrote {os.path.abspath(outpath)}")

print("\n🎯 Segmentation complete.")
