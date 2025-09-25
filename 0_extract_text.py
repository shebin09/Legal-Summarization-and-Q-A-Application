import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

INPUT_DIR = r"D:\project_law_1\data\raw_pdfs"
OUTPUT_DIR = r"D:\project_law_1\data\raw_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_with_pypdf2(pdf_path):
    """Try text extraction with PyPDF2."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def extract_with_ocr(pdf_path):
    """Fallback: OCR using pdf2image + pytesseract."""
    pages = convert_from_path(pdf_path)
    text = []
    for i, page in enumerate(pages, start=1):
        page_text = pytesseract.image_to_string(page, lang="eng")
        text.append(page_text)
        print(f"   🔍 OCR processed page {i}/{len(pages)}")
    return "\n".join(text)

for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith(".pdf"):
        pdf_path = os.path.join(INPUT_DIR, fname)
        print(f"\n📄 Processing: {os.path.abspath(pdf_path)}")

        # Step 1: Try PyPDF2
        text = extract_with_pypdf2(pdf_path)
        method = "PyPDF2"

        # Step 2: If empty, fallback to OCR
        if not text.strip():
            print(f"⚠️ No text with PyPDF2 → falling back to OCR")
            text = extract_with_ocr(pdf_path)
            method = "OCR"

        # Step 3: Build proper .txt filename
        base, _ = os.path.splitext(fname)
        outpath = os.path.join(OUTPUT_DIR, base + ".txt")

        # Step 4: Write text (even if empty)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved: {os.path.abspath(outpath)} ({len(text)} chars, method: {method})")

print("\n📂 Final contents of OUTPUT_DIR:")
for f in os.listdir(OUTPUT_DIR):
    print(" -", os.path.abspath(os.path.join(OUTPUT_DIR, f)))
