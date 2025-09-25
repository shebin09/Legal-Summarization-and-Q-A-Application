# ⚖️ Legal Document Summarizer & Q&A Chatbot

Extract text from legal PDFs, generate structured summaries, and ask questions through a chatbot powered by **RAG (InLegalBERT + FAISS + FLAN-T5)**.

##  How to Run
```python 
# Preprocess PDFs
python scripts/0_extract_text.py
python scripts/1_segment_text.py
python scripts/2_build_index.py

# Train summarizer
python scripts/3_train_summarizer.py

# Launch web app
python app.py
