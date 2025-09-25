# -------------------------------
# ⿠ Install specific transformers version
# -------------------------------
#!pip install -q --upgrade transformers==4.30.2 datasets==2.13.0 accelerate==0.21.0 evaluate==0.4.6

# -------------------------------
# ⿡ Imports
# -------------------------------
import os, json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# -------------------------------
# ⿢ Paths
# -------------------------------
JSON_DIR = "/kaggle/input/json-files"
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "models/flan_legal_workaround"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# ⿣ Load JSON -> dataset
# -------------------------------
def load_jsons(json_dir):
    samples = []
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            sections = data.get("sections", {})
            input_text = (
                f"Facts: {' '.join(sections.get('facts', [])[:5])} "
                f"Arguments: {' '.join(sections.get('arguments', [])[:5])} "
                f"Statutes: {' '.join(sections.get('statutes', [])[:3])}"
            )
            target_text = " ".join(sections.get("decision", [])[:2])
            if input_text.strip() and target_text.strip():
                samples.append({"input": input_text, "target": target_text})
    return samples

data = load_jsons(JSON_DIR)
dataset = Dataset.from_list(data).train_test_split(test_size=0.1, seed=42)

# -------------------------------
# ⿤ Tokenizer + preprocess
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    model_inputs = tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# -------------------------------
# ⿥ Load model + collator
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -------------------------------
# ⿦ TrainingArguments (old-version workaround)
# -------------------------------
# old versions ignore eval_strategy, so we manually pass eval_steps
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    save_steps=200,
    fp16=False,
    predict_with_generate=True,
    generation_max_length=128,
    report_to="none"
)

# -------------------------------
# ⿧ Trainer
# -------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -------------------------------
# ⿨ Train
# -------------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("✅ Training finished & model saved to",OUTPUT_DIR)