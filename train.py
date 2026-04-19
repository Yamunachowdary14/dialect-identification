import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# 🔹 Load dataset
df = pd.read_csv("dialect_dataset_large.csv")

# 🔹 Label mapping (10 dialects)
label_map = {
    "telangana": 0,
    "guntur": 1,
    "krishna": 2,
    "godavari": 3,
    "chittoor": 4,
    "nellore": 5,
    "srikakulam": 6,
    "vizag": 7,
    "kurnool": 8,
    "anantapur": 9
}

df["label"] = df["label"].map(label_map)

# 🔹 Convert to dataset
dataset = Dataset.from_pandas(df)

# 🔹 Load tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize)

# 🔹 Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=10
)

# 🔹 Training config (FAST)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,                 # 🔥 fast
    per_device_train_batch_size=8,      # 🔥 fast
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# 🔹 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 🔹 Train
trainer.train()

# 🔹 Save model
model.save_pretrained("dialect_model")
tokenizer.save_pretrained("dialect_model")

print("✅ Model trained and saved!")