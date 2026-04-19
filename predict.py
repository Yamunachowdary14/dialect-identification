from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 🔹 Load model
model_path = "./dialect_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 🔹 Label mapping
label_map = {
    0: "Telangana",
    1: "Guntur",
    2: "Krishna",
    3: "Godavari",
    4: "Chittoor",
    5: "Nellore",
    6: "Srikakulam",
    7: "Vizag",
    8: "Kurnool",
    9: "Anantapur"
}

print("🔹 Telugu Dialect Predictor (type 'exit' to stop)\n")

while True:
    text = input("Enter sentence: ")

    if text.lower() == "exit":
        break

    # 🔹 Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # 🔹 Predict
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()

    # 🔹 Confidence (optional)
    probs = torch.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()

    print(f"Dialect: {label_map[pred]} (Confidence: {confidence:.2f})\n")