from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
app.config['DATABASE'] = 'contacts.db'


def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT,
            created_at TEXT NOT NULL
        )
        '''
    )
    conn.commit()
    conn.close()


init_db()

MODEL_PATH = "./dialect_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

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

def predict_dialect(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = float(torch.max(probs))

    return label_map[pred], round(confidence * 100, 2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("sentence")
        if text:
            result, confidence = predict_dialect(text)

    return render_template("predict.html", result=result, confidence=confidence)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    success = False
    error = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()

        if not name or not email:
            error = 'Name and email are required.'
        else:
            try:
                conn = get_db_connection()
                conn.execute(
                    'INSERT INTO contacts (name, email, message, created_at) VALUES (?, ?, ?, ?)',
                    (name, email, message, datetime.utcnow().isoformat())
                )
                conn.commit()
                conn.close()
                success = True
            except Exception as e:
                error = 'Failed to save message.'

    return render_template("contact.html", success=success, error=error)

if __name__ == "__main__":
    app.run(debug=True)