from flask import Flask, request, jsonify, send_from_directory
import joblib
import pyodbc

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=Fakejobpostings;"
    "Trusted_Connection=yes;"
)


def get_conn():
    return pyodbc.connect(CONN_STR)


def save_prediction(job_text, status, prob_fake, ui_message, reasons_text):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO dbo.Predictions (JobText, Status, ProbFake, UiMessage, Reasons)
            VALUES (?, ?, ?, ?, ?)
        """, (job_text, status, prob_fake, ui_message, reasons_text))
        conn.commit()

app = Flask(__name__, static_folder="static")

model = joblib.load("fake_job_model_pipeline.pkl")
THRESHOLD = 0.5

@app.get("/")
def home():
  return send_from_directory("static", "fakepostings.html")

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    p = float(model.predict_proba([text])[0][1])
    return jsonify({"prob_fake": p})

@app.post("/save")
def save():
    data = request.get_json(force=True)

    text = (data.get("text") or "").strip()
    status = (data.get("status") or "").strip()
    prob_fake = float(data.get("prob_fake", 0))
    ui_message = (data.get("ui_message") or "").strip()
    reasons_text = (data.get("reasons_text") or "").strip() or None

    if not text:
        return jsonify({"error": "Text is required"}), 400

    if status not in ("LIKELY_REAL", "BORDERLINE", "LIKELY_FAKE"):
        return jsonify({"error": "Invalid status"}), 400

    save_prediction(text, status, prob_fake, ui_message, reasons_text)
    return jsonify({"ok": True})


if __name__ == "__main__":
  app.run(debug=True)
