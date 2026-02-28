import os
import json
from dotenv import load_dotenv
import pyodbc
from flask import Flask, request, jsonify, send_from_directory
from predict import predict_job

load_dotenv()
def get_conn():
    conn_str = os.getenv("DB_CONN_STR")
    if not conn_str:
        raise RuntimeError("DB_CONN_STR env var is missing.")
    return pyodbc.connect(conn_str)


app = Flask(__name__, static_folder="static")

@app.get("/")
def home():
    return app.send_static_file("fakepostings.html")

@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400
    result = predict_job(text)
    return jsonify(result)

@app.post("/save")
def save():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    result = data.get("result") or {}

    if not text:
        return jsonify({"error": "Text is required"}), 400
    if not result:
        return jsonify({"error": "Result is required"}), 400

    p = float(result.get("prob_fake", 0) or 0)
    status = "LIKELY_FAKE" if p >= 0.70 else "LIKELY_REAL"

    reasons = []
    reasons += (result.get("flags", {}).get("reasons") or [])
    reasons += (result.get("skill_check", {}).get("reasons") or [])
    reasons += (result.get("salary_check", {}).get("reasons") or [])
    reasons_text = "\n".join(dict.fromkeys(reasons)) if reasons else None

    sc = result.get("skill_check") or {}
    sal = result.get("salary_check") or {}

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO dbo.Predictions
            (JobText, Status, ProbFake, Reasons,
             RoleGuess, RoleConfidence, SkillMismatchScore, SkillsFound, SkillReasons,
             SalaryMin, SalaryMax, SalaryZone, SalaryAnomalyScore, SalaryFlag, SalaryReasons,
             InsightsJson)
            VALUES
            (?, ?, ?, ?,
             ?, ?, ?, ?, ?,
             ?, ?, ?, ?, ?, ?,
             ?)
        """, (
            text, status, p, reasons_text,

            sc.get("role_guess"),
            float(sc.get("role_confidence", 0) or 0),
            sc.get("mismatch_score"),
            json.dumps(sc.get("skills_found", []), ensure_ascii=False),
            "\n".join(sc.get("reasons", []) or []),

            sal.get("offered_min"),
            sal.get("offered_max"),
            sal.get("zone"),
            sal.get("anomaly_score"),
            1 if sal.get("flag") else 0,
            "\n".join(sal.get("reasons", []) or []),

            json.dumps(result, ensure_ascii=False)
        ))
        conn.commit()

    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)