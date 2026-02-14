import joblib

model = joblib.load("fake_job_model_pipeline.pkl")

samples = [
    "Work from home. Earn 50,000 per week. No experience needed. WhatsApp now!",
    "We are hiring a Python developer. Responsibilities include API development, unit testing, and code reviews."
]

proba = model.predict_proba(samples)[:, 1]
pred = (proba >= 0.5).astype(int)

for s, p, y in zip(samples, proba, pred):
    print("-"*60)
    print("P(fake):", round(float(p), 4), "=>", "FAKE" if y==1 else "REAL")
    print("Text:", s[:120], "...")
