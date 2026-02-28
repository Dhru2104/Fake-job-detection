print("✅ make_scored_csv started")

import pandas as pd
import joblib
import numpy as np
from skill_salary_rules import run_skill_check, run_salary_check

# 1) Load dataset
df = pd.read_csv("fake_job_postings.csv")

# ✅ BALANCED SAMPLE (fast + graphs clear)
df_fake = df[df["fraudulent"] == 1]
df_real = df[df["fraudulent"] == 0]

n = min(200, len(df_fake))  # if less than 200 fake, take all
df_fake = df_fake.sample(n, random_state=42)
df_real = df_real.sample(n, random_state=42)

df = pd.concat([df_real, df_fake], ignore_index=True)
print("Balanced sample sizes -> REAL:", len(df_real), "FAKE:", len(df_fake))

# 2) Build full_text (same as training)
text_cols = [
    "title","location","department","company_profile","description",
    "requirements","benefits","employment_type","required_experience",
    "required_education","industry","function",
    "salary_range"
]
text_cols = [c for c in text_cols if c in df.columns]
df["full_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

# 3) Load saved ML model
model = joblib.load("fake_job_model_pipeline.pkl")

# 4) ML outputs
df["prob_fake"] = model.predict_proba(df["full_text"])[:, 1]
THRESH = 0.50
df["pred_label"] = (df["prob_fake"] >= THRESH).astype(int)

role_guess_list = []
role_conf_list = []
mismatch_list = []
salary_anom_list = []

texts = df["full_text"].tolist()

for i, text in enumerate(texts, start=1):
    if i % 50 == 0:
        print(f"Processed {i}/{len(texts)} rows...")

    skill_out = run_skill_check(text)

    role_guess = skill_out["role_guess"]
    role_conf  = skill_out["role_confidence"]
    mismatch   = skill_out["mismatch_score"]

    sal_out = run_salary_check(text, role_guess, role_conf)
    salary_anom = sal_out["anomaly_score"]

    role_guess_list.append(role_guess)
    role_conf_list.append(role_conf)

    # graphs-only neutral fill (does NOT affect your main app)
    mismatch_list.append(0.30 if mismatch is None else float(mismatch))
    salary_anom_list.append(0.0 if salary_anom is None else float(salary_anom))

df["role_guess"] = role_guess_list
df["role_conf"] = role_conf_list
df["skill_mismatch_score"] = mismatch_list
df["salary_anomaly_score"] = salary_anom_list

out_cols = [
    "job_id", "fraudulent", "prob_fake", "pred_label",
    "role_guess", "role_conf",
    "skill_mismatch_score", "salary_anomaly_score"
]
out_cols = [c for c in out_cols if c in df.columns]

df[out_cols].to_csv("scored_posts.csv", index=False)
print("✅ Saved scored_posts.csv with:", out_cols)

print("Mismatch NaN:", df["skill_mismatch_score"].isna().sum())
print("Salary NaN:", df["salary_anomaly_score"].isna().sum())

print(df.loc[(df["fraudulent"]==1) | (df["pred_label"]==1),
             ["job_id","fraudulent","prob_fake","pred_label","role_guess","skill_mismatch_score","salary_anomaly_score"]
            ].head(30))

print(df.sort_values("prob_fake", ascending=False)[["job_id","fraudulent","prob_fake","pred_label"]].head(20))

# quick preview
print(pd.read_csv("scored_posts.csv").head())
print("✅ make_scored_csv finished")
