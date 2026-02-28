import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1) Load scored data
df = pd.read_csv("scored_posts.csv")
raw = pd.read_csv("fake_job_postings.csv")

text_cols = [
    "title","location","department","company_profile","description",
    "requirements","benefits","employment_type","required_experience",
    "required_education","industry","function","salary_range"
]
text_cols = [c for c in text_cols if c in raw.columns]
raw["full_text"] = raw[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

df = df.merge(raw[["job_id","full_text"]], on="job_id", how="left")

# Balance for clean graphs
df_fake = df[df["fraudulent"] == 1]
n = min(200, len(df_fake))
df_fake = df_fake.sample(n, random_state=42)

df_real = df[df["fraudulent"] == 0].sample(n, random_state=42)
df = pd.concat([df_real, df_fake], ignore_index=True)

print("Balanced sample sizes -> REAL:", len(df_real), "FAKE:", len(df_fake))

y_true = df["fraudulent"].astype(int).values
proba  = df["prob_fake"].astype(float).values

# ---------- Chart 1: Confusion Matrix ----------
THRESH = 0.50
pred = (proba >= THRESH).astype(int)

cm = confusion_matrix(y_true, pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["REAL(0)", "FAKE(1)"])
disp.plot(values_format="d")
plt.title(f"Confusion Matrix (threshold={THRESH})")
plt.show()

# ---------- Chart 2: Fraud Rate by Risk Level ----------
bins = [0, 0.33, 0.66, 1.0]
labels = ["Low Risk", "Medium Risk", "High Risk"]

df["risk_level"] = pd.cut(df["prob_fake"], bins=bins, labels=labels, include_lowest=True)

rate = df.groupby("risk_level")["fraudulent"].mean().reindex(labels)

plt.figure()
plt.bar(rate.index.astype(str), rate.values)
plt.xlabel("Risk Level")
plt.title("Fraud Rate by Risk Level")
print(df["risk_level"].value_counts())
print(df.groupby("risk_level")["fraudulent"].mean())
plt.show()

# ---------- Chart 3: Threshold vs Precision ----------
thresholds = np.linspace(0.05, 0.95, 19)
P, R, F = [], [], []

for t in thresholds:
    p = (proba >= t).astype(int)
    P.append(precision_score(y_true, p, zero_division=0))
    R.append(recall_score(y_true, p, zero_division=0))
    F.append(f1_score(y_true, p, zero_division=0))

plt.figure()
plt.plot(thresholds, P, label="Precision")
plt.plot(thresholds, R, label="Recall")
plt.plot(thresholds, F, label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold vs Metrics")
plt.legend()
plt.show()


# ---------- Chart 4: Linear Regression  ----------
feat_cols = ["skill_mismatch_score", "salary_anomaly_score", "role_conf"]

X = df[feat_cols].astype(float)
y = df["prob_fake"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression()
lin.fit(X_train, y_train)
pred_y = lin.predict(X_test)

print("Linear Regression R2:", round(r2_score(y_test, pred_y), 3))
print("Linear Regression MAE:", round(mean_absolute_error(y_test, pred_y), 3))

plt.figure()
plt.scatter(y_test, pred_y)
mn = min(y_test.min(), pred_y.min())
mx = max(y_test.max(), pred_y.max())
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel("Actual prob_fake")
plt.ylabel("Predicted prob_fake")
plt.title("Linear Regression (Actual vs Predicted)")
plt.show()


