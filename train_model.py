import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# 1) Load data
df = pd.read_csv("fake_job_postings.csv")

# 2) Target
y = df["fraudulent"].astype(int)

# 3) Combine text fields
text_cols = [
    "title","location","department","company_profile","description",
    "requirements","benefits","employment_type","required_experience",
    "required_education","industry","function"
]
text_cols = [c for c in text_cols if c in df.columns]
X_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=200000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

model.fit(X_train, y_train)

# 6) Evaluate
proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, (proba >= 0.5).astype(int), digits=4))

# 7) Save model
joblib.dump(model, "fake_job_model_pipeline.pkl")
print("Saved: fake_job_model_pipeline.pkl")
