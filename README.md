# 🚨 Fake Job Posting Detection System

A real-time fraud detection system that analyzes job postings against rule-based and ML patterns to flag suspicious job listings and protect job seekers.

## 🎯 Problem Statement

Job scams cost Indian job seekers time, money, and trust. Fake postings exploit common red flags:
- Unrealistic salary claims (e.g., ₹500k/month for entry-level roles)
- Vague job descriptions with minimal details
- Pressure for upfront payments or personal data
- Copy-pasted or templated language
- Suspicious company domains or missing company verification

This system catches these signals **in real-time** as users input job postings, providing instant fraud risk scores.

## 📊 Approach

**Two-layer detection:**

1. **Rule-based checks** — Validates postings against industry knowledge:
   - Salary bounds by role/experience level (stored in `salary_bands_inr.json`)
   - Red-flag keywords (e.g., "easy money," "guaranteed salary," "no experience required")
   - Job description quality (length, specificity, language patterns)
   - Company verification against known scam domains

2. **Machine Learning model** — Trained to recognize subtle fraud patterns:
   - Combination of features that correlate with scams
   - Handles edge cases that simple rules miss
   - Outputs fraud probability score (0-1)

3. **SQL Database** — Tracks submitted postings and salary anomalies across industries over time

## 🔍 Real-Time Analysis

When a user submits a job posting:
1. System extracts job description, salary, company, designation
2. Validates against salary_bands_inr.json (industry reference)
3. Checks against rules_catalog.json (fraud indicators)
4. Runs ML model to compute overall fraud score
5. Returns risk level: ✅ Likely Genuine | ⚠️ Moderate Risk | 🚨 High Risk

**Tested on:** 150+ job postings from multiple sources (Naukri, Indeed, LinkedIn)

## 🛠️ Tech Stack

- **Python** — scikit-learn (ML model), Flask (web app)
- **SQL** — Relational database for storing submissions and salary trends
- **Data Files** — JSON-based rule catalogs and salary reference bands
- **Serialization** — Pickle for model persistence

## 📁 Project Structure

```
├── app.py                       # Flask web app for real-time predictions
├── predict.py                   # Prediction logic using trained model + rules
├── train_model.py               # Model training pipeline
├── skill_salary_rules.py        # Salary validation logic by role/experience
├── make_graphs.py               # Visualize fraud patterns
├── make_scored_csv.py           # Batch scoring (if needed)
├── fake_job_model_pipeline.pkl  # Trained ML model
├── salary_bands_inr.json        # Reference salary ranges by role/level
├── rules_catalog.json           # Rule-based fraud indicators
├── fake_job_postings.csv        # Reference dataset
└── static/                      # Frontend assets
```

## 🚀 How to Use

### Setup
```bash
pip install -r requirements.txt
```

### Run Web App
```bash
python app.py
# Visit http://localhost:5000 to submit a job posting for analysis
```

The system will analyze the posting and return:
- **Fraud Risk Score** (0-1, higher = more suspicious)
- **Flagged Red Flags** (specific issues detected)
- **Recommendation** (Likely Genuine / Moderate Risk / High Risk)

## 📈 What the System Detects

✅ **Salary Anomalies** — Compares claimed salary against industry benchmarks by role/experience
✅ **Keyword Red Flags** — Patterns like "guaranteed," "easy money," "work from home no experience"
✅ **Description Quality** — Vague or suspiciously short job descriptions
✅ **Company Verification** — Cross-checks against known scam domains
✅ **Predictive Signals** — ML model catches subtle patterns from training

## 🎓 What I Learned

- Building a fraud detection model requires domain knowledge (salary research, industry red flags)
- Data imbalance (more genuine postings than fakes) needs careful handling
- Real-world data is messy: job titles aren't standardized, salaries are in different currencies/formats
- Rule-based checks complement ML: some red flags are too obvious to miss with data alone

## 💡 Future Improvements

- Integrate with job boards in real-time to flag postings as they're posted
- Track scammer patterns over time (which companies/domains appear repeatedly)
- Build browser extension for auto-checking while job searching

## 📞 Questions?
Feel free to reach out if you'd like to discuss the approach or see predictions on a specific posting.

---
