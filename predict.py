# predict.py
import os
import re
import joblib
from skill_salary_rules import run_skill_check, run_salary_check

# ------------------ LOAD MODEL ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_job_model_pipeline.pkl")  # or fake_job_pipeline_v2.pkl
model = joblib.load(MODEL_PATH)


# ------------------ STRONG SCAM INDICATORS ------------------
BANK_WORDS = [
    "ifsc", "account number", "bank account", "salary credit",
    "pan", "aadhaar", "aadhar", "upi", "passbook"
]
TELEGRAM_WORDS = ["telegram", "t me", "@telegram", "t.me"]
FEE_WORDS = [
    "registration fee", "processing fee", "security deposit", "refundable deposit",
    "pay a fee", "payment required", "pay to", "transfer fee", "deposit"
]

NO_INTERVIEW_WORDS = [
    "no interview", "no interview required", "no screening", "direct selection",
    "direct joining", "selection guaranteed", "no hr round"
]
GUARANTEE_WORDS = [
    "guaranteed job", "job guaranteed", "guaranteed selection", "confirm selection",
    "offer letter in", "offer letter within", "instant offer letter"
]
EARN_FAST_WORDS = [
    "earn daily", "earn per day", "earn weekly", "earn per week",
    "quick money", "easy money", "earn from day 1", "start earning today"
]
DATA_ENTRY_SCAM_WORDS = [
    "captcha", "form filling", "copy paste", "typing job", "data entry work",
    "pay per form", "pay per page", "payment per form", "work per submission"
]

# ------------------ SOFT INDICATORS ------------------
WHATSAPP_WORDS = ["whatsapp", "wa me", "wa.me"]
FAST_HIRE_WORDS = ["shortlist today", "immediate joining", "limited seats", "urgent hiring"]
NO_EXP_MONEY = ["no experience needed", "easy money", "quick money", "earn daily"]

# ------------------ LEGIT SIGNALS (DAMPENER) ------------------
LEGIT_SIGNALS = [
    "hr interview", "ops round", "technical interview", "assessment",
    "apply via company", "company website", "careers", "official website",
    "joining within", "notice period", "background verification"
]

# ------------------ NORMALIZATION HELPERS ------------------
def norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s@.\-]", " ", t)  # keep basic email chars
    t = re.sub(r"\s+", " ", t).strip()
    return t

def has_any(t: str, phrases) -> bool:
    return any(p in t for p in phrases)

def has_email(t: str) -> bool:
    # simple email regex (good enough for project)
    return bool(re.search(r"\b[\w.\-]+@[\w\-]+\.(com|in|org|net)\b", t))

# ------------------ PREDICT ------------------
def predict_job(text: str) -> dict:
    raw = (text or "").strip()
    t = norm(raw)

    # ---- Correct probability of FAKE (class 1) ----
    proba = model.predict_proba([raw])[0]
    classes = list(model.classes_)
    model_prob = float(proba[classes.index(1)])  # 1 = fake/fraudulent

    strongFlags = 0
    softFlags = 0
    reasons = []

    # ---------- STRONG RULES ----------
    if has_any(t, BANK_WORDS):
        strongFlags += 1
        reasons.append("Asks for bank/ID details (IFSC/account/Aadhaar/PAN) before an official offer — common scam sign.")

    if has_any(t, TELEGRAM_WORDS):
        strongFlags += 1
        reasons.append("Interview/communication only via Telegram — high scam risk.")

    if has_any(t, FEE_WORDS):
        strongFlags += 1
        reasons.append("Mentions registration/processing fee or deposit for a job — very common scam pattern.")

    # ✅ FIX: No interview should be strong BY ITSELF (not dependent)
    if has_any(t, NO_INTERVIEW_WORDS):
        strongFlags += 1
        reasons.append("No interview/direct selection — strong scam pattern.")

    # Guaranteed/instant offer wording
    if has_any(t, GUARANTEE_WORDS):
        strongFlags += 1
        reasons.append("Guaranteed/instant offer letter promise — very high scam likelihood.")

    # Earn fast strong only with vague easy conditions (reduces false positives)
    if has_any(t, EARN_FAST_WORDS) and has_any(t, ["work from home", "wfh", "part time", "no experience"]):
        strongFlags += 1
        reasons.append("Promises fast earnings (daily/weekly) with vague requirements — common scam pattern.")

    if has_any(t, DATA_ENTRY_SCAM_WORDS):
        strongFlags += 1
        reasons.append("Mentions captcha/form-filling/pay-per-form work — extremely common scam format.")

    # ---------- SOFT RULES ----------
    if has_any(t, WHATSAPP_WORDS):
        softFlags += 1
        reasons.append("WhatsApp-only contact can be suspicious if company cannot be verified.")

    if has_any(t, FAST_HIRE_WORDS):
        softFlags += 1
        reasons.append("Overly urgent hiring language (shortlist today / immediate joining).")

    if has_any(t, NO_EXP_MONEY):
        softFlags += 1
        reasons.append("Vague hiring conditions (WFH/part-time/no experience) can be suspicious in scam posts.")

    # ---- Feature checks ----
    skill_check = run_skill_check(raw)
    salary_check = run_salary_check(raw, skill_check["role_guess"], skill_check["role_confidence"])

    # ---- Combine ML + rules ----
    final_prob = float(model_prob)

    # ✅ IMPORTANT: don’t instantly force 0.80 for 1 flag (ML becomes useless)
    if strongFlags == 1:
        final_prob = max(final_prob, 0.65)  # CHECK-ish
    elif strongFlags >= 2:
        final_prob = max(final_prob, 0.85)  # FAKE-ish

    # Soft indicators can push to borderline (not FAKE)
    if strongFlags == 0 and softFlags >= 2 and model_prob >= 0.35:
        final_prob = max(final_prob, 0.45)

    # Salary RED can push toward FAKE when role confidence is decent
    if salary_check.get("zone") == "RED" and (skill_check.get("role_confidence", 0) >= 0.60):
        final_prob = max(final_prob, 0.70)

    # ✅ LEGIT DAMPENER: if strongFlags=0 and legit signals exist, cap risk
    legit = 0
    if has_any(t, LEGIT_SIGNALS):
        legit += 1
    if has_email(t):
        legit += 1

    if strongFlags == 0 and legit >= 2:
        final_prob = min(final_prob, 0.45)

    # ---- Label ----
    label = "FAKE" if final_prob >= 0.70 else ("CHECK" if final_prob >= 0.40 else "REAL")

    return {
        "prob_fake": round(final_prob, 4),
        "model": {"prob_fake": round(model_prob, 4), "label": label},
        "flags": {"strong": int(strongFlags), "soft": int(softFlags), "reasons": reasons},
        "skill_check": skill_check,
        "salary_check": salary_check
    }
