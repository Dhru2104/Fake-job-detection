import json, re, math
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def CFG():
    return json.loads((Path(__file__).with_name("rules_catalog.json")).read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def BANDS():
    return json.loads((Path(__file__).with_name("salary_bands_inr.json")).read_text(encoding="utf-8"))

def _norm(t):
    # IMPORTANT: do NOT remove commas here (salary needs original separators)
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _alias(t):
    a = CFG().get("aliases", {})
    for k in sorted(a, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(k)}\b", a[k], t)
    return t

@lru_cache(maxsize=1)
def SKILL_RE():
    vocab = sorted({s for r in CFG()["roles"] for s in r.get("skills", [])}, key=len, reverse=True)
    parts = [re.escape(s).replace(r"\ ", r"\s+") for s in vocab]
    return re.compile(rf"\b({'|'.join(parts)})\b", re.I) if parts else re.compile(r"$^")

def extract_skills(text):
    hits = SKILL_RE().findall(_alias(_norm(text)))
    return sorted({re.sub(r"\s+", " ", h.strip().lower()) for h in hits})

def guess_role(text):
    raw = text or ""
    title = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    T = _alias(_norm(title))
    B = _norm(raw)   # no alias on full raw (speed)

    best, score = "Generic", 0
    for r in CFG()["roles"]:
        kws = r.get("keywords", [])
        s = 2*sum(k in T for k in kws) + 1*sum(k in B for k in kws)
        if s > score:
            best, score = r["name"], s

    conf = 0.30 if best == "Generic" or score <= 0 else (0.80 if score >= 3 else 0.62)
    return best, conf

def run_skill_check(text):
    role, conf = guess_role(text)
    found = extract_skills(text)

    role_obj = next((r for r in CFG()["roles"] if r["name"] == role), None)
    exp = set(role_obj.get("skills", [])) if role_obj else set()

    # Evidence gate
    enough = (role != "Generic" and conf >= 0.60 and len(found) >= 4 and len(exp) >= 5)

    # Not enough evidence: return N/A (None)
    if not enough:
        rs = []
        if len(found) < 4:
            rs.append("Not enough explicit skills detected to judge mismatch confidently.")
        if role == "Generic" or conf < 0.60:
            rs.append("Role confidence is low, so mismatch detection is conservative.")
        if not rs:
            rs.append("Skill mismatch check skipped (low evidence).")

        return {
            "role_guess": role,
            "role_confidence": round(conf, 3),
            "skills_found": found,
            "mismatch_score": None,
            "off_role_skills": [],
            "flag": False,
            "reasons": rs
        }

    off = [s for s in found if s not in exp]
    score = len(off) / max(1, len(found))
    flag = score >= 0.70

    rs = (
        [
            f"Many detected skills don’t match typical {role} requirements (off-role ratio ≈ {score:.2f}).",
            "This often happens when posts are copy-pasted or mismatched templates."
        ]
        if flag
        else [f"Skills look broadly consistent for {role}."]
    )

    return {
        "role_guess": role,
        "role_confidence": round(conf, 3),
        "skills_found": found,
        "mismatch_score": round(score, 3),
        "off_role_skills": off,
        "flag": bool(flag),
        "reasons": rs
    }

# ---------------- SALARY PARSING (FIXED) ----------------

def _num(s: str) -> int:
    """
    Parses salary tokens like:
    90,000   1,20,000   1 20 000   1.20.000   120000   80k   120k   12000/-
    """
    s = (s or "").strip().lower()
    s = s.replace("₹", "").replace("rs", "").strip()
    s = s.replace("/-", "").strip()
    # remove commas, spaces, dots used as separators
    s = re.sub(r"[,\s\.]", "", s)

    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    return int(float(s))

def parse_salary_inr_month(text):
    t = _alias(_norm(text)).replace("₹", " rs ")
    month_hint = bool(re.search(r"\b(per month|monthly|month|stipend|pm)\b", t))

    # RANGE: 90,000 – 1,20,000 per month  OR  80k-120k pm
    m = re.search(r"\b([\d,\.\s]+k?)\s*(to|\-|–)\s*([\d,\.\s]+k?)\b", t)
    if m and month_hint:
        a, b = _num(m.group(1)), _num(m.group(3))
        return {"ok": True, "min": min(a, b), "max": max(a, b), "confidence": "HIGH"}

    # SINGLE: 12,000 per month  OR  stipend: 15k pm
    m = re.search(r"\b(stipend\s*[:\-]?\s*)?([\d,\.\s]+k?)\s*(per month|monthly|month|pm)\b", t)
    if m:
        v = _num(m.group(2))
        return {"ok": True, "min": v, "max": v, "confidence": "MED"}

    # RANGE LPA: 6-10 lpa
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(to|\-|–)\s*(\d+(?:\.\d+)?)\s*\b(lpa)\b", t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(3))
        return {
            "ok": True,
            "min": int(min(lo, hi) * 100000 / 12),
            "max": int(max(lo, hi) * 100000 / 12),
            "confidence": "HIGH"
        }

    # SINGLE LPA: 8 lpa / ctc 8 lpa
    m = re.search(r"\b(ctc\s*)?(\d+(?:\.\d+)?)\s*\b(lpa)\b", t)
    if m:
        v = float(m.group(2))
        v = int(v * 100000 / 12)
        return {"ok": True, "min": v, "max": v, "confidence": "MED"}

    return {"ok": False, "reason": "No clear salary detected."}

def run_salary_check(text, role_guess, role_conf):
    p = parse_salary_inr_month(text)
    if not p["ok"]:
        return {
            "salary_parsed": False,
            "zone": "NO_SALARY",
            "anomaly_score": None,
            "flag": False,
            "reasons": [p.get("reason", "Salary not found.")],
            "ui": {"gauge_pct": 0, "label": "No salary found", "theme": "NEUTRAL"}
        }

    role_key = CFG().get("salary_role_map", {}).get(role_guess, "Generic")
    band = BANDS().get(role_key) or BANDS().get("Generic")

    mn, mx, hi = int(band["market_min"]), int(band["market_max"]), int(band["high_end_max"])
    off_min, off_max = int(p["min"]), int(p["max"])
    mid = (off_min + off_max) // 2

    if mid <= mx:
        zone, score, label, theme = (
            "GREEN",
            0.10 + 0.25 * max(0, min(1, (mid - mn) / max(1, (mx - mn)))),
            "Within market range",
            "SAFE",
        )
    elif mid <= hi:
        zone, score, label, theme = (
            "YELLOW",
            0.35 + 0.35 * max(0, min(1, (mid - mx) / max(1, (hi - mx)))),
            "High-end (Competitive)",
            "HOT",
        )
    else:
        zone, score, label, theme = (
            "RED",
            0.70 + min(0.30, math.log(mid / max(1, hi) + 1, 10) / 2),
            "Too Good to be True (Scam Territory)",
            "DANGER",
        )

    score = float(max(0, min(1, score)))
    flag = bool(zone == "RED" and role_conf >= 0.60 and p["confidence"] == "HIGH")

    rs = []
    if role_conf < 0.60:
        rs.append("Role confidence is low, so salary anomaly detection is conservative.")
    if p["confidence"] != "HIGH":
        rs.append("Salary parse confidence is not high; treating anomaly cautiously.")
    rs.append(
        ("Salary looks normal" if zone == "GREEN" else "Salary looks competitive" if zone == "YELLOW" else "Salary is above typical high-end")
        + f" for {role_guess}."
    )

    return {
        "salary_parsed": True,
        "currency": "INR",
        "pay_basis": "MONTH_EQUIV",
        "offered_min": off_min,
        "offered_max": off_max,
        "offered_mid": mid,
        "market_min": mn,
        "market_max": mx,
        "high_end_max": hi,
        "zone": zone,
        "anomaly_score": round(score, 3),
        "flag": flag,
        "reasons": rs,
        "ui": {"gauge_pct": int(round(score * 100)), "label": label, "theme": theme},
    }