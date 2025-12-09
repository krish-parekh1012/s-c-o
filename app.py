
# app.py
# Sleep–Caffeine Balance Oracle — India tuned, quantitative core, onboarding, anonymized export, share card
import streamlit as st
import numpy as np
import math
from datetime import datetime, timedelta
import pandas as pd
import json
import io
from PIL import Image, ImageDraw, ImageFont
import requests
import os

# Optional LLM import (Gemini / Generative AI). If not available we gracefully fallback.
# LLM support is optional. We try to import google.generativeai at runtime only if a key is present.
HAVE_GENAI = False
genai = None

def try_init_genai():
    """
    Attempt to import google.generativeai dynamically.
    If it's not installed on the environment, return False.
    This keeps the build from failing on platforms where the package is unavailable.
    """
    global HAVE_GENAI, genai
    if HAVE_GENAI:
        return True
    try:
        import importlib
        genai = importlib.import_module("google.generativeai")
        HAVE_GENAI = True
        return True
    except Exception:
        HAVE_GENAI = False
        return False


st.set_page_config(page_title="Sleep–Caffeine Oracle (India)", page_icon="☕🇮🇳", layout="wide")
st.title("☕🔮 Sleep–Caffeine Balance Oracle — India tuned")
st.caption("Scientific PK + Two-Process sleep model + circadian rhythm + LLM enrichment (optional).")

# -----------------------------
# Beverage defaults (Indian tuned)
# -----------------------------
BEVERAGES = {
    "Instant (1 cup - Nescafé style)": {"mg": 60, "a": 2.0, "F": 0.95, "milk_slow": 0.0, "freshness_boost": -0.05},
    "South Indian Filter (cup)": {"mg": 90, "a": 1.4, "F": 0.98, "milk_slow": 0.0, "freshness_boost": 0.00},
    "Espresso (single shot ~30ml)": {"mg": 63, "a": 2.2, "F": 0.99, "milk_slow": 0.0, "freshness_boost": 0.02},
    "Cold Coffee / Milkshake (large)": {"mg": 120, "a": 1.0, "F": 0.9, "milk_slow": 0.25, "freshness_boost": 0.00},
    "Decoction / Kaapi (strong filter with chicory)": {"mg": 70, "a": 1.3, "F": 0.85, "milk_slow": 0.0, "freshness_boost": 0.00},
    "Black Tea / Masala Chai (cup)": {"mg": 40, "a": 1.2, "F": 0.9, "milk_slow": 0.0, "freshness_boost": 0.0},
    "Green Tea (cup)": {"mg": 25, "a": 1.0, "F": 0.85, "milk_slow": 0.0, "freshness_boost": 0.0},
    "Other (custom)": {"mg": 80, "a": 1.5, "F": 0.9, "milk_slow": 0.0, "freshness_boost": 0.0}
}

# -----------------------------
# Session-state containers
# -----------------------------
if "anon_records" not in st.session_state:
    st.session_state["anon_records"] = []

# -----------------------------
# ONBOARDING micro-survey (sidebar)
# -----------------------------
st.sidebar.header("Onboarding micro-survey (optional)")
st.sidebar.info("This sets priors for sensitivity & half-life to improve predictions.")
tolerance = st.sidebar.number_input("Usual caffeine tolerance (0–10)", min_value=0, max_value=10, value=5, step=1)
weekly_avg = st.sidebar.number_input("Weekly avg caffeine (mg)", min_value=0, max_value=5000, value=700)
typical_first_drink = st.sidebar.selectbox("Typical first drink", options=list(BEVERAGES.keys()))
usual_window = st.sidebar.selectbox("Usual drinking window", options=["Morning only", "All day", "Afternoon heavy", "Evening sometimes"])

def priors_to_modifiers(tolerance, weekly_avg, window):
    sens = 1.0 + (5 - tolerance) * 0.08
    if weekly_avg >= 2000:
        half_life_factor = 0.9
    elif weekly_avg >= 1000:
        half_life_factor = 0.95
    else:
        half_life_factor = 1.0
    if window == "All day":
        half_life_factor *= 1.05
    elif window == "Evening sometimes":
        sens *= 1.05
    return sens, half_life_factor

prior_sens, prior_halflife = priors_to_modifiers(tolerance, weekly_avg, usual_window)
st.sidebar.markdown(f"**Priors applied:** sensitivity ×{prior_sens:.2f}, half-life factor ×{prior_halflife:.2f}")

# Anonymized consent
st.sidebar.markdown("---")
consent = st.sidebar.checkbox("I consent to share an anonymized record of this session (no PII)", value=False)
st.sidebar.caption("If you consent, the app will prepare an anonymized record you can review and optionally upload to a webhook you control.")

# -----------------------------
# Inputs: main form
# -----------------------------
st.header("Your Sleep & Beverage Log (today)")

with st.form("main"):
    left, right = st.columns(2)
    with left:
        sleep_start = st.time_input("Sleep start time", value=datetime.now().time().replace(hour=0, minute=30))
        sleep_end = st.time_input("Sleep end time", value=datetime.now().time().replace(hour=7, minute=30))
        awakenings = st.number_input("Awakenings during sleep", min_value=0, max_value=10, value=0)
        chronotype = st.selectbox("Chronotype", ["Early (Lion)", "Normal (Bear)", "Late (Wolf)"])
    with right:
        st.markdown("**Add drinks** — type, qty, time (up to 6)")
        drinks = []
        for i in range(6):
            c1, c2, c3 = st.columns([3,1,2])
            with c1:
                bev = st.selectbox(f"Drink {i+1}", options=list(BEVERAGES.keys()), key=f"bev{i}")
            with c2:
                qty = st.number_input(f"Qty {i+1}", min_value=0, max_value=5, value=0, key=f"qty{i}")
            with c3:
                tval = st.time_input(f"Time {i+1}", value=datetime.now().time().replace(hour=8+i, minute=0), key=f"time{i}")
            if qty > 0:
                drinks.append({"type": bev, "qty": int(qty), "time": tval.strftime("%H:%M")})
        custom_notes = st.text_input("Notes (e.g., extra chicory, very strong brew)")
    submitted = st.form_submit_button("Compute prediction")

# -----------------------------
# Quantitative model functions
# -----------------------------
@st.cache_data
def make_time_grid(res_minutes=1):
    return np.arange(0, 24, res_minutes/60.0)

def dose_contribution(D, F, a, half_life, t_grid, t0):
    k = math.log(2) / half_life
    rel = t_grid - t0
    rel[rel < 0] += 24
    conc = D * F * (1 - np.exp(-a * rel)) * np.exp(-k * rel)
    return conc

def caffeine_series(drinks, t_grid, half_life=5.0, personal_half_life_factor=1.0):
    hl = half_life * personal_half_life_factor
    total = np.zeros_like(t_grid)
    for d in drinks:
        m = BEVERAGES.get(d["type"], BEVERAGES["Other (custom)"])
        D = m["mg"] * d.get("qty",1)
        F = m.get("F", 0.9)
        a = m.get("a", 1.5)
        hh, mm = map(int, d["time"].split(":"))
        t0 = hh + mm/60.0
        total += dose_contribution(D, F, a, hl, t_grid.copy(), t0)
    return total

def sleep_pressure_curve(sleep_start_h, sleep_end_h, t_grid, tau_w=18.0, tau_s=4.0):
    S = np.zeros_like(t_grid)
    awake_elapsed = 0.0
    for i, h in enumerate(t_grid):
        if sleep_start_h <= sleep_end_h:
            in_sleep = (sleep_start_h <= h < sleep_end_h)
        else:
            in_sleep = (h >= sleep_start_h or h < sleep_end_h)
        if in_sleep:
            if i == 0:
                S[i] = 0.7
            else:
                dt = t_grid[1] - t_grid[0]
                S[i] = S[i-1] * math.exp(-dt / tau_s)
        else:
            awake_elapsed += (t_grid[1] - t_grid[0])
            S[i] = 1 - math.exp(-awake_elapsed / tau_w)
    return S

def circadian_curve(t_grid, chronotype="Normal"):
    if chronotype.startswith("Early"):
        phi = 4.0
    elif chronotype.startswith("Late"):
        phi = 8.0
    else:
        phi = 6.0
    return np.cos(2 * np.pi * (t_grid - phi) / 24.0)

def performance_curve(t_grid, caffeine_vals, sleep_start_h, sleep_end_h, chronotype="Normal",
                      sens=1.0):
    S = sleep_pressure_curve(sleep_start_h, sleep_end_h, t_grid)
    C = circadian_curve(t_grid, chronotype)
    P = 100 + 30 * C - 50 * S + 0.15 * caffeine_vals * sens
    return P, S, C

def find_peak_crash(P, t_grid):
    peak_idx = int(np.argmax(P))
    if peak_idx < len(P)-2:
        crash_idx = int(np.argmin(P[peak_idx:])) + peak_idx
    else:
        crash_idx = peak_idx
    return t_grid[peak_idx], t_grid[crash_idx], peak_idx, crash_idx

# -----------------------------
# Personal modifiers & calibration
# -----------------------------
def personal_modifiers(age, weight, smoker, pregnant, prior_sens, prior_hf):
    sens = 1.0 * prior_sens
    hf = 1.0 * prior_hf
    if age > 60:
        hf *= 1.2
    if smoker == "Yes":
        hf *= 0.8
    if pregnant == "Yes":
        hf *= 1.5
        sens *= 1.2
    if weight < 60:
        sens *= 1.1
    elif weight > 90:
        sens *= 0.95
    return sens, hf

# Simple calibration routine (grid search)
def calibrate_from_jitter(drink, reported_jitter, t_grid, prior_s=1.0, prior_hf=1.0):
    s_vals = np.linspace(0.6, 2.0, 30)
    hf_vals = np.linspace(0.7, 1.5, 30)
    best = None
    best_err = float("inf")
    # small beta mapping from peak->jitter (empirical init, can be tuned)
    beta = 0.04
    for s in s_vals:
        for hf in hf_vals:
            cs = caffeine_series([drink], t_grid, half_life=5.0, personal_half_life_factor=hf)
            peak = np.max(cs) * s
            pred = beta * peak
            err = (reported_jitter - pred)**2
            if err < best_err:
                best_err = err
                best = (s, hf, peak, pred)
    return {"s": best[0], "half_life_factor": best[1], "predicted_jitter": best[3], "peak": best[2], "err": best_err}

# -----------------------------
# Creature seeds (deterministic mapping)
# -----------------------------
CREATURE_SEED = [
    {"id":"kapi_gremlin", "label":"Kapi Gremlin", "emoji":"🐀", "desc":"Fast, unpredictable; strong filter fan."},
    {"id":"spice_dragon", "label":"Spice Dragon", "emoji":"🐉", "desc":"Warm, fiery buzz — masala + coffee combos."},
    {"id":"monsoon_sloth", "label":"Monsoon Sloth", "emoji":"🐢", "desc":"Slow but steady energy; small sips win."},
    {"id":"chaai_sprite", "label":"Chaai Sprite", "emoji":"🫖", "desc":"Tea-first focus; light and consistent."},
    {"id":"turbo_raccoon", "label":"Turbo Raccoon", "emoji":"🦝", "desc":"Chaotic late-day energy, noisy and fast."},
    {"id":"zen_griffin", "label":"Zen Griffin", "emoji":"🦅", "desc":"Balanced: good sleep, moderate intake."}
]

def seed_creature_key(peak_val, chaos_score, caffeine_total):
    key = int((peak_val * 3 + chaos_score * 2 + caffeine_total/50)) % len(CREATURE_SEED)
    return CREATURE_SEED[key]

# -----------------------------
# Main compute & UI (on submit)
# -----------------------------
if submitted:
    # time grid (1-minute)
    t_grid = make_time_grid(res_minutes=1)
    # sleep hours -> numeric hour values
    dt1 = datetime.combine(datetime.today(), sleep_start)
    dt2 = datetime.combine(datetime.today(), sleep_end)
    if dt2 <= dt1:
        dt2 += timedelta(days=1)
    sleep_hours = (dt2 - dt1).seconds / 3600.0
    sleep_debt = max(0, 8.0 - sleep_hours)
    # gather drinks
    total_caffeine = 0
    for d in drinks:
        meta = BEVERAGES.get(d["type"], BEVERAGES["Other (custom)"])
        total_caffeine += meta.get("mg",80) * d["qty"]
    # demographics inputs (simple optional block)
    with st.expander("Optional demographics (helps priors)", expanded=False):
        age = st.number_input("Age (years)", min_value=12, max_value=100, value=28)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        smoker = st.selectbox("Smoker?", ["No", "Yes"])
        pregnant = st.selectbox("Pregnant or breastfeeding?", ["No", "Yes (use conservative defaults)"])
    # combine priors and demographics
    sens_prior, half_prior = prior_sens, prior_halflife
    sens, half_life_factor = personal_modifiers(age, weight, smoker, pregnant, sens_prior, half_prior)
    # compute caffeine series
    Cc = caffeine_series(drinks, t_grid, half_life=5.0, personal_half_life_factor=half_life_factor)
    # performance
    sleep_start_hour = dt1.hour + dt1.minute/60.0
    sleep_end_hour = dt2.hour + dt2.minute/60.0
    P, S, C = performance_curve(t_grid, Cc, sleep_start_hour, sleep_end_hour, chronotype=chronotype, sens=sens)
    peak_t, crash_t, peak_idx, crash_idx = find_peak_crash(P, t_grid)
    # round times to HH:MM
    def fmt_hour(h):
        hh = int(h) % 24
        mm = int((h - int(h))*60)
        return f"{hh:02d}:{mm:02d}"
    peak_time = fmt_hour(peak_t)
    crash_time = fmt_hour(crash_t)
    # chaos score
    circ_phase = 4 if chronotype.startswith("Early") else (8 if chronotype.startswith("Late") else 6)
    circ_misalignment = abs(circ_phase - (sleep_start.hour if hasattr(sleep_start, "hour") else sleep_start.hour))
    chaos_score = int(min(100, max(0, (total_caffeine/10) + sleep_debt*8 + circ_misalignment*3)))
    creature_seed = seed_creature_key(P[peak_idx]/100.0, chaos_score, total_caffeine)
    masala_meter = int(min(100, (total_caffeine/5.0) + (chaos_score/2)))
    brew_respect = int(100 - min(80, abs(BEVERAGES.get(drinks[0]["type"], {"mg":60})["mg"] - 80) if drinks else 20))
    # show charts & key numbers
    st.subheader("Scientific outputs")
    df_perf = pd.DataFrame({"minute": t_grid, "Performance": P, "Caffeine": Cc})
    st.line_chart(df_perf.set_index("minute")[["Performance"]])
    st.line_chart(df_perf.set_index("minute")[["Caffeine"]])
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak (time)", peak_time)
    c2.metric("Predicted crash", crash_time)
    c3.metric("Total caffeine (est mg)", int(total_caffeine))
    st.markdown(f"**Sleep hours:** {sleep_hours:.2f}  — Sleep debt: {sleep_debt:.2f} hrs")
    st.markdown(f"**Chaos Score:** {chaos_score}  — Masala Meter: {masala_meter}  — Brew Respect: {brew_respect}")
    # LLM enrichment (optional)
    st.subheader("🔮 Oracle interpretation (AI-enhanced)")
    prompt_payload = {
        "meta": {
            "sleep_hours": round(sleep_hours,2),
            "sleep_debt": round(sleep_debt,2),
            "total_caffeine_mg": int(total_caffeine),
            "peak_time": peak_time,
            "crash_time": crash_time,
            "chronotype": chronotype,
            "chaos_score": chaos_score,
            "masala_meter": masala_meter
        },
        "seed_creature": creature_seed,
        "instructions": "Return JSON only with keys: creature, creature_bio, oracle_text, rituals (list of 3), tips (list of 3). Tone: playful, India-savvy, avoid medical advice."
    }
ai_result = None

# Only attempt AI enrichment if user provided a GEMINI_API_KEY
if "GEMINI_API_KEY" in st.secrets:
    # Try to import the library at runtime (avoids build-time pip issues)
    if not try_init_genai():
        st.warning("AI library not installed in this environment; AI enrichment disabled. To enable, install 'google-generative-ai' or deploy a small proxy that calls the LLM.")
    else:
        try:
            # Configure and call the model
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            sys_prompt = (
                "You are the Sleep–Caffeine Oracle. Use the structured payload and produce a short India-aware creature summary "
                "and 3 practical rituals. Return strict JSON."
            )
            resp = model.generate_content(sys_prompt + "\n\n" + json.dumps(prompt_payload))
            raw = resp.text.strip()

            # Robust JSON extraction: try direct parse, then attempt to extract the first JSON object substring
            try:
                ai_result = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    ai_result = json.loads(raw[start:end+1])
                else:
                    # couldn't parse JSON out of model response
                    st.warning("AI returned an unexpected response format; using deterministic fallback.")
                    ai_result = None

        except Exception as e:
            st.warning(f"AI enrichment attempted and failed: {str(e)} — showing deterministic fallback.")
            ai_result = None

# Fallback deterministic response (unchanged)
if not ai_result:
    ai_result = {
        "creature": f"{creature_seed['label']} {creature_seed.get('emoji','')}",
        "creature_bio": creature_seed["desc"],
        "oracle_text": f"Peak at {peak_time}. Crash at {crash_time}. Chaos Score {chaos_score}.",
        "rituals": [
            "Drink 250–300 ml water after your next coffee.",
            "Try a 15–20 minute low-light nap before the predicted crash.",
            "Switch to tea or decaf after midday if chaos > 40."
        ],
        "tips": [
            "Avoid mixing strong filter + cold coffee in the same afternoon.",
            "Space chai and coffee by 3+ hours if possible.",
            "Keep a small protein snack & water ready to blunt crashes."
        ]
    }

    # Prepare anonymized record (no PII)
    anon_record = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "sleep_hours": round(sleep_hours,2),
        "sleep_debt": round(sleep_debt,2),
        "total_caffeine_mg": int(total_caffeine),
        "chronotype": chronotype,
        "chaos_score": chaos_score,
        "masala_meter": masala_meter,
        "brew_respect": brew_respect,
        "creature_id": creature_seed["id"],
        "priors": {"tolerance": tolerance, "weekly_avg": weekly_avg, "window": usual_window}
    }
    st.subheader("Share & Save (anonymized)")
    if consent:
        st.success("You opted in to anonymized data for this session.")
        st.session_state["anon_records"].append(anon_record)
        st.download_button("Download this anonymized record (JSON)", data=json.dumps(anon_record, indent=2), file_name="anon_record.json", mime="application/json")
        if st.session_state["anon_records"]:
            df_anon = pd.DataFrame(st.session_state["anon_records"])
            st.download_button("Download all session records (CSV)", data=df_anon.to_csv(index=False).encode("utf-8"), file_name="anon_records.csv", mime="text/csv")
        webhook_url = st.secrets.get("ANON_WEBHOOK") if "ANON_WEBHOOK" in st.secrets else st.text_input("Optional: paste webhook URL to auto-send anonymized record", value="")
        if st.button("Send anonymized record to webhook"):
            target = webhook_url.strip()
            if not target:
                st.error("Provide webhook URL or set ANON_WEBHOOK in Streamlit Secrets.")
            else:
                try:
                    r = requests.post(target, json=anon_record, timeout=10)
                    if r.status_code in (200,201,202):
                        st.success("Anonymized record posted.")
                    else:
                        st.warning(f"Webhook returned {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Webhook post failed: {e}")
    else:
        st.info("You did not consent to anonymized sharing. You can still download a local anonymized record.")
        st.download_button("Download anonymized record locally (JSON)", data=json.dumps(anon_record, indent=2), file_name="anon_record.json", mime="application/json")
    # Calibration flow
    st.subheader("⚖️ Personal calibration (optional)")
    with st.expander("Calibrate using a recent known drink + jitter rating"):
        c_drink = st.selectbox("Which drink did you have recently?", options=list(BEVERAGES.keys()))
        c_amount = st.number_input("Servings", min_value=1, max_value=5, value=1)
        c_time = st.time_input("Time drunk", value=datetime.now().time().replace(hour=9, minute=0), key="cal_time")
        jitter = st.slider("Self-reported jitter (0–10)", 0, 10, 3)
        if st.button("Run calibration"):
            tgrid = make_time_grid(res_minutes=1)
            drink_obj = {"type": c_drink, "qty": int(c_amount), "time": c_time.strftime("%H:%M")}
            cal = calibrate_from_jitter(drink_obj, float(jitter), tgrid, prior_sens, prior_halflife)
            st.success(f"Calibration result: sensitivity s={cal['s']:.3f}, half-life factor={cal['half_life_factor']:.3f}")
            st.info("Calibration is session-local. Persist to DB/webhook if you want long-term personalization.")
    # Shareable card generation
    st.subheader("📸 Shareable card (download)")
    card_w, card_h = 1200, 630
    bg = (255,250,240)
    card = Image.new("RGB", (card_w, card_h), color=bg)
    draw = ImageDraw.Draw(card)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
        font_med = ImageFont.truetype("DejaVuSans.ttf", 30)
    except Exception:
        font_big = ImageFont.load_default()
        font_med = ImageFont.load_default()
    title = "Sleep–Caffeine Oracle"
    creature_line = ai_result.get("creature", creature_seed["label"])
    prophecy = ai_result.get("oracle_text", "")[:140]
    stats = f"Peak: {peak_time}  •  Crash: {crash_time}  •  Chaos: {chaos_score}"
    draw.text((60,40), title, fill=(30,30,30), font=font_big)
    draw.text((60,140), creature_line, fill=(20,20,60), font=font_med)
    draw.text((60,200), prophecy, fill=(60,30,30), font=font_med)
    draw.text((60,260), stats, fill=(40,40,40), font=font_med)
    draw.text((60, card_h-60), "Built with ☕ + science. Data anonymized on consent.", fill=(100,100,100), font=font_med)
    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    st.image(card, use_column_width=True)
    st.download_button("Download shareable card (PNG)", data=buf, file_name="oracle_card.png", mime="image/png")
    st.markdown("---")
    st.caption("This data is informative and not medical advice. If posting anonymized data to a webhook, ensure compliance with local privacy laws.")

# End of app.py
