# app.py
# Sleep–Caffeine Balance Oracle — India-tuned, deterministic & scientific (no AI)
# Streamlit app: quantitative PK + Two-Process sleep model + circadian + Monte Carlo uncertainty
import streamlit as st
import numpy as np
import math
from datetime import datetime, timedelta
import pandas as pd
import json
import io
from PIL import Image, ImageDraw, ImageFont
import requests
import random

st.set_page_config(page_title="Sleep–Caffeine Oracle", page_icon="☕", layout="wide")
st.title("☕")
st.caption("Calibrate & share your result.")

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

# Creature seeds (deterministic mapping to metrics)
CREATURE_SEED = [
    {"id":"kapi_gremlin", "label":"Kapi Gremlin", "emoji":"🐀", "desc":"Turbo spikes from strong filter coffee."},
    {"id":"spice_dragon", "label":"Spice Dragon", "emoji":"🐉", "desc":"Warm, long buzz from masala combos."},
    {"id":"monsoon_sloth", "label":"Monsoon Sloth", "emoji":"🐢", "desc":"Slow & steady — small frequent sips."},
    {"id":"chaai_sprite", "label":"Chai Sprite", "emoji":"🫖", "desc":"Gentle lift from tea — steady focus."},
    {"id":"turbo_raccoon", "label":"Turbo Raccoon", "emoji":"🦝", "desc":"Chaotic late-day energy bursts."},
    {"id":"zen_griffin", "label":"Zen Griffin", "emoji":"🦅", "desc":"Balanced — good sleep & moderate intake."}
]

# -----------------------------
# Session storage for anonymized records
# -----------------------------
if "anon_records" not in st.session_state:
    st.session_state["anon_records"] = []

# -----------------------------
# Sidebar: onboarding priors & consent
# -----------------------------
st.sidebar.header("Onboarding micro-survey (optional)")
st.sidebar.info("These priors tune sensitivity & half-life to your habits.")
tolerance = st.sidebar.number_input("Usual caffeine tolerance (0–10)", 0, 10, 5)
weekly_avg = st.sidebar.number_input("Weekly avg caffeine (mg)", 0, 5000, 700)
usual_window = st.sidebar.selectbox("Usual drinking window", ["Morning only", "All day", "Afternoon heavy", "Evening sometimes"])

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
st.sidebar.markdown(f"**Priors:** sensitivity ×{prior_sens:.2f}, half-life factor ×{prior_halflife:.2f}")

st.sidebar.markdown("---")
consent = st.sidebar.checkbox("I consent to share an anonymized record of this session (no PII)", value=False)
st.sidebar.caption("If you consent, you can download/export an anonymized record or post it to your webhook.")

# -----------------------------
# Main input form
# -----------------------------
st.header("Enter your sleep & drinks for the day")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        sleep_start = st.time_input("Sleep start", value=datetime.now().time().replace(hour=0, minute=30))
        sleep_end = st.time_input("Sleep end", value=datetime.now().time().replace(hour=7, minute=30))
        awakenings = st.number_input("Awakenings during sleep", 0, 10, 0)
        chronotype = st.selectbox("Chronotype", ["Early (Lion)", "Normal (Bear)", "Late (Wolf)"])
    with col2:
        st.markdown("Add drinks (type, qty, time). Up to 6 entries.")
        drinks = []
        for i in range(6):
            b1, b2, b3 = st.columns([3,1,2])
            with b1:
                bev = st.selectbox(f"Drink {i+1}", options=list(BEVERAGES.keys()), key=f"bev_{i}")
            with b2:
                qty = st.number_input(f"Qty {i+1}", min_value=0, max_value=5, value=0, key=f"qty_{i}")
            with b3:
                tval = st.time_input(f"Time {i+1}", value=datetime.now().time().replace(hour=8+i, minute=0), key=f"time_{i}")
            if qty > 0:
                drinks.append({"type": bev, "qty": int(qty), "time": tval.strftime("%H:%M")})
        notes = st.text_input("Notes (optional)")
    submitted = st.form_submit_button("Compute prediction")

# -----------------------------
# Core quantitative functions (deterministic)
# -----------------------------
@st.cache_data
def make_time_grid(res_minutes=5):
    # 5-minute resolution default for speed (change to 1 for finer)
    return np.arange(0, 24, res_minutes / 60.0)

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
        meta = BEVERAGES.get(d["type"], BEVERAGES["Other (custom)"])
        D = meta.get("mg", 80) * d.get("qty", 1)
        F = meta.get("F", 0.9)
        a = meta.get("a", 1.5)
        hh, mm = map(int, d["time"].split(":"))
        t0 = hh + mm / 60.0
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
                S[i] = S[i - 1] * math.exp(-dt / tau_s)
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

def performance_curve(t_grid, caffeine_vals, sleep_start_h, sleep_end_h, chronotype="Normal", sens=1.0):
    S = sleep_pressure_curve(sleep_start_h, sleep_end_h, t_grid)
    C = circadian_curve(t_grid, chronotype)
    P = 100 + 30 * C - 50 * S + 0.15 * caffeine_vals * sens
    return P, S, C

def find_peak_crash(P, t_grid):
    peak_idx = int(np.argmax(P))
    if peak_idx < len(P) - 2:
        crash_idx = int(np.argmin(P[peak_idx:])) + peak_idx
    else:
        crash_idx = peak_idx
    return t_grid[peak_idx], t_grid[crash_idx], peak_idx, crash_idx

# -----------------------------
# Calibration routine (self-report jitter)
# -----------------------------
def calibrate_from_jitter(drink, reported_jitter, t_grid, prior_s=1.0, prior_hf=1.0):
    s_vals = np.linspace(0.6, 2.0, 25)
    hf_vals = np.linspace(0.7, 1.4, 25)
    best = None
    best_err = float("inf")
    beta = 0.04
    for s in s_vals:
        for hf in hf_vals:
            cs = caffeine_series([drink], t_grid, half_life=5.0, personal_half_life_factor=hf)
            peak = np.max(cs) * s
            pred = beta * peak
            err = (reported_jitter - pred) ** 2
            if err < best_err:
                best_err = err
                best = (s, hf, peak, pred)
    return {"s": best[0], "half_life_factor": best[1], "predicted_jitter": best[3], "peak": best[2], "err": best_err}

# -----------------------------
# Deterministic creature mapping & human-friendly rules
# -----------------------------
def seed_creature_key(peak_val, chaos_score, total_caffeine):
    key = int((peak_val * 3 + chaos_score * 2 + total_caffeine / 50)) % len(CREATURE_SEED)
    return CREATURE_SEED[key]

def creature_explanations(creature):
    # turn seed into friendly cards (no AI)
    return {
        "title": f"{creature['label']} {creature.get('emoji','')}",
        "bio": creature.get("desc", ""),
        "personality": {
            "kapi_gremlin": "You spike fast. Avoid late espresso before meetings.",
            "spice_dragon": "Warm long tail — careful with chai+coffee stacks.",
            "monsoon_sloth": "Steady energy. Hydrate and small snacks help.",
            "chaai_sprite": "Tea-first. Gentle focus, but watch doubling with coffee.",
            "turbo_raccoon": "Erratic late bursts. Avoid late heavy drinks.",
            "zen_griffin": "Balanced rhythms. Keep current routine."
        }.get(creature["id"], "A curious caffeine creature.")
    }

# -----------------------------
# Monte Carlo wrapper for uncertainty (samples beverage mg ±10% and half-life variance)
# -----------------------------
def monte_carlo_performance(drinks, t_grid, sens, half_life_factor, n_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    peak_times = []
    crash_times = []
    perf_samples = []
    for _ in range(n_samples):
        # perturb beverage mg by ±10% and half-life from Normal(1,0.1)
        pert_drinks = []
        for d in drinks:
            meta = BEVERAGES.get(d["type"], BEVERAGES["Other (custom)"])
            base = meta.get("mg", 80)
            pert = float(base) * rng.normal(1.0, 0.06)  # small noise
            pert_drinks.append({"type": d["type"], "qty": d["qty"], "time": d["time"], "_pert_mg": pert})
        hl_factor = half_life_factor * max(0.7, rng.normal(1.0, 0.08))
        # build caffeine series using perturbed mg via a quick inlined calc
        Cc = np.zeros_like(t_grid)
        for d in pert_drinks:
            meta = BEVERAGES.get(d["type"], BEVERAGES["Other (custom)"])
            D = d["_pert_mg"] * d.get("qty", 1)
            F = meta.get("F", 0.9)
            a = meta.get("a", 1.5)
            hh, mm = map(int, d["time"].split(":"))
            t0 = hh + mm / 60.0
            Cc += dose_contribution(D, F, a, 5.0 * hl_factor, t_grid.copy(), t0)
        P, _, _ = performance_curve(t_grid, Cc, sleep_start_hour, sleep_end_hour, chronotype=chronotype, sens=sens)
        pt, ct, p_i, c_i = find_peak_crash(P, t_grid)
        peak_times.append(pt)
        crash_times.append(ct)
        perf_samples.append(P)
    # compute median curves & percentiles
    perf_arr = np.array(perf_samples)  # shape (n_samples, len(t_grid))
    perf_median = np.median(perf_arr, axis=0)
    perf_low = np.percentile(perf_arr, 5, axis=0)
    perf_high = np.percentile(perf_arr, 95, axis=0)
    return {
        "peak_times": peak_times,
        "crash_times": crash_times,
        "perf_median": perf_median,
        "perf_low": perf_low,
        "perf_high": perf_high
    }

# -----------------------------
# MAIN: on submit compute everything deterministically
# -----------------------------
if submitted:
    # time grid
    t_grid = make_time_grid(res_minutes=5)
    # compute sleep hours
    dt1 = datetime.combine(datetime.today(), sleep_start)
    dt2 = datetime.combine(datetime.today(), sleep_end)
    if dt2 <= dt1:
        dt2 += timedelta(days=1)
    sleep_hours = (dt2 - dt1).seconds / 3600.0
    sleep_debt = max(0, 8.0 - sleep_hours)
    # total caffeine estimate
    total_caffeine = 0
    for d in drinks:
        meta = BEVERAGES.get(d["type"], BEVERAGES["Other (custom)"])
        total_caffeine += meta.get("mg", 80) * d["qty"]
    # demographics and personal modifiers (optional block)
    with st.expander("Optional demographics (improves priors)"):
        age = st.number_input("Age (years)", 12, 100, 28)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        smoker = st.selectbox("Smoker?", ["No", "Yes"])
        pregnant = st.selectbox("Pregnant/breastfeeding?", ["No", "Yes (conservative defaults)"])
    # combine priors and demographics
    def personal_modifiers_demo(age, weight, smoker, pregnant, prior_sens, prior_hf):
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

    sens, half_life_factor = personal_modifiers_demo(age, weight, smoker, pregnant, prior_sens, prior_halflife)

    # compute base caffeine curve & performance
    # variables used by Monte Carlo are needed below; compute numeric hours for sleep window
    sleep_start_hour = dt1.hour + dt1.minute / 60.0
    sleep_end_hour = dt2.hour + dt2.minute / 60.0

    Cc = caffeine_series(drinks, t_grid, half_life=5.0, personal_half_life_factor=half_life_factor)
    P, S, C = performance_curve(t_grid, Cc, sleep_start_hour, sleep_end_hour, chronotype=chronotype, sens=sens)
    peak_t, crash_t, peak_idx, crash_idx = find_peak_crash(P, t_grid)
    def fmt_hour(h):
        hh = int(h) % 24
        mm = int((h - int(h)) * 60)
        return f"{hh:02d}:{mm:02d}"
    peak_time = fmt_hour(peak_t)
    crash_time = fmt_hour(crash_t)

    # chaos score (simple deterministic rule)
    circ_phase = 4 if chronotype.startswith("Early") else (8 if chronotype.startswith("Late") else 6)
    circ_misalignment = abs(circ_phase - sleep_start_hour)
    chaos_score = int(min(100, max(0, (total_caffeine / 10) + sleep_debt * 8 + circ_misalignment * 3)))

    # creature mapping
    creature_seed = seed_creature_key(P[peak_idx] / 100.0, chaos_score, total_caffeine)
    creature_card = creature_explanations(creature_seed)

    # masala meter & brew respect
    masala_meter = int(min(100, (total_caffeine / 5.0) + (chaos_score / 2)))
    brew_respect = int(100 - min(80, abs(BEVERAGES.get(drinks[0]["type"], {"mg": 60})["mg"] - 80) if drinks else 20))

    # Monte Carlo uncertainty (fast, n=120 by default for speed)
    mc = monte_carlo_performance(drinks, t_grid, sens, half_life_factor, n_samples=120, seed=random.randint(1,10000))

    # Present results
    st.subheader("Scientific outputs")
    perf_df = pd.DataFrame({"time": t_grid, "performance": P, "caffeine": Cc})
    st.line_chart(perf_df.set_index("time")[["performance"]])
    st.line_chart(perf_df.set_index("time")[["caffeine"]])

    left, mid, right = st.columns(3)
    left.metric("Peak", peak_time)
    mid.metric("Predicted crash", crash_time)
    right.metric("Total caffeine (est mg)", int(total_caffeine))

    st.markdown(f"**Sleep hours:** {sleep_hours:.2f} — Sleep debt: {sleep_debt:.2f} hrs")
    st.markdown(f"**Chaos Score:** {chaos_score} — Masala Meter: {masala_meter} — Brew Respect: {brew_respect}")

    # show MC median and CI for performance (small chart)
    st.subheader("Uncertainty (Monte Carlo)")
    mc_df = pd.DataFrame({
        "time": t_grid,
        "median": mc["perf_median"],
        "low": mc["perf_low"],
        "high": mc["perf_high"]
    })
    st.line_chart(mc_df.set_index("time")[["median", "low", "high"]])

    # Creature card (deterministic, fun)
    st.subheader("Your Creature & Interpretation")
    st.markdown(f"### {creature_card['title']}")
    st.markdown(f"**Bio:** {creature_card['bio']}")
    st.markdown(f"**Quick take:** {creature_card['personality']}")

    # Practical rituals derived deterministically from numbers
    rituals = []
    rituals.append("Drink 250–300 ml water within 10 minutes of your next beverage.")
    if chaos_score > 50:
        rituals.append("Avoid caffeine after 2 PM today; use decaf or tea instead.")
    else:
        rituals.append("If you feel a dip, try a 15–20 minute restorative nap before the predicted crash.")
    rituals.append("Eat a small protein snack (e.g., nuts or egg) to stabilise post-lunch crash.")
    st.write("**Rituals:**")
    for r in rituals:
        st.write("•", r)

    # deterministic tips (India-tuned)
    st.write("**Practical tips:**")
    tips = [
        "Filter + cold coffee same afternoon increases chaos — space them by 3+ hours.",
        "If you had chai + instant coffee, offset with water & protein.",
        "Small sips across the day (Monsoon Sloth style) reduce big crashes."
    ]
    for t in tips:
        st.write("•", t)

    # ------------------------------
    # Anonymized record & consented export
    # ------------------------------
    anon_record = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "sleep_hours": round(sleep_hours, 2),
        "sleep_debt": round(sleep_debt, 2),
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
        st.success("You opted-in to anonymized sharing for this session.")
        st.session_state["anon_records"].append(anon_record)
        st.download_button("Download this anonymized record (JSON)", data=json.dumps(anon_record, indent=2), file_name="anon_record.json", mime="application/json")
        if st.session_state["anon_records"]:
            df_anon = pd.DataFrame(st.session_state["anon_records"])
            st.download_button("Download all session records (CSV)", data=df_anon.to_csv(index=False).encode("utf-8"), file_name="anon_records.csv", mime="text/csv")
        webhook_url = st.text_input("Optional: webhook URL to POST anonymized record (enter and click send)", value="")
        if st.button("Send anonymized record to webhook"):
            target = webhook_url.strip()
            if not target:
                st.error("Provide a webhook URL to send anonymized record.")
            else:
                try:
                    r = requests.post(target, json=anon_record, timeout=10)
                    if r.status_code in (200,201,202):
                        st.success("Anonymized record posted successfully.")
                    else:
                        st.warning(f"Webhook returned {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Webhook call failed: {e}")
    else:
        st.info("You did not consent to anonymized sharing. Download a local anonymized record if you want to share manually.")
        st.download_button("Download anonymized record (JSON)", data=json.dumps(anon_record, indent=2), file_name="anon_record.json", mime="application/json")

    # ------------------------------
    # Personal calibration (optional)
    # ------------------------------
    st.subheader("Calibration (optional): tune to your sensitivity")
    with st.expander("Calibrate from a known drink + jitter rating"):
        c_drink = st.selectbox("Which drink?", options=list(BEVERAGES.keys()))
        c_qty = st.number_input("Servings", min_value=1, max_value=5, value=1)
        c_time = st.time_input("Time you drank it", value=datetime.now().time().replace(hour=9, minute=0), key="cal_time")
        jitter = st.slider("Self-reported jitter (0–10)", 0, 10, 3)
        if st.button("Run calibration"):
            tgrid_cal = make_time_grid(res_minutes=5)
            drink_obj = {"type": c_drink, "qty": int(c_qty), "time": c_time.strftime("%H:%M")}
            cal = calibrate_from_jitter(drink_obj, float(jitter), tgrid_cal, prior_sens, prior_halflife)
            st.success(f"Calibration suggests sensitivity s={cal['s']:.3f}, half-life factor={cal['half_life_factor']:.3f}")
            st.info("Calibration is session-only unless you store it externally.")

    # ------------------------------
    # Shareable PNG card (deterministic)
    # ------------------------------
    st.subheader("📸 Shareable card")
    card_w, card_h = 1200, 630
    bg = (255, 250, 240)
    card = Image.new("RGB", (card_w, card_h), color=bg)
    draw = ImageDraw.Draw(card)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
        font_med = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font_big = ImageFont.load_default()
        font_med = ImageFont.load_default()
    title = "Sleep–Caffeine Oracle"
    creature_line = f"{creature_card['title']}"
    prophecy = f"{creature_card['bio']} Peak {peak_time}. Crash {crash_time}."
    stats = f"Chaos {chaos_score}  •  Masala {masala_meter}  •  Caffeine {int(total_caffeine)}mg"
    draw.text((60, 40), title, fill=(30,30,30), font=font_big)
    draw.text((60, 140), creature_line, fill=(20,20,60), font=font_med)
    draw.text((60, 200), prophecy, fill=(60,30,30), font=font_med)
    draw.text((60, 260), stats, fill=(40,40,40), font=font_med)
    draw.text((60, card_h-60), "Built with coffee, math & empathy. Data anonymized on consent.", fill=(100,100,100), font=font_med)
    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    st.image(card, use_column_width=True)
    st.download_button("Download shareable card (PNG)", data=buf, file_name="oracle_card.png", mime="image/png")

    st.markdown("---")
    st.caption("This app is informational and not medical advice. Use responsibly.")
