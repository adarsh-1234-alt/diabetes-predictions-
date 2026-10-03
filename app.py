import json, hmac, itertools, datetime as dt
import joblib, numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
HELP = {
    "Pregnancies": "Number of times pregnant",
    "Glucose": "Plasma glucose, 2 h oral glucose tolerance test (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin-fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (kg/m²)",
    "DiabetesPedigreeFunction": "Family-history score (higher = stronger history)",
    "Age": "Age in years"}

# ================= Authentication (with lockout) =================
def login():
    if st.session_state.get("auth"):
        return True
    st.title("🩺 Diabetes Risk Predictor")
    tries = st.session_state.get("tries", 0)
    if tries >= 5:
        st.error("Too many failed attempts. Refresh the page to try again."); return False
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        try:
            gu, gp = st.secrets["auth"]["username"], st.secrets["auth"]["password"]
        except Exception:
            gu, gp = "admin", "smit123"
        if hmac.compare_digest(u, gu) and hmac.compare_digest(p, gp):
            st.session_state["auth"] = True; st.rerun()
        st.session_state["tries"] = tries + 1
        st.error(f"Invalid credentials ({4 - tries} attempts left)")
    return False

if not login():
    st.stop()

# ================= Resources =================
@st.cache_resource
def load():
    try:
        bundle = joblib.load("model.joblib")
    except Exception:  # library version mismatch: retrain once on the server
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)
        bundle = joblib.load("model.joblib")
    meta = json.load(open("metrics.json"))
    bg = pd.read_csv("background.csv")
    final = bundle["final"]
    f = lambda X: final.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]
    explainer = shap.Explainer(f, bg, feature_names=FEATURES)
    return final, bundle["members"], meta, bg, explainer

@st.cache_data
def load_data():
    d = pd.read_csv("diabetes.csv")
    d[ZERO_AS_MISSING] = d[ZERO_AS_MISSING].replace(0, np.nan)
    return d

@st.cache_resource(max_entries=64)
def explain_one(vals_tuple):
    return explainer(pd.DataFrame([vals_tuple], columns=FEATURES))

@st.cache_resource
def global_importance():
    sv = explainer(bg)
    return pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values()

final, members, meta, bg, explainer = load()
data = load_data()
rec_th = meta["recommended_threshold"]

# ================= Sidebar inputs =================
with st.sidebar:
    st.header("Patient inputs")
    vals = {"Pregnancies": st.number_input("Pregnancies", 0, 20, 1, help=HELP["Pregnancies"]),
            "Glucose": st.slider("Glucose", 40, 250, 120, help=HELP["Glucose"]),
            "BloodPressure": st.slider("Blood pressure", 30, 130, 70, help=HELP["BloodPressure"])}
    for name, lo, hi, dflt in [("SkinThickness", 5, 100, 25), ("Insulin", 10, 850, 80)]:
        unk = st.checkbox(f"{name} unknown", key=f"u_{name}", help="The model fills in missing values.")
        v = st.slider(name, lo, hi, dflt, help=HELP[name], disabled=unk)
        vals[name] = np.nan if unk else v
    vals["BMI"] = st.slider("BMI", 15.0, 70.0, 28.0, 0.1, help=HELP["BMI"])
    vals["DiabetesPedigreeFunction"] = st.slider("Pedigree function", 0.05, 2.5, 0.4, 0.01,
                                                 help=HELP["DiabetesPedigreeFunction"])
    vals["Age"] = st.slider("Age", 18, 90, 35, help=HELP["Age"])
    st.divider()
    thr = st.slider("Decision threshold", 0.05, 0.90, float(rec_th), 0.01,
                    help="Lower = catches more cases but raises false alarms. Default is tuned for screening.")
    if st.button("Sign out"):
        st.session_state.clear(); st.rerun()

x = pd.DataFrame([vals])[FEATURES]
risk = float(final.predict_proba(x)[0, 1])
flagged = risk >= thr
level = "High" if risk >= max(thr, .5) else "Elevated" if flagged else "Low"

st.title("🩺 Diabetes Risk Predictor")
st.caption(f"Model: **{meta['best_model']}** · Calibrated probabilities · Educational tool, not a medical diagnosis.")
tabs = st.tabs(["🔮 Prediction", "🧠 Explainability", "🎯 Risk-reduction plan",
                "📁 Batch", "📊 Performance", "🔎 Data explorer"])

# ================= 1. Prediction =================
with tabs[0]:
    c1, c2, c3 = st.columns([1, 1.3, 1.3])
    c1.metric("Estimated risk", f"{risk:.1%}", f"{level} (threshold {thr:.0%})", delta_color="off")
    c1.progress(min(risk, 1.0))
    c1.caption("Population average in training data: "
               f"{np.mean(meta['test_y']):.0%}")
    msg = {"High": c2.error, "Elevated": c2.warning, "Low": c2.success}[level]
    msg({"High": "High estimated risk. Please see a healthcare professional for testing.",
         "Elevated": "Above the screening threshold. Consider a confirmatory HbA1c / fasting glucose test.",
         "Low": "Below the screening threshold for the values entered."}[level])
    flags = []
    if vals["Glucose"] >= 140: flags.append("Glucose is elevated (≥140 mg/dL after OGTT)")
    if vals["BMI"] >= 30: flags.append("BMI is in the obese range (≥30)")
    elif vals["BMI"] >= 25: flags.append("BMI is in the overweight range (25–30)")
    if vals["BloodPressure"] >= 90: flags.append("Diastolic BP is high (≥90 mm Hg)")
    if flags: c2.markdown("**Notable inputs**\n" + "\n".join(f"- {f}" for f in flags))
    # model agreement
    mp = {n: float(m.predict_proba(x)[0, 1]) for n, m in members.items()}
    c3.markdown("**Model agreement** (individual models)")
    c3.bar_chart(pd.Series(mp, name="Risk"), height=170)
    spread = max(mp.values()) - min(mp.values())
    c3.caption("Models agree closely." if spread < .15 else "Models disagree noticeably, so treat this estimate with extra caution.")

    st.markdown("**How you compare with the training population** (percentile)")
    pc = {f: float((data[f].dropna() < vals[f]).mean() * 100) for f in FEATURES if not np.isnan(vals[f])}
    st.bar_chart(pd.Series(pc, name="Percentile"), horizontal=True, height=260)

    cA, cB = st.columns(2)
    if cA.button("➕ Save to session history"):
        st.session_state.setdefault("hist", []).append(
            {"time": dt.datetime.now().strftime("%H:%M:%S"), **vals, "Risk": round(risk, 4)})
    report = (f"Diabetes risk report ({dt.datetime.now():%Y-%m-%d %H:%M})\n"
              f"Estimated risk: {risk:.1%}   Threshold: {thr:.0%}   Result: {level}\n\n"
              + "\n".join(f"{k}: {v}" for k, v in vals.items())
              + "\n\nEducational tool only. Not a medical diagnosis.")
    cB.download_button("⬇️ Download report", report, "risk_report.txt")
    if st.session_state.get("hist"):
        h = pd.DataFrame(st.session_state["hist"])
        st.subheader("Session history")
        st.dataframe(h, width="stretch", hide_index=True)
        st.line_chart(h.set_index("time")["Risk"])

# ================= 2. Explainability =================
with tabs[1]:
    st.subheader("Why this prediction? (SHAP)")
    sv = explain_one(tuple(float(v) for v in x.iloc[0].values))
    fig = plt.figure(); shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig, bbox_inches="tight"); plt.close(fig)
    st.caption("Red bars push risk up, blue bars push it down, relative to the average patient.")
    contrib = pd.DataFrame({"Feature": FEATURES, "Value": x.iloc[0].values, "Impact": sv.values[0]})
    st.dataframe(contrib.sort_values("Impact", key=abs, ascending=False)
                 .style.format({"Impact": "{:+.3f}"}), hide_index=True, width="stretch")
    st.subheader("Global feature importance")
    st.bar_chart(global_importance(), horizontal=True)
    st.subheader("What-if analysis")
    f = st.selectbox("Vary a feature", FEATURES, index=1)
    rng = np.linspace(data[f].min(), data[f].max(), 40)
    sweep = pd.concat([x] * len(rng), ignore_index=True); sweep[f] = rng
    st.line_chart(pd.DataFrame({f: rng, "Risk": final.predict_proba(sweep)[:, 1]}).set_index(f))

# ================= 3. Risk-reduction plan (counterfactual) =================
with tabs[2]:
    st.subheader("What would bring the risk down?")
    st.caption("Searches small, realistic reductions in glucose, BMI and blood pressure that lower the model's "
               "estimate below your threshold. This shows model associations, not medical advice.")
    if not flagged:
        st.success("Risk is already below the threshold, so there is nothing to reduce.")
    elif any(np.isnan(vals[k]) for k in ["Glucose", "BMI", "BloodPressure"]):
        st.info("Enter glucose, BMI and blood pressure to run this analysis.")
    else:
        combos = list(itertools.product(range(0, 41, 5), range(0, 9, 1), range(0, 16, 5)))
        grid = pd.concat([x] * len(combos), ignore_index=True)
        d = np.array(combos)
        grid["Glucose"] = np.maximum(vals["Glucose"] - d[:, 0], 70)
        grid["BMI"] = np.maximum(vals["BMI"] - d[:, 1], 18.5)
        grid["BloodPressure"] = np.maximum(vals["BloodPressure"] - d[:, 2], 55)
        grid["Risk"] = final.predict_proba(grid[FEATURES])[:, 1]
        grid["Cost"] = d[:, 0] / 40 + d[:, 1] / 8 + d[:, 2] / 15      # normalised effort
        ok = grid[grid.Risk < thr].sort_values("Cost").head(3)
        if ok.empty:
            st.warning("No combination in the searched range gets below the threshold. Consider medical advice.")
        for i, r in enumerate(ok.itertuples(), 1):
            st.markdown(f"**Option {i}:** Glucose **−{vals['Glucose'] - r.Glucose:.0f}** mg/dL · "
                        f"BMI **−{vals['BMI'] - r.BMI:.1f}** · Diastolic BP **−{vals['BloodPressure'] - r.BloodPressure:.0f}** "
                        f"→ risk **{r.Risk:.1%}** (from {risk:.1%})")

# ================= 4. Batch =================
with tabs[3]:
    st.subheader("Batch prediction from CSV")
    st.caption("Upload a CSV with columns: " + ", ".join(FEATURES) + ". Zeros in medical columns are treated as missing.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        try:
            b = pd.read_csv(up)
            miss = [c for c in FEATURES if c not in b.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                B = b[FEATURES].copy(); B[ZERO_AS_MISSING] = B[ZERO_AS_MISSING].replace(0, np.nan)
                b["Risk"] = final.predict_proba(B)[:, 1].round(4)
                b["Flagged"] = b["Risk"] >= thr
                st.dataframe(b, width="stretch")
                st.write(f"{b['Flagged'].sum()} of {len(b)} flagged at threshold {thr:.0%}")
                st.download_button("⬇️ Download results", b.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    st.download_button("Download CSV template", ",".join(FEATURES) + "\n", "template.csv")

# ================= 5. Performance =================
with tabs[4]:
    st.subheader("Model comparison (held-out 20% test set, threshold 0.5)")
    res = pd.DataFrame(meta["results"]).T
    st.dataframe(res.style.highlight_max(axis=0, color="#cfe8cf", subset=res.columns.drop("brier"))
                 .highlight_min(axis=0, color="#cfe8cf", subset=["brier"]), width="stretch")
    st.caption("Brier score: lower is better (measures probability quality).")
    a, b_, c = st.columns(3)
    with a:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.plot(meta["roc"]["fpr"], meta["roc"]["tpr"])
        ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        st.pyplot(fig); plt.close(fig)
    with b_:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.plot(meta["pr"]["r"], meta["pr"]["p"])
        ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall (AP={meta['pr']['ap']:.2f})")
        st.pyplot(fig); plt.close(fig)
    with c:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.plot(meta["calib"]["pred"], meta["calib"]["obs"], "o-")
        ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set(xlabel="Predicted risk", ylabel="Observed rate", title="Calibration")
        st.pyplot(fig); plt.close(fig)

    st.subheader(f"Trade-off at your threshold ({thr:.0%})")
    p, yt = np.array(meta["test_probs"]), np.array(meta["test_y"])
    def stats(t):
        pr = p >= t; tp = (pr & (yt == 1)).sum(); fp = (pr & (yt == 0)).sum()
        fn = (~pr & (yt == 1)).sum(); tn = (~pr & (yt == 0)).sum()
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                    prec=tp / max(tp + fp, 1), acc=(tp + tn) / len(yt))
    s = stats(thr)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensitivity (recall)", f"{s['sens']:.0%}"); m2.metric("Specificity", f"{s['spec']:.0%}")
    m3.metric("Precision", f"{s['prec']:.0%}"); m4.metric("Accuracy", f"{s['acc']:.0%}")
    fig, ax = plt.subplots(figsize=(3.4, 3.2)); cm = [[s["tn"], s["fp"]], [s["fn"], s["tp"]]]
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(np.array(cm)): ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["No", "Yes"]); ax.set_yticks([0, 1], ["No", "Yes"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix"); st.pyplot(fig); plt.close(fig)
    ts = np.arange(0.1, 0.9, 0.02)
    st.line_chart(pd.DataFrame({"Threshold": ts, "Sensitivity": [stats(t)["sens"] for t in ts],
                                "Specificity": [stats(t)["spec"] for t in ts]}).set_index("Threshold"))
    st.info("Lowering the threshold catches more true cases (higher sensitivity) at the cost of more false alarms.")

# ================= 6. Data explorer =================
with tabs[5]:
    st.write(f"{len(data)} records · {data['Outcome'].mean():.1%} diabetic")
    feat = st.selectbox("Feature distribution", FEATURES, key="dist")
    fig, ax = plt.subplots(figsize=(6, 3))
    for o, col in [(0, "tab:blue"), (1, "tab:red")]:
        ax.hist(data.loc[data.Outcome == o, feat].dropna(), bins=25, alpha=.55, color=col,
                label=["No diabetes", "Diabetes"][o])
    if not np.isnan(vals[feat]): ax.axvline(vals[feat], c="k", ls="--", label="You")
    ax.legend(); st.pyplot(fig); plt.close(fig)
    st.dataframe(data.corr().round(2), width="stretch")

st.divider()
st.caption("Trained on the Pima Indians Diabetes dataset (adult women of Pima heritage); predictions may not "
           "generalise to other populations. Not a substitute for professional medical advice.")
