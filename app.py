import json, hmac
import joblib, numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
HELP = {
    "Pregnancies": "Number of times pregnant",
    "Glucose": "Plasma glucose, 2 h oral glucose tolerance test (mg/dL)",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skin-fold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/ml)",
    "BMI": "Body mass index (kg/m²)",
    "DiabetesPedigreeFunction": "Family-history score (higher = stronger history)",
    "Age": "Age in years"}

# ---------- Authentication (credentials live in st.secrets, not in code) ----------
def login():
    if st.session_state.get("auth"):
        return True
    st.title("🩺 Diabetes Risk Predictor")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        good_u = st.secrets.get("auth", {}).get("username", "")
        good_p = st.secrets.get("auth", {}).get("password", "")
        if hmac.compare_digest(u, good_u) and hmac.compare_digest(p, good_p):
            st.session_state["auth"] = True
            st.rerun()
        st.error("Invalid credentials")
    return False

if not login():
    st.stop()

# ---------- Cached resources ----------
@st.cache_resource
def load():
    try:
        model = joblib.load("model.joblib")
    except Exception:  # library version mismatch: retrain once on the server
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)
        model = joblib.load("model.joblib")
    meta = json.load(open("metrics.json"))
    bg = pd.read_csv("background.csv")
    f = lambda X: model.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]
    explainer = shap.Explainer(f, bg, feature_names=FEATURES)
    return model, meta, bg, explainer

@st.cache_data
def load_data():
    d = pd.read_csv("diabetes.csv")
    return d

model, meta, bg, explainer = load()

with st.sidebar:
    st.header("Patient inputs")
    vals = {
        "Pregnancies": st.number_input("Pregnancies", 0, 20, 1, help=HELP["Pregnancies"]),
        "Glucose": st.slider("Glucose", 40, 250, 120, help=HELP["Glucose"]),
        "BloodPressure": st.slider("Blood pressure", 30, 130, 70, help=HELP["BloodPressure"]),
        "SkinThickness": st.slider("Skin thickness", 5, 100, 25, help=HELP["SkinThickness"]),
        "Insulin": st.slider("Insulin", 10, 850, 80, help=HELP["Insulin"]),
        "BMI": st.slider("BMI", 15.0, 70.0, 28.0, 0.1, help=HELP["BMI"]),
        "DiabetesPedigreeFunction": st.slider("Pedigree function", 0.05, 2.5, 0.4, 0.01,
                                              help=HELP["DiabetesPedigreeFunction"]),
        "Age": st.slider("Age", 18, 90, 35, help=HELP["Age"])}
    if st.button("Sign out"):
        st.session_state.clear(); st.rerun()

st.title("🩺 Diabetes Risk Predictor")
st.caption(f"Model: **{meta['best_model']}** · Educational tool, not a medical diagnosis.")
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "🧠 Explainability", "📊 Model performance", "🔎 Data explorer"])

x = pd.DataFrame([vals])[FEATURES]
risk = float(model.predict_proba(x)[0, 1])

with tab1:
    c1, c2 = st.columns([1, 2])
    level = "Low" if risk < .3 else "Moderate" if risk < .6 else "High"
    c1.metric("Estimated risk", f"{risk:.1%}", level)
    c1.progress(risk)
    if level == "High":
        c2.error("High estimated risk. Please consult a healthcare professional for proper testing.")
    elif level == "Moderate":
        c2.warning("Moderate estimated risk. Consider lifestyle review and a screening test.")
    else:
        c2.success("Low estimated risk based on the values entered.")
    # Reference ranges
    flags = []
    if vals["Glucose"] >= 140: flags.append("Glucose is elevated (≥140 mg/dL after OGTT)")
    if vals["BMI"] >= 30: flags.append("BMI is in the obese range (≥30)")
    if vals["BloodPressure"] >= 90: flags.append("Diastolic BP is high (≥90 mm Hg)")
    if flags:
        c2.markdown("**Notable inputs:**\n" + "\n".join(f"- {f}" for f in flags))

with tab2:
    st.subheader("Why this prediction?")
    sv = explainer(x)
    fig = plt.figure(); shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig, bbox_inches="tight"); plt.close(fig)
    st.caption("Red bars push the risk up, blue bars push it down, relative to the average patient "
               "in the training data.")
    contrib = pd.DataFrame({"Feature": FEATURES, "Value": x.iloc[0].values,
                            "Impact on risk": sv.values[0]}).sort_values("Impact on risk", key=abs, ascending=False)
    st.dataframe(contrib.style.format({"Impact on risk": "{:+.3f}"}), hide_index=True, use_container_width=True)

    st.subheader("Global feature importance")
    gsv = explainer(bg.head(60))
    imp = pd.Series(np.abs(gsv.values).mean(0), index=FEATURES).sort_values()
    st.bar_chart(imp)

    st.subheader("What-if analysis")
    f = st.selectbox("Vary a feature", FEATURES, index=1)
    rng = np.linspace(bg[f].min(), bg[f].max(), 40)
    sweep = pd.concat([x] * len(rng), ignore_index=True); sweep[f] = rng
    st.line_chart(pd.DataFrame({f: rng, "Risk": model.predict_proba(sweep)[:, 1]}).set_index(f))

with tab3:
    st.subheader("Model comparison (held-out 20% test set)")
    res = pd.DataFrame(meta["results"]).T.drop(columns="params")
    st.dataframe(res.style.highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(meta["roc"]["fpr"], meta["roc"]["tpr"], label=meta["best_model"])
        ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate"); ax.set_title("ROC curve"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(meta["confusion"], cmap="Blues")
        for (i, j), v in np.ndenumerate(np.array(meta["confusion"])):
            ax.text(j, i, v, ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["No diabetes", "Diabetes"]); ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        st.pyplot(fig); plt.close(fig)
    st.info("Recall matters most in screening: a missed case is costlier than a false alarm.")

with tab4:
    d = load_data()
    st.write(f"{len(d)} records · {d['Outcome'].mean():.1%} diabetic")
    feat = st.selectbox("Feature distribution", FEATURES, key="dist")
    fig, ax = plt.subplots(figsize=(6, 3))
    for o, c in [(0, "tab:blue"), (1, "tab:red")]:
        ax.hist(d.loc[d.Outcome == o, feat], bins=25, alpha=.55, color=c, label=["No diabetes", "Diabetes"][o])
    ax.axvline(vals[feat], c="k", ls="--", label="You"); ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.dataframe(d.corr().round(2), use_container_width=True)

st.divider()
st.caption("Trained on the Pima Indians Diabetes dataset (adult women of Pima heritage). "
           "Predictions may not generalise to other populations.")
