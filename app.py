import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩸", layout="wide")

# --- Login Logic ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    st.title("🔐 Hospital Portal Login")
    st.write("Please enter your credentials to access the ML Prediction System.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username == "admin" and password == "smit123":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("❌ Incorrect Username or Password!")

def main_app():
    # --- Logout Button ---
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({'logged_in': False}))
    
    st.title("🩸 ML Diabetes Prediction & Explainability")
    st.write("Enter patient vitals in the sidebar to check the model's prediction.")

    # --- Cache Model Training ---
    @st.cache_resource
    def train_model():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        df = pd.read_csv(url, names=features + ['Outcome'])
        
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
            
        X = df[features]
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
        rf.fit(X_train_resampled, y_train_resampled)
        
        # Calculate Metrics
        rf_preds = rf.predict(X_test_scaled)
        rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, rf_preds)
        roc_auc = roc_auc_score(y_test, rf_probs)
        
        return rf, scaler, features, accuracy, roc_auc

    model, scaler, feature_names, accuracy, roc_auc = train_model()

    # --- Sidebar Input ---
    st.sidebar.header("Patient Vitals Input")
    def get_user_input():
        pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 2)
        glucose = st.sidebar.slider("Glucose Level", 40, 500, 120) 
        bp = st.sidebar.slider("Blood Pressure", 40, 140, 70)
        skin = st.sidebar.slider("Skin Thickness", 5, 100, 20)
        insulin = st.sidebar.slider("Insulin Level", 15, 800, 79)
        bmi = st.sidebar.slider("BMI", 15.0, 60.0, 32.0)
        dpf = st.sidebar.slider("Diabetes Pedigree", 0.05, 2.5, 0.47)
        age = st.sidebar.slider("Age", 21, 90, 33)
        
        user_data = pd.DataFrame({
            'Pregnancies': [pregnancies], 'Glucose': [glucose], 'BloodPressure': [bp],
            'SkinThickness': [skin], 'Insulin': [insulin], 'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf], 'Age': [age]
        })
        return user_data

    user_input_df = get_user_input()
    
    # --- Sidebar Metrics ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Performance")
    st.sidebar.info(f"**Accuracy:** {accuracy * 100:.2f}%\n\n**ROC-AUC Score:** {roc_auc:.4f}")

    st.subheader("Patient Data Overview")
    st.write(user_input_df)

    # --- Prediction Logic ---
    scaled_input = scaler.transform(user_input_df)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Diabetes**")
        else:
            st.success(f"✅ **Low Risk of Diabetes**")
            
    with col2:
        st.info(f"**Probability:** {prediction_proba[1] * 100:.2f}% Diabetic")

    # --- SHAP Graph ---
    st.subheader("Why did the model make this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    if isinstance(shap_values, list):
        user_shap_values = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        user_shap_values = shap_values[0, :, 1]
    else:
        user_shap_values = shap_values[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['red' if val > 0 else 'green' for val in user_shap_values]
    ax.barh(feature_names, user_shap_values, color=colors)
    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    ax.set_title("Feature Impact for Current Patient")
    plt.axvline(0, color='black', linewidth=1)
    st.pyplot(fig)

    # --- Medical Context Logic ---
    st.markdown("### 📝 Detailed Explanation & Health Context")
    st.write("Red bars indicate factors increasing your risk, while green bars indicate factors lowering it. Here is a breakdown of your most impactful features compared to normal human limits:")

    normal_limits = {
        'Glucose': 140,       
        'BloodPressure': 80,  
        'BMI': 25.0,          
        'Insulin': 160        
    }

    feature_impacts = list(zip(feature_names, user_shap_values))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    for feature, impact in feature_impacts[:3]:
        user_val = user_input_df[feature].iloc[0]
        
        if impact > 0:
            st.write(f"- 🔴 **{feature}** significantly **increased** your risk.")
            if feature in normal_limits and user_val > normal_limits[feature]:
                limit = normal_limits[feature]
                exceeded_by_pct = ((user_val - limit) / limit) * 100
                st.markdown(f"  > *Medical Note: Normal {feature} is up to **{limit}**. Your value is **{user_val}**, which exceeded the normal range by **{exceeded_by_pct:.1f}%**.*")
        else:
            st.write(f"- 🟢 **{feature}** actually **lowered** your risk.")

# --- App Routing ---
if not st.session_state['logged_in']:
    login_page()
else:
    main_app()
