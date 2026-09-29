import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩸", layout="wide")
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
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)
    
    return rf, scaler, features

model, scaler, feature_names = train_model()

# --- Sidebar Input ---
st.sidebar.header("Patient Vitals Input")
def get_user_input():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 2)
    glucose = st.sidebar.slider("Glucose Level", 50, 250, 120)
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

# --- SHAP Graph & Text Explanation ---
st.subheader("Why did the model make this prediction?")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(scaled_input)

if isinstance(shap_values, list):
    user_shap_values = shap_values[1][0]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    user_shap_values = shap_values[0, :, 1]
else:
    user_shap_values = shap_values[0]

# Plotting the Graph
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['red' if val > 0 else 'green' for val in user_shap_values]
ax.barh(feature_names, user_shap_values, color=colors)
ax.set_xlabel("SHAP Value (Impact on Prediction)")
ax.set_title("Feature Impact for Current Patient")
plt.axvline(0, color='black', linewidth=1)
st.pyplot(fig)

# Auto-generating Text Explanation
st.markdown("### 📝 Graph Explanation")
st.write("Red bars push the prediction towards 'Diabetic', while green bars push it towards 'Non-Diabetic'. Based on your specific vitals, here are the top 3 factors driving your result:")

# Pair features with their SHAP values and sort by absolute impact
feature_impacts = list(zip(feature_names, user_shap_values))
feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

for feature, impact in feature_impacts[:3]:
    if impact > 0:
        st.write(f"- 🔴 **{feature}** significantly **increased** your risk.")
    else:
        st.write(f"- 🟢 **{feature}** actually **lowered** your risk.")
