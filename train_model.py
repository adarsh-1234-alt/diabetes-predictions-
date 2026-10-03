"""Train, compare and save the diabetes model.  Usage: python train_model.py"""
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv("diabetes.csv")
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)   # 0 is impossible here
X, y = df[FEATURES], df["Outcome"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def pipe(clf):
    return Pipeline([("prep", ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), FEATURES)])), ("clf", clf)])

candidates = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (pipe(RandomForestClassifier(random_state=42, class_weight="balanced")),
                      {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8, None],
                       "clf__min_samples_leaf": [2, 5]}),
    "Gradient Boosting": (pipe(GradientBoostingClassifier(random_state=42)),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
                           "clf__max_depth": [2, 3]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
results, best_name, best_est, best_auc = {}, None, None, -1
for name, (p, grid) in candidates.items():
    gs = GridSearchCV(p, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
    prob = gs.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results[name] = dict(cv_auc=round(gs.best_score_, 4),
        test_auc=round(roc_auc_score(y_te, prob), 4), accuracy=round(accuracy_score(y_te, pred), 4),
        precision=round(precision_score(y_te, pred), 4), recall=round(recall_score(y_te, pred), 4),
        f1=round(f1_score(y_te, pred), 4), params={k: str(v) for k, v in gs.best_params_.items()})
    print(name, results[name])
    if gs.best_score_ > best_auc:
        best_name, best_est, best_auc = name, gs.best_estimator_, gs.best_score_

prob = best_est.predict_proba(X_te)[:, 1]
fpr, tpr, _ = roc_curve(y_te, prob)
meta = dict(best_model=best_name, results=results,
            confusion=confusion_matrix(y_te, (prob >= .5).astype(int)).tolist(),
            roc=dict(fpr=fpr.tolist(), tpr=tpr.tolist()),
            n_train=len(X_tr), n_test=len(X_te))
joblib.dump(best_est, "model.joblib")
X_tr.sample(100, random_state=1).to_csv("background.csv", index=False)  # for SHAP
json.dump(meta, open("metrics.json", "w"))
print("Saved model:", best_name)
