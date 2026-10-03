"""Train v2: tuned models + soft-voting ensemble + probability calibration.
Run:  python train_model.py   (needs diabetes.csv next to it)"""
import json, joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv("diabetes.csv")
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
X, y = df[FEATURES], df["Outcome"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def pipe(clf):
    return Pipeline([("prep", ColumnTransformer([("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), FEATURES)])), ("clf", clf)])

grids = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (pipe(RandomForestClassifier(random_state=42, class_weight="balanced")),
                      {"clf__n_estimators": [300], "clf__max_depth": [4, 6, None],
                       "clf__min_samples_leaf": [2, 5]}),
    "Gradient Boosting": (pipe(GradientBoostingClassifier(random_state=42)),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1],
                           "clf__max_depth": [2, 3]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
members, rows = {}, {}
def score(name, est, cv_auc):
    p = est.predict_proba(X_te)[:, 1]; pred = (p >= .5).astype(int)
    rows[name] = dict(cv_auc=round(float(cv_auc), 4), test_auc=round(roc_auc_score(y_te, p), 4),
        accuracy=round(accuracy_score(y_te, pred), 4), precision=round(precision_score(y_te, pred), 4),
        recall=round(recall_score(y_te, pred), 4), f1=round(f1_score(y_te, pred), 4),
        brier=round(brier_score_loss(y_te, p), 4))
    print(name, rows[name])

for name, (p, g) in grids.items():
    gs = GridSearchCV(p, g, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
    members[name] = gs.best_estimator_
    score(name, gs.best_estimator_, gs.best_score_)

ens = VotingClassifier([(n, m) for n, m in members.items()], voting="soft")
ens_auc = cross_val_score(ens, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
ens.fit(X_tr, y_tr); score("Ensemble (voting)", ens, ens_auc)

# calibrate the ensemble so probabilities are trustworthy
final = CalibratedClassifierCV(VotingClassifier([(n, m) for n, m in members.items()], voting="soft"),
                               method="sigmoid", cv=5).fit(X_tr, y_tr)
fcv = cross_val_score(final, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
score("Calibrated Ensemble", final, fcv)

# choose a screening threshold from out-of-fold training predictions (maximise F1.5, mildly recall-weighted)
oof = cross_val_predict(final, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
ths = np.arange(0.10, 0.91, 0.01)
def f2(t):
    pr = (oof >= t).astype(int); p_, r_ = precision_score(y_tr, pr, zero_division=0), recall_score(y_tr, pr)
    return 0 if p_ + r_ == 0 else 3.25 * p_ * r_ / (2.25 * p_ + r_)
rec_th = float(ths[int(np.argmax([f2(t) for t in ths]))])

prob = final.predict_proba(X_te)[:, 1]
fpr, tpr, _ = roc_curve(y_te, prob); pp, rr, _ = precision_recall_curve(y_te, prob)
fo, mp = calibration_curve(y_te, prob, n_bins=6, strategy="quantile")
meta = dict(best_model="Calibrated Ensemble (LR + RF + GB)", results=rows, recommended_threshold=round(rec_th, 2),
            roc=dict(fpr=fpr.tolist(), tpr=tpr.tolist()), pr=dict(p=pp.tolist(), r=rr.tolist(),
            ap=float(average_precision_score(y_te, prob))), calib=dict(obs=fo.tolist(), pred=mp.tolist()),
            test_probs=prob.tolist(), test_y=y_te.tolist(), n_train=len(X_tr), n_test=len(X_te))
joblib.dump(dict(final=final, members=members), "model.joblib")
X_tr.sample(80, random_state=1).to_csv("background.csv", index=False)
json.dump(meta, open("metrics.json", "w"))
print("Recommended threshold:", rec_th, "| saved.")
