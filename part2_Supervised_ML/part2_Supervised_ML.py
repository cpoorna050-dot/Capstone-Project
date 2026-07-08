import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge
from sklearn.metrics import (mean_squared_error,r2_score,confusion_matrix,classification_report,roc_curve,roc_auc_score,precision_score,recall_score,f1_score)
from imblearn.over_sampling import SMOTE

Plot_dir = "plot_part2"
os.makedirs(Plot_dir, exist_ok=True)
RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILENNAME = "Cleaned_data.CSV"
DATA_SEARCH_PATH = [
     os.path.join(SCRIPT_DIR, CSV_FILENNAME), 
     os.path.join(SCRIPT_DIR,"..","part1_data_prep_eda",CSV_FILENNAME),
]

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def get_data_path():
    for path in DATA_SEARCH_PATH:
        if os.path.exists(path):
            print(f"Found Cleaned_data.csv at: {path}")
            return path
    raise FileNotFoundError(
        f"Could not found file '{CSV_FILENNAME}' in any of {DATA_SEARCH_PATH}."
        "Make sure part 1 has been run and cleaned_data.csv exists in"
        "'part1_data_prep_eda/', or place a copy next to this script."
    )

# ---------------------------------------------------------------------------
# TASK 1: Load data & define X, y_reg, y_clf
# ---------------------------------------------------------------------------

section("TASK 1: Load data & define X, y_reg, y_clf")
Data_Path = get_data_path()
df = pd.read_csv(Data_Path)
print(f"loaded cleaned csv: {df.shape}")

REG_TARGET = "MonthlyIncome"
CLF_TARGET = "Attrition"

y_REG = df[REG_TARGET].copy()
y_CLF = df[CLF_TARGET].map({"Yes":1,"No":0}).astype(int)

X = df.drop(columns=[REG_TARGET,CLF_TARGET])

print(f"y_reg (MonthlyIncome) - first 5: {y_REG.head().tolist()}")
print(f"y_clf (Attrition, Yes=1/No=0) - value counts: \n{y_CLF.value_counts()}")
print(f"X shape(Feature only):{X.shape}")

# ---------------------------------------------------------------------------
# TASK 2: Encode categorical columns
# ---------------------------------------------------------------------------

section("TASK 2: Encode categorical columns")

if "BusinessTravel" in X.columns:
    travel_order = {"Non-Travel": 0,"Travel_Rarely": 1,"Travel_Frequently":2}
    X["BusinessTravel"] = X["BusinessTravel"].map(travel_order)
    print("Label-encoded 'BusinessTravel' as Non-Travel=0 < Travel_Rarely=1 "
          "< Travel_Frequently=2 (ordinal: reflects increasing travel intensity).")
    

if "OverTime" in X.columns:
    X["OverTime"] =X["OverTime"].map({"No":0,"Yes":1})
    print("Encoded 'OverTime' as No=0, Yes=1 (binary, no ordinality assumed).")

nominal_cols = [ c for c in["Department", "EducationField", "Gender","JobRole", "MaritalStatus", "Over18"]
                    if c in X.columns]

print(f"\nOne-hot encoding nominal columns: {nominal_cols}")

X = pd.get_dummies(X,columns=nominal_cols,drop_first=True)

remaining_Object_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
for col in remaining_Object_cols:
    X[col] =X[col].astype("category").cat.codes

print(f"\n X columns after encoding:{X.shape}")

# ---------------------------------------------------------------------------
# TASK 3: Leak-free train-test split and scaling
# ---------------------------------------------------------------------------
section("TASK 3: Leak-free train-test split and scaling")

X_train,X_test,y_REG_train,y_REG_test = train_test_split(X,y_REG,test_size=0.2,random_state=RANDOM_STATE)
X_train,X_test,y_CLF_train,y_CLF_test = train_test_split(X,y_CLF,test_size=0.2,random_state=RANDOM_STATE)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# TASK 4: Regression — Linear Regression + Ridge comparison
# ---------------------------------------------------------------------------

section("TASK 4: Regression — Linear Regression + Ridge comparison")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled,y_REG_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_REG_test,y_pred_reg)
r2_lin = r2_score(y_REG_test,y_pred_reg)

print(f"Linear Regression -> MSE :{mse_lin:.2F} | R2 :{r2_lin:.4F}")

Coef_table = pd.DataFrame({
    "feature" : X.columns,
    "coefficient" : lin_reg.coef_
}).sort_values("coefficient",key=lambda s: s.abs(),ascending=False)

print("\n Top 10| coefficient | features:")
print(Coef_table.head(10).to_string(index=False))

top3_features = Coef_table.head(3)
print(f"\nTop 3 largest |coefficient| features:\n{top3_features.to_string(index=False)}")

# ---------------------------------------------------------------------------
# TASK 5: Classification — Logistic Regression
# ---------------------------------------------------------------------------

section("TASK 5: Classification — Logistic Regression")
class_counts = y_CLF_train.value_counts()
class_pct = (class_counts/class_counts.sum())*100
print(f"Training class counts:\n{class_counts}")
print(f"\nTraining class percentages:\n{class_pct.round(2)}")

minority_pct = class_pct.min()
USE_SMOTE = minority_pct < 35
print(f"\nMinority class share: {minority_pct:.2f}% "
      f"({'< 35% -> imbalance handling required' if USE_SMOTE else '>= 35% -> no handling required'})")

if USE_SMOTE:
        print("Chosen strategy: SMOTE oversampling applied ONLY to the training set.")
        print(
            "Why SMOTE over class_weight: SMOTE creates synthetic minority-class "
            "examples so the model sees a balanced training distribution directly, "
            "rather than just re-weighting the loss function — this tends to work "
            "well when the minority class has too few real examples for the model "
            "to learn its decision boundary confidently."
        )
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_scaled_bal,y_CLF_train_bal = smote.fit_resample(X_train_scaled,y_CLF_train)
        print(f"\nBefore SMOTE: {y_CLF_train.value_counts().to_dict()}")
        print(f"After SMOTE:  {pd.Series(y_CLF_train_bal).value_counts().to_dict()}")
else:
     X_train_scaled_bal,y_CLF_train_bal = X_train_scaled,y_CLF_train
     print("Class balanced no resampling applied")

section("TASK 5 (CONT.): LOGISTIC REGRESSION (BASELINE, C=1.0)")

log_reg = LogisticRegression(max_iter=1000,C=1.0,random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled_bal,y_CLF_train_bal)

y_pred_clf = log_reg.predict(X_test_scaled)
y_prob_clf = log_reg.predict_proba(X_test_scaled)[:,1]

cm = confusion_matrix(y_CLF_test,y_pred_clf)
print("confusion matrix")
print(cm)

print("\n Classification report:")
print(classification_report(y_CLF_test,y_pred_clf,target_names=["No Attrition", "Attrition"]))

auc_baseline = roc_auc_score(y_CLF_test,y_prob_clf)
print(f"AUC : {auc_baseline:.4F}")

fpr,tpr, threshold_roc =  roc_curve(y_CLF_test,y_prob_clf)
plt.figure()
plt.plot(fpr, tpr, label= f"Roc curve (AUC = {auc_baseline:.3f})")
plt.plot([0,1],[0,1], linestyle = "--", color= "gray", label= "Random guess")
plt.xlabel("Flase Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc Curve - Logistic regression (c=1.0)")
plt.annotate(f"AUC = {auc_baseline:.3f}", xy=(0.6, 0.2))
plt.legend()
plt.tight_layout()
plt.savefig(f"{Plot_dir}/roc_curve_baseline.png")
plt.close()
print(f"Saved: {Plot_dir}/roc_curve_baseline.png")

# ---------------------------------------------------------------------------
# TASK 5b: Decision-threshold sensitivity (0.30 to 0.70)
# ---------------------------------------------------------------------------

section("TASK 5b: Decision-threshold sensitivity (0.30 to 0.70)")

threshold_rows = []
for threshold in np.arange(0.30,0.71,0.10):
     y_pred_thres = (y_prob_clf >= threshold).astype(int)
     P = precision_score(y_CLF_test,y_pred_thres,zero_division=0)
     R = recall_score(y_CLF_test,y_pred_thres,zero_division=0)
     F1 = f1_score(y_CLF_test,y_pred_thres,zero_division=0)
     threshold_rows.append({"Threshold" : round(threshold,2), "Precision Score" : P, "Recall Score": R, "F1" : F1})

threshold_table = pd.DataFrame(threshold_rows)
print(threshold_table.to_string(index=False))

best_f1_row = threshold_table.loc[threshold_table["F1"].idxmax()]
print(f"\nThreshold maximizing F1: {best_f1_row['Threshold']} (F1 = {best_f1_row['F1']:.4f})")

# ---------------------------------------------------------------------------
# TASK 6: Regularization experiment (C=0.01 vs C=1.0)
# ---------------------------------------------------------------------------
section("TASK 6: Regularization experiment (C=0.01 vs C=1.0)")

log_reg_strong = LogisticRegression(max_iter=1000, C=0.01, random_state=RANDOM_STATE)
log_reg_strong.fit(X_train_scaled_bal,y_CLF_train_bal)

y_pred_strong = log_reg_strong.predict(X_test_scaled)
y_prob_strong = log_reg_strong.predict_proba(X_test_scaled)[:,1]

precision_baseline = precision_score(y_CLF_test,y_pred_clf,zero_division=0)
recall_baseline = recall_score(y_CLF_test,y_pred_clf,zero_division=0)

precision_strong = precision_score(y_CLF_test,y_pred_strong,zero_division=0)
recall_strong = recall_score(y_CLF_test,y_pred_strong,zero_division=0)
auc_strong = roc_auc_score(y_CLF_test,y_prob_strong)

reg_comparison_table = pd.DataFrame({
     "Model" : ["Logistic regression (c=1.0)",  "Logistic regression (c=0.01)"],
     "Precision": [precision_baseline,precision_strong],
     "Recall" : [recall_baseline,recall_strong],
     "AUC" : [auc_baseline,auc_strong]
})

print(reg_comparison_table.to_string(index=False))

# ---------------------------------------------------------------------------
# TASK 6b: Bootstrap confidence interval for AUC difference
# ---------------------------------------------------------------------------
section("TASK 6b: Bootstrap confidence interval for AUC difference")

np.random.seed(RANDOM_STATE)
n_bootsrap = 500
y_CLF_test_arr = np.array(y_CLF_test)
auc_diffs = []

for i in range(n_bootsrap):
     idx = np.random.choice(len(y_CLF_test_arr),size=len(y_CLF_test_arr),replace=True)
     y_sample = y_CLF_test_arr[idx]

     if len(np.unique(y_sample)) < 2:
          continue
     
     prba_baseline_clf = y_prob_clf[idx]
     proba_strong_sample = y_prob_strong[idx]

     auc_b = roc_auc_score(y_sample,prba_baseline_clf)
     auc_s = roc_auc_score(y_sample,proba_strong_sample)
     auc_diffs.append(auc_b - auc_s)

auc_diffs = np.array(auc_diffs)
mean_diff = auc_diffs.mean()
ci_lower = np.percentile(auc_diffs,2.5)
ci_upper = np.percentile(auc_diffs, 97.5)

print(f"Bootstrap iterations used: {len(auc_diffs)} / {n_bootsrap}")
print(f"Mean AUC difference (C=1.0 minus C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
 
excludes_zero = (ci_lower > 0) or (ci_upper < 0)
print(f"95% CI excludes zero: {excludes_zero}")

