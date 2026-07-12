import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (train_test_split,cross_val_score,StratifiedKFold,GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import clone

RANDOM_STATE = 42
SCIRIP_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILENAME = "cleaned_data.csv"
DATA_SEARCH_PATH = [
    os.path.join(SCIRIP_DIR, CSV_FILENAME),
    os.path.join(SCIRIP_DIR, "..", "part1_data_prep_eda", CSV_FILENAME)
]

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def get_data_path():
    for path in DATA_SEARCH_PATH:
        if os.path.exists(path):
            print(f"CSV file found at : {path}")
            return path
    raise FileNotFoundError(
        f"Could not find '{CSV_FILENAME}' in any of {DATA_SEARCH_PATH}. "
        "Make sure Part 1 has been run and cleaned_data.csv exists in "
        "'part1_data_prep_eda/', or place a copy next to this script."
    )

# ---------------------------------------------------------------------------
# SETUP: Re-create X_train_scaled, X_test_scaled, y_clf_train, y_clf_test
# EXACTLY as in Part 2 (same encoding, same split, same random_state)
# ---------------------------------------------------------------------------

section("SETUP: Reproducing Part 2 preprocessing (X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)")

DATA_PATH = get_data_path()
df = pd.read_csv(DATA_PATH)
print(f"\n Loaded cleaned CSV data - shape: {df.shape}")

REG_TARGET = "MonthlyIncome"
CLF_TARGET = "Attrition"

y_clf = df[CLF_TARGET].map({"Yes" : 1, "No" : 0}).astype(int)
X = df.drop(columns=[REG_TARGET,CLF_TARGET])

if "BusinnessTravel" in X.columns:
    travel_order = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Freaquently": 2}
    X["BusineesTravel"] = X["BusineesTravel"].map(travel_order)

if "OverTime" in X.columns:
    X["OverTime"] = X["OverTime"].map({"Yes": 1, "No": 0})

nominal_columns = [c for c in ["Department", "EducationField", "Gender",
                             "JobRole", "MaritalStatus", "Over18"]
                             if c in X.columns]

X = pd.get_dummies(X,columns=nominal_columns,drop_first=True)

remaining_object_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

for col in remaining_object_cols:
    X[col] = X[col].astype("category").cat.codes


X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=RANDOM_STATE
)
 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=X_test.index)
 
print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
print(f"y_clf_train value counts:\n{y_clf_train.value_counts()}")

# ---------------------------------------------------------------------------
# TASK 1: Decision Tree baseline (unconstrained)
# ---------------------------------------------------------------------------

section("TASK 1: Decision Tree baseline (unconstrained)")

dt_unconstrained = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_unconstrained.fit(X_train_scaled,y_clf_train)

train_acc_unconstrained = accuracy_score(y_clf_train,dt_unconstrained.predict(X_train_scaled))
test_acc_unconstrained = accuracy_score(y_clf_test,dt_unconstrained.predict(X_test_scaled))

print(f"Decision tree unconstrained -> Train accuracy: {train_acc_unconstrained:.4f}" f"| Test accuracy : {test_acc_unconstrained:.4f}")
print(f" Train-Test gap : {train_acc_unconstrained-test_acc_unconstrained:.4f}")

# ---------------------------------------------------------------------------
# TASK 2: Controlled Decision Tree
# ---------------------------------------------------------------------------

section("TASK 2: Controlled Decision Tree")

dt_controlled = DecisionTreeClassifier(max_depth=5,min_samples_split=20,random_state=RANDOM_STATE)
dt_controlled.fit(X_train_scaled,y_clf_train)

train_acc_controlled = accuracy_score(y_clf_train,dt_controlled.predict(X_train_scaled))
test_acc_controlled = accuracy_score(y_clf_test,dt_controlled.predict(X_test_scaled))

print(f"Controlled Decision tree -> Train accurcy :{train_acc_controlled:.4f}" f"| Test accuracy : {test_acc_controlled:.4f}")
print(f"Train-Test Gap: {train_acc_controlled-test_acc_controlled:.4f}")
print(f"\n Comparison of Train-Test Gaps:")
print(f" Unconstrained : {train_acc_unconstrained-test_acc_unconstrained:.4f}")
print(f"Controlled: {train_acc_controlled-test_acc_controlled:.4f}")

# ---------------------------------------------------------------------------
# TASK 3: Gini vs Entropy comparison
# ---------------------------------------------------------------------------

section("TASK 3: Gini vs Entropy comparison")

dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini",random_state=RANDOM_STATE)
dt_gini.fit(X_train_scaled,y_clf_train)
test_acc_gini = accuracy_score(y_clf_test,dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5,criterion="entropy", random_state=RANDOM_STATE)
dt_entropy.fit(X_train_scaled,y_clf_train)
test_acc_entropy = accuracy_score(y_clf_test,dt_entropy.predict(X_test_scaled))

print(f"Gini Criterion -> Test Accuracy: {test_acc_gini:.4f}")
print(f"Entropy Criterion -> Test Accurcy: {test_acc_entropy:.4f}")

# ---------------------------------------------------------------------------
# TASK 4: Random Forest
# ---------------------------------------------------------------------------
section("TASK 4: Random Forest (n_estimators=100,max_depth=1)")

rf_model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=RANDOM_STATE)
rf_model.fit(X_train_scaled,y_clf_train)

train_acc_rf = accuracy_score(y_clf_train,rf_model.predict(X_train_scaled))
test_acc_rf = accuracy_score(y_clf_test,rf_model.predict(X_test_scaled))
rf_proba_test = rf_model.predict_proba(X_test_scaled)[:,1]
auc_rf = roc_auc_score(y_clf_test,rf_proba_test)

print(f"Random Forest -> Train Accuracy: {train_acc_rf:.4f} | Test Accuracy: {test_acc_rf:.4f} | Test Roc-Auc: {auc_rf:.4f}")

feature_importance_table = pd.DataFrame({
    "features": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance",ascending=False)

top5_features = feature_importance_table.head(5)
print(f"\n Top 5 features by importance:")
print(top5_features.to_string(index=False))

# ---------------------------------------------------------------------------
# TASK 4a: Gradient Boosting
# ---------------------------------------------------------------------------
section("TASK 4a: Gradient Boosting (n_estimators=100,learning_rate=0.1,max_depth=3)")

gb_model = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=RANDOM_STATE)
gb_model.fit(X_train_scaled,y_clf_train)

train_acc_gb = accuracy_score(y_clf_train,gb_model.predict(X_train_scaled))
test_acc_gb = accuracy_score(y_clf_test,gb_model.predict(X_test_scaled))
gb_proba_test = gb_model.predict_proba(X_test_scaled)[:,1]
gb_auc_test = roc_auc_score(y_clf_test,gb_proba_test)

print(f"Gradiant Boosting -> Train Accurcy: {train_acc_gb:.4f} | Test Accuracy: {test_acc_gb:.4f} | Test ROC-AUC : {gb_auc_test:.4f}")

# ---------------------------------------------------------------------------
# TASK 4b: Feature ablation study
# ---------------------------------------------------------------------------
section("TASK 4b: Feature ablation study")

lowest5_feature = feature_importance_table.tail(5)["features"].tolist()
print(f"5 lowest-importance feature (to be removed): {lowest5_feature}")

feature_cols = X.columns.tolist()
keep_idx = [i for i, c in enumerate("feature_cols") if c not in lowest5_feature]

X_train_reduced = X_train_scaled[:,keep_idx]
X_test_reduced = X_test_scaled[:,keep_idx]

rf_reduced = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=RANDOM_STATE)
rf_reduced.fit(X_train_reduced,y_clf_train)
rf_reduced_proba_test = rf_reduced.predict_proba(X_test_reduced)[:,1]
auc_rf_reduced = roc_auc_score(y_clf_test,rf_reduced_proba_test)

print(f"\nFull Random Forest (all {len(feature_cols)} features)      -> Test AUC: {auc_rf:.4f}")
print(f"Reduced Random Forest ({len(keep_idx)} features, 5 removed) -> Test AUC: {auc_rf_reduced:.4f}")
print(f"AUC change: {auc_rf_reduced - auc_rf:+.4f}")

# ---------------------------------------------------------------------------
# TASK 5: Cross-validated comparison
# ---------------------------------------------------------------------------
section("TASK 5: Cross-validated comparison")

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)

cv_model = {
    "Logistic Regression (c=1.0)" : LogisticRegression(max_iter=1000,C=1.0,random_state=RANDOM_STATE),
    "Decision Tree(Max_depth = 5)" : DecisionTreeClassifier(max_depth=5,min_samples_split=20,random_state=RANDOM_STATE),
    "Random Forest" : RandomForestClassifier(n_estimators=100,max_depth=10,random_state=RANDOM_STATE),
    "Gradiant Boosting" : GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=RANDOM_STATE),
}

cv_results = []
for name, model in cv_model.items():
    scores = cross_val_score(model,X_train_scaled,y_clf_train,cv=cv,scoring="roc_auc",n_jobs=-1)
    cv_results.append({"Model" : name, "Mean Auc" : scores.mean(), "Std Auc" : scores.std()})
    print(f"{name:35s} -> Mean AUC: {scores.mean():.4f} | Std AUC: {scores.std():.4f}")
 
cv_results_table = pd.DataFrame(cv_results)
print("\nFull CV comparison table:")
print(cv_results_table.to_string(index=False))

# ---------------------------------------------------------------------------
# TASK 6: Hyperparameter tuning with GridSearchCV
# ---------------------------------------------------------------------------

section("TASK 6: Hyperparameter tuning with GridSearchCV")

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=RANDOM_STATE)
)

param_grid = {
    "randomforestclassifier__n_estimators": [50,100,200],
    "randomforestclassifier__max_depth": [5,10,None],
    "randomforestclassifier__min_samples_leaf": [1,5],
}

grid_search = GridSearchCV(
    pipeline,param_grid,
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE),
    scoring="roc_auc", n_jobs=-1
)

grid_search.fit(X_train,y_clf_train)

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
n_fits = n_configs * 5

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score (roc_auc): {grid_search.best_score_:.4f}")
print(f"\nTotal configurations evaluated: {n_configs} (grid combinations) x 5 folds = {n_fits} total fits")

best_pipeline = grid_search.best_estimator_

# ---------------------------------------------------------------------------
# TASK 6b: Manual learning curve
# ---------------------------------------------------------------------------
section("TASK 6b: Manual learning curve")

learning_curve_rows = []
X_train_reset = X_train.reset_index(drop=True)
y_clf_train_reset = y_clf_train.reset_index(drop=True)

for frac in [0.2,0.4,0.6,0.8,1.0]:
    n_rows = int(frac * len(X_train_reset))
    X_subset = X_train_reset.iloc[:n_rows]
    y_subset = y_clf_train_reset.iloc[:n_rows]

    pipeline_fraction = clone(best_pipeline)
    pipeline_fraction.fit(X_subset,y_subset)

    train_proba = pipeline_fraction.predict_proba(X_subset)[:,1]
    train_auc = roc_auc_score(y_subset,train_proba)

    test_proba = pipeline_fraction.predict_proba(X_test_scaled_df)[:,1]
    test_auc = roc_auc_score(y_clf_test,test_proba)

    learning_curve_rows.append({
        "Training fraction": frac, "Training AUC": train_auc, "Test AUC": test_auc
    })
    print(f"Fraction {frac:.1f} (n={n_rows:4d}) -> Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")
 
learning_curve_table = pd.DataFrame(learning_curve_rows)
print("\nFull learning curve table:")
print(learning_curve_table.to_string(index=False))

# ---------------------------------------------------------------------------
# TASK 7: Serialize the best model
# ---------------------------------------------------------------------------
section("TASK 7: Serialize the best model")

model_path = os.path.join(SCIRIP_DIR,"best_model.pk1")
joblib.dump(best_pipeline,model_path)
print(f"Saved best pipeline to : {model_path}")

# Reload-and-predict block (>= 5 lines, runs without errors)
loaded_model = joblib.load(model_path)
sample_rows = X_test.iloc[:2]
sample_predictions =  loaded_model.predict(sample_rows)
sample_probalities = loaded_model.predict_proba(sample_rows)[:,1]
print(f"Reloaded model predictions on 2 sample rows: {sample_predictions.tolist()}")
print(f"Reloaded model probabilities on 2 sample rows: {sample_probalities.tolist()}")


# ---------------------------------------------------------------------------
# TASK 8: Summary comparison table (Parts 2 + 3)
# ---------------------------------------------------------------------------
section("TASK 8: Summary comparison table (Parts 2 + 3)")

summary_rows = []
for _,row in cv_results_table.iterrows():
    summary_rows.append({
        "Model" : row["Model"],
        "5-fold cv Mean Auc" : row["Mean Auc"],
        "5-fold cv Std Auc" : row["Std Auc"],
    })

# Attach test-set AUC for each CV'd model (fit fresh on full X_train_scaled)
test_auc_lookup = {}
for name, model in cv_model.items():
    m = clone(model)
    m.fit(X_train_scaled, y_clf_train)
    proba = m.predict_proba(X_test_scaled)[:, 1]
    test_auc_lookup[name] = roc_auc_score(y_clf_test, proba)
 
for row in summary_rows:
    row["Test-set AUC"] = test_auc_lookup[row["Model"]]
 
# Add the tuned GridSearchCV pipeline as its own row
tuned_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
tuned_test_auc = roc_auc_score(y_clf_test, tuned_test_proba)
summary_rows.append({
    "Model": "Tuned Random Forest (GridSearchCV)",
    "5-fold CV Mean AUC": grid_search.best_score_,
    "5-fold CV Std AUC": np.nan,
    "Test-set AUC": tuned_test_auc,
})
 
summary_table = pd.DataFrame(summary_rows)
print(summary_table.to_string(index=False))

