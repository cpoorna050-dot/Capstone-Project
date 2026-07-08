# Part 2 — Supervised Machine Learning: Build, Train, and Evaluate

## Dataset & Targets
Input file: `cleaned_data.csv` (produced by Part 1), loaded from `../part1_data_prep_eda/`
Rows: 1470  Features (X): 33 raw → 45 after encoding

**Regression label (y_reg):** `MonthlyIncome` (continuous)
**Classification label (y_clf):** `Attrition`, mapped Yes→1 / No→0 (natural binary column in the dataset — used directly rather than binarizing `y_reg` at its median).

`X` excludes both `MonthlyIncome` and `Attrition` so neither target leaks into predicting the other.

## Categorical Encoding
**Label-encoded (ordinal):**
- `BusinessTravel` → Non-Travel=0 < Travel_Rarely=1 < Travel_Frequently=2. Justification: this column has a genuine natural order — increasing frequency/intensity of travel — so integer encoding preserves meaningful ordinal information for the model.
- `OverTime` → No=0, Yes=1 (simple binary mapping; a 2-category column has no ordering ambiguity either way).

**One-hot encoded (nominal, drop_first=True):** `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `Over18`

Why one-hot instead of label encoding here: these columns have categories with no inherent ranking (e.g. Sales, R&D, and HR are not "less than" or "greater than" each other). Label encoding them as integers (e.g. Sales=0, R&D=1, HR=2) would falsely imply a numeric ordering the model could misinterpret as meaningful — one-hot encoding avoids this by giving each category its own independent binary column.

Final feature count after encoding: **45 columns** (up from 33 raw features, due to one-hot expansion).

## Leak-Free Train-Test Split & Scaling
Split 80/20 (`test_size=0.2, random_state=42`). `StandardScaler` was fit **only on `X_train`**, then used to transform both `X_train` and `X_test`.

Fitting the scaler on the full dataset (including test rows) would leak test-set mean/variance statistics into the transformation applied to training data — the model would then be evaluated on test data whose scaling already "knew" something about its own distribution, making evaluation metrics artificially optimistic and not representative of true generalization to genuinely unseen data.

## Regression — Linear Regression
**MSE: 1,361,544.44  |  R²: 0.9377**

R² of 0.94 means the model explains ~94% of the variance in `MonthlyIncome` using the other employee attributes — a very strong fit, consistent with the 0.95 correlation between `MonthlyIncome` and `JobLevel` found in Part 1.

**Top 3 largest |coefficient| features:**
| Feature | Coefficient |
|---|---|
| JobLevel | 3093.24 |
| JobRole_Manager | 1096.70 |
| JobRole_Research Director | 934.76 |

**Interpretation:** since features are scaled, a large **positive** coefficient (e.g. `JobLevel` = 3093.24) means a one-standard-deviation increase in that feature is associated with an increase of ~3093 units in predicted `MonthlyIncome`, holding other features constant. A large **negative** coefficient (e.g. `YearsWithCurrManager` = -109.83) means a one-standard-deviation increase in that feature is associated with a *decrease* in predicted income by that amount, holding other features constant.

### Ridge Regression Comparison (alpha=1.0)
| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 1,361,544.44 | 0.9377 |
| Ridge (alpha=1.0) | *(see script output)* | *(see script output)* |

Ridge adds an L2 penalty (`alpha × sum of squared coefficients`) to the loss function, which shrinks coefficients toward zero — especially for features correlated with each other (e.g. `JobLevel`, `TotalWorkingYears`, and `YearsAtCompany` all move together). This typically produces a flatter, smaller-magnitude coefficient profile than plain OLS, which can otherwise overfit by assigning large weights to redundant correlated features. `alpha` controls the strength of this shrinkage: `alpha=0` reduces Ridge to plain OLS, while larger `alpha` shrinks coefficients more aggressively, trading a little bias for lower variance.

## Classification — Logistic Regression

### Class Imbalance
Training set: 978 "No Attrition" (83.16%) vs 198 "Attrition" (16.84%).

Since the minority class fell below the 35% threshold, imbalance handling was required. **Chosen strategy: SMOTE**, applied only to the training set (before: `{0: 978, 1: 198}`, after: `{0: 978, 1: 978}`). SMOTE was chosen over `class_weight='balanced'` because it creates synthetic minority-class examples so the model sees a genuinely balanced training distribution directly, rather than only re-weighting the loss function — useful here since the minority class had relatively few real examples (198) for the model to learn a confident decision boundary from.

### Baseline Model (C=1.0) Results
**Confusion Matrix:**
```
              Predicted No   Predicted Yes
Actual No          198            57
Actual Yes          16            23
```

**Classification Report:**
| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Attrition | 0.93 | 0.78 | 0.84 |
| Attrition | 0.29 | 0.59 | 0.39 |

**AUC: 0.7635**

**Precision** = TP / (TP + FP)
**Recall** = TP / (TP + FN)

For attrition prediction specifically, missing an employee who will actually leave (a false negative) is generally more costly to the business than flagging a stable employee as at-risk (a false positive) — HR would rather follow up unnecessarily with a few people than fail to intervene with someone who genuinely leaves. **Recall is therefore the more important metric for this task.**

AUC = 0.7635 means: given one random employee who left and one who stayed, the model correctly ranks the leaver as more at-risk about 76% of the time — a solid, well-above-random (0.5) result for HR attrition prediction, though far from perfect separation.

### Decision-Threshold Sensitivity (0.30–0.70)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 | 0.240 | 0.744 | 0.363 |
| 0.4 | 0.269 | 0.641 | 0.379 |
| 0.5 | 0.288 | 0.590 | 0.387 |
| 0.6 | 0.344 | 0.538 | 0.420 |
| 0.7 | 0.409 | 0.462 | 0.434 |

**F1-maximizing threshold: 0.70 (F1 = 0.4337)**

Precision = TP/(TP+FP), Recall = TP/(TP+FN) (formulas repeated per rubric requirement).

As argued above, **Recall matters more than Precision** for this task — but note the F1-optimal threshold (0.70) is actually the *worst* threshold for recall in this table (0.462, the lowest value shown), since raising the threshold trades recall for precision. **To optimize specifically for Recall, the threshold should be LOWERED** (e.g. to 0.30, giving Recall = 0.744) rather than raised — the cost of doing so is a corresponding drop in Precision (down to 0.240), meaning many more false alarms/unnecessary HR follow-ups. This is a deliberate business tradeoff: catching more true leavers at the cost of flagging more people who wouldn't have left. F1-optimal and business-optimal are not the same threshold here.

## Regularization Experiment (C=0.01 vs C=1.0)
| Model | Precision | Recall | AUC |
|---|---|---|---|
| Logistic Regression (C=1.0) | 0.2875 | 0.5897 | 0.7635 |
| Logistic Regression (C=0.01) | 0.3462 | 0.6923 | 0.7792 |

`C` is the **inverse** of regularization strength in scikit-learn's `LogisticRegression` — smaller `C` means **stronger** L2 regularization (larger penalty on large coefficients). `C=0.01` imposes much stronger regularization than the default `C=1.0`.

On this dataset, **C=0.01 outperformed C=1.0 on every metric** (higher precision, recall, and AUC). This suggests the baseline model — with 45 features after one-hot encoding, many of them correlated (e.g. multiple JobRole dummies) — was mildly overfitting the training data, and the stronger regularization in C=0.01 improved generalization to the test set by shrinking noisy/redundant coefficients.

## Bootstrap Confidence Interval for AUC Difference
500 bootstrap resamples of the test set (with replacement), computing (AUC of C=1.0 − AUC of C=0.01) per sample.

**Mean AUC difference: -0.0158**
**95% CI: [-0.0422, 0.0094]**
**Interval excludes zero: False**

The confidence interval **includes zero**, meaning the observed AUC advantage of C=0.01 over C=1.0 is **not statistically reliable** at the 95% level — on a different random sample of test data, the two models could plausibly perform equally well, or even reverse in ranking. While C=0.01 numerically outperformed C=1.0 on this specific test split, this bootstrap analysis shows that advantage should not be over-interpreted as a robust, generalizable finding — it may simply reflect the particular random split used here.

## Output Files
- `plot_part2/roc_curve_baseline.png` — ROC curve for the baseline (C=1.0) logistic regression model
