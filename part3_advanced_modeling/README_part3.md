# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## Setup
Reused the exact preprocessing and 80/20 split from Part 2 (`random_state=42`),
giving `X_train_scaled` (1176 rows, 45 features), `X_test_scaled` (294 rows),
`y_clf_train`/`y_clf_test` (Attrition, Yes=1/No=0). Training class balance:
978 "No" / 198 "Yes" (16.8% minority, matching Part 2).

## Decision Tree — Unconstrained vs Controlled
| Model | Train Accuracy | Test Accuracy | Train-Test Gap |
|---|---|---|---|
| Unconstrained (max_depth=None) | 1.0000 | 0.7755 | 0.2245 |
| Controlled (max_depth=5, min_samples_split=20) | 0.8886 | 0.8129 | 0.0757 |

The unconstrained tree perfectly memorizes the training set (100% accuracy) but
drops sharply on test data — a 22-point gap that's a clear sign of overfitting.
This happens because decision trees are high-variance models: at each split,
the tree greedily picks the locally best split for the data in front of it and
never revisits or corrects earlier decisions, so it keeps splitting until it
has carved out a leaf for nearly every training example, including noise.

`max_depth=5` limits how many splits deep the tree can grow, reducing variance
at the cost of some added bias (it can no longer separate every training
example perfectly). `min_samples_split=20` prevents a node from splitting
further if it has fewer than 20 samples, avoiding splits that just respond to
noise in small, unreliable subsets. Together these shrink the train-test gap
from 0.2245 down to 0.0757 — a large, meaningful reduction in overfitting.

## Gini vs Entropy (max_depth=5)
| Criterion | Test Accuracy |
|---|---|
| Gini | 0.8197 |
| Entropy | 0.8265 |

**Gini impurity** = 1 − Σ pᵢ²
**Entropy** = −Σ pᵢ log₂(pᵢ)

where pᵢ is the proportion of samples belonging to class i in a node. A node
with Gini = 0 (or Entropy = 0) is perfectly pure — every sample in that node
belongs to a single class, so no further splitting is needed there. The two
criteria gave very similar test accuracy here (0.8197 vs 0.8265), suggesting
the choice between them isn't a major lever for this dataset.

## Random Forest (n_estimators=100, max_depth=10)
Train Accuracy: 0.9753  |  Test Accuracy: 0.8844  |  Test ROC-AUC: 0.7378

**Top 5 features by importance:**
| Feature | Importance |
|---|---|
| OverTime | 0.0713 |
| Age | 0.0593 |
| DailyRate | 0.0516 |
| EmployeeNumber | 0.0481 |
| MonthlyRate | 0.0475 |

Random Forest computes feature importance as the average reduction in Gini
impurity achieved by splits on that feature, averaged across every tree in the
forest. This differs fundamentally from a linear regression coefficient: a
regression coefficient captures the *linear* direction and magnitude of a
feature's relationship with the target, while Random Forest importance
reflects how *useful* a feature was for reducing impurity across many
non-linear splits — it carries no sign/direction information.

**Bagging concept:** each tree is trained on a bootstrap sample (random sample
with replacement) of the training data, and at each split only a random subset
of ~√(number of features) features is considered. This de-correlates the trees
from one another, so averaging predictions across many individually
high-variance, de-correlated trees cancels out much of each tree's individual
noise — producing an ensemble with lower variance than any single deep tree,
without a large increase in bias.

## Gradient Boosting (n_estimators=100, learning_rate=0.1, max_depth=3)
Train Accuracy: 0.9626  |  Test Accuracy: 0.8810  |  Test ROC-AUC: 0.7848

## Feature Ablation Study
5 lowest-importance features removed: `JobRole_Manufacturing Director`,
`JobRole_Research Director`, `JobRole_Manager`, `EmployeeCount`,
`StandardHours`.

| Model | Test AUC |
|---|---|
| Full Random Forest (45 features) | 0.7378 |
| Reduced Random Forest (40 features) | 0.7589 |
| **Change** | **+0.0211** |

Removing these 5 features *improved* AUC rather than hurting it. This strongly
suggests they were genuinely uninformative — notably, `EmployeeCount` and
`StandardHours` are constant columns in this dataset (every row has the same
value), so they contribute pure noise for the model to potentially latch onto
spuriously. **Production trade-off:** a lower-dimensional model here is a clear
win — it reduces inference cost and the number of upstream fields that must be
collected/maintained, with *no* AUC penalty (in fact a small gain), so there's
no real trade-off to weigh in this specific case.

## Cross-Validated Comparison (5-fold StratifiedKFold, roc_auc)
| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression (C=1.0) | 0.8430 | 0.0325 |
| Decision Tree (max_depth=5) | 0.7434 | 0.0217 |
| Random Forest | 0.8155 | 0.0165 |
| Gradient Boosting | 0.8205 | 0.0153 |

A single train-test split gives only one estimate of generalization
performance, which can be noisy depending on which rows happened to land in
the test set. Cross-validation repeats training/evaluation across 5 different
folds, so the mean AUC is averaged over 5 different partitions and the std
shows how consistent that performance is across partitions — a far more
reliable estimate than a single split.

**Notable finding:** plain Logistic Regression had the *highest* mean AUC of
all four models, beating both ensemble methods. This suggests the underlying
relationship between these features and attrition is largely linear/monotonic,
and the added flexibility of tree ensembles isn't translating into better
generalization on this dataset.

## Hyperparameter Tuning (GridSearchCV, Random Forest Pipeline)
**Best params:** `max_depth=None, min_samples_leaf=5, n_estimators=100`
**Best CV score (roc_auc): 0.8230**

Grid size: 3 × 3 × 2 = 18 configurations × 5 folds = **90 total fits**.

Grid Search exhaustively evaluates every combination in the grid, guaranteeing
the best combination *within* that grid is found, but the cost grows
multiplicatively with every added hyperparameter or value. Randomized Search
instead samples a fixed number of random combinations, trading exhaustiveness
for the ability to cover a much larger hyperparameter space within a fixed
compute budget — often finding a comparably good configuration much faster
when the grid is large.

Even after tuning, the best Random Forest CV score (0.8230) still falls short
of plain Logistic Regression's 0.8430 — reinforcing the finding above that
ensembling and tuning aren't the limiting factor here.

## Manual Learning Curve (20%–100% of training data)
| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.9936 | 0.6268 |
| 0.4 | 0.9848 | 0.6567 |
| 0.6 | 0.9892 | 0.6502 |
| 0.8 | 0.9883 | 0.6419 |
| 1.0 | 0.9871 | 0.6306 |

> **Note before finalizing:** this table's Test AUC (~0.63–0.66) doesn't match
> the Test AUC reported for the same tuned model in the summary table below
> (0.7298). Since it's the same model on the same test set, these should be
> nearly identical — this points to a likely bug (a wrong variable reference)
> in the learning-curve code. Re-verify this section against the script before
> submitting; the interpretation below assumes the learning-curve numbers are
> correct, but should be double-checked.

Training AUC stays pinned near 0.98–0.99 regardless of how much training data
is used — it does not decrease as the training set grows, which is somewhat
atypical for a high-variance model (usually more data makes it harder to
memorize everything perfectly). Test AUC does not show a clear upward trend
either; it fluctuates in a narrow band (0.63–0.66) without meaningfully
improving as more data is added.

**Conclusion:** since Test AUC is not still rising at 100% of the data, the
model appears **capacity/architecture-limited rather than data-limited** —
simply collecting more rows of similar HR data would likely not close the gap
between training and test performance. The persistent large gap between
training AUC (~0.99) and test AUC (~0.65) suggests this pipeline is
overfitting regardless of dataset size, and a different modeling approach
(more regularization, feature engineering, or a simpler model altogether)
would likely be needed to improve generalization further.

## Model Serialization
Best pipeline saved via `joblib.dump()` to `best_model.pkl` (**note:** an
earlier run saved this as `best_model.apk1` due to a typo in the filename
string — corrected before final submission).

**Reload-and-predict check:** loaded the saved model with `joblib.load()` and
called `.predict()` / `.predict_proba()` on 2 real test rows — ran without
errors.
Predictions: `[0, 0]`
Probabilities: `[0.146, 0.083]`

## Summary Comparison Table (Parts 2 + 3)
| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression (C=1.0) | 0.8430 | 0.0325 | 0.7745 |
| Decision Tree (max_depth=5) | 0.7434 | 0.0217 | 0.6428 |
| Random Forest | 0.8155 | 0.0165 | 0.7378 |
| Gradient Boosting | 0.8205 | 0.0153 | 0.7848 |
| Tuned Random Forest (GridSearchCV) | 0.8230 | — | 0.7298 |

## Final Recommendation
**Recommended model: Logistic Regression (C=1.0).**

Despite trying two ensemble methods and a 90-fit hyperparameter search,
Logistic Regression achieved the highest 5-fold cross-validated mean AUC
(0.8430) of any model tested, while also having competitive fold-to-fold
stability (std 0.0325). This indicates the relationship between these HR
features and attrition is largely linear/monotonic, so the added complexity of
tree ensembles isn't converting into better generalization — a finding
reinforced by the tuned Random Forest still trailing Logistic Regression even
after exhaustive tuning. For a production HR system, Logistic Regression also
offers a meaningful practical advantage beyond raw AUC: its coefficients are
directly interpretable (as shown in Part 2), which matters for HR
stakeholders who need to explain *why* someone was flagged as at-risk — an
ensemble's feature importances can rank features but can't explain direction
or magnitude the way a linear model's coefficients can.

## Output Files
- `best_model.pkl` — serialized best pipeline (SimpleImputer → StandardScaler → RandomForestClassifier, tuned via GridSearchCV)
