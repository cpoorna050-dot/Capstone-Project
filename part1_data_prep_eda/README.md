# Part 1 — Data Prep & EDA

## Dataset
IBM HR Analytics Employee Attrition & Performance
Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Rows: 1470  Columns: 35
Numeric target (used in Part 2): MonthlyIncome
Categorical target (used in Part 2): Attrition

Why this dataset: it is a well-documented, widely-used HR dataset that supports
dual target modeling — a numeric target (MonthlyIncome) and a categorical target
(Attrition) — allowing comparison of what drives pay versus what drives quitting.

## Null Value Analysis
No missing values were found in any of the 35 columns (confirmed via
`df.isnull().sum()` — every column showed a null count and null percentage of 0).
No columns exceeded the 20% null threshold, so no median imputation was required
for Part 1.

## Duplicate Detection
Duplicates found: 0
Rows removed: 0
Shape after dedup: (1470, 35) — unchanged
Change in null % after removal: 0.0000 (no duplicates existed, so no change)

## Data Type Correction
Columns converted to category dtype: `Attrition`, `BusinessTravel`, `Department`,
`EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `Over18`, `OverTime`
Memory usage before conversion: 1040.53 KB
Memory usage after conversion: 313.47 KB
Memory reduction: 727.06 KB (69.9% smaller) — a large reduction because these
columns store repeated string labels (e.g. Department has only 3 unique values
repeated across 1470 rows); `category` dtype stores each unique label once and
references it by integer code instead of duplicating the full string each row.

## Skewness
Most skewed column: `YearsSinceLastPromotion` (skew = 1.984)
Interpretation: this is strong positive skew — a long right tail, meaning most
employees were promoted recently (low values) while a smaller group hasn't been
promoted in many years (high values pulling the tail out). Because of this skew,
the mean (2.19) is pulled upward compared to the median (1.00) — filling missing
values with the mean would overestimate the "typical" time since promotion for
most employees, so median is the safer imputation choice for this column.

## Outlier Detection (IQR)
**YearsSinceLastPromotion**: Q1=0.00, Q3=3.00, IQR=3.00, bounds=[-4.50, 7.50], outliers=107
**Age**: Q1=30.00, Q3=43.00, IQR=13.00, bounds=[10.50, 62.50], outliers=0

Decision: outliers in `YearsSinceLastPromotion` are retained (not dropped) for
Part 2. They represent a real, meaningful subset of long-tenured employees who
haven't been promoted recently — likely a genuine predictor of attrition risk
rather than noise or a data error. `Age` has zero IQR outliers, so no decision
is needed there.

## Visualizations
1. **Line plot**: shows `MonthlyIncome` across row index — no obvious trend by
   row order, confirming rows aren't pre-sorted by income (as expected for raw
   HR data).
2. **Bar chart**: mean `MonthlyIncome` by `Department` — Research & Development
   and Sales carry more senior/high-paying roles than Human Resources on average.
3. **Histogram**: `YearsSinceLastPromotion` is heavily right-skewed — a large
   spike near 0-1 years with a long thinning tail out to 15 years, matching the
   1.984 skew value.
4. **Scatter plot**: `YearsAtCompany` vs `MonthlyIncome` shows a positive but
   noisy relationship — income generally rises with tenure, but with wide
   spread at every tenure level (job role/level matters more than tenure alone).
5. **Box plot**: `MonthlyIncome` by `Attrition` — employees who left (`Attrition
   = Yes`) show a visibly lower median income and a tighter, lower spread than
   those who stayed, suggesting pay level relates to attrition risk.
6. **Correlation heat map**: highest |correlation| pair = `MonthlyIncome` and
   `JobLevel` (r = 0.950). This is very unlikely to be purely causal in the
   "income causes job level" sense — it's more likely both are driven by a
   third factor, seniority/role structure, since pay scales are typically
   set *by* job level rather than the reverse.

## Imputation Strategy Comparison (Task 8a)
**YearsSinceLastPromotion**: mean=2.19, median=1.00 → chose median, since the
positive skew (1.984) pulls the mean upward away from what's typical for most
employees.
**PerformanceRating**: mean=3.15, median=3.00 → chose median, since this column
is also positively skewed (1.922), so the mean is inflated by fewer high-rating
outliers relative to the bulk of typical ratings.
Nulls remaining after imputation: 0 (dataset had none to begin with; strategy
is documented here per the task requirement and confirmed via `isnull().sum()`).

## Spearman vs Pearson (Task 8b)
Top 3 pairs with largest |Spearman − Pearson| difference:
1. **PercentSalaryHike & PerformanceRating**: diff = 0.145 → |Spearman| >
   |Pearson|, indicating a monotonic but non-linear relationship (salary hike
   moves consistently with performance rating, but not proportionally — likely
   because hikes are given in fixed bands per rating tier rather than a smooth
   linear scale).
2. **YearsAtCompany & YearsSinceLastPromotion**: diff = 0.098 → monotonic
   non-linear relationship — time since promotion tends to rise with tenure,
   but not at a constant rate (promotions cluster early, then slow).
3. **YearsAtCompany & YearsInCurrentRole**: diff = 0.095 → similarly monotonic
   non-linear — longer-tenured employees tend to stay in role longer, but the
   relationship isn't strictly proportional.

Measure to use for Part 2 feature selection: **Spearman**, since several of the
strongest relationships in this dataset are monotonic-but-non-linear rather
than strictly linear, so Spearman better captures real predictive signal that
Pearson would understate.

## Grouped Aggregation (Task 8c)
Group: `JobRole` by `MonthlyIncome`
Highest mean group: **Manager** (17,181.68)
Highest std group: **Research Director** (2,827.62)
Mean ratio (highest/lowest): **6.54** (Manager vs Sales Representative, 2,626.00)

Interpretation: the 6.54x ratio between the highest- and lowest-paid roles is
large, indicating `JobRole` carries strong predictive signal for `MonthlyIncome`.
However, `Research Director` having the highest standard deviation (2,827.62)
means income varies substantially even within that single role — so `JobRole`
alone is not sufficient to predict income precisely for that group; additional
features (e.g. `JobLevel`, `TotalWorkingYears`) are needed to narrow the estimate.

## Output Files
- `cleaned_data.csv` — cleaned dataset (1470 rows, 35 columns) used in Part 2
- `plots/` — all 6 visualizations:
  `1_line_plot.png`, `2_bar_chart.png`, `3_histogram.png`, `4_scatter_plot.png`,
  `5_box_plot.png`, `6_correlation_heatmap.png`
