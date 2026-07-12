import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14,11)

KAGGELE_DATASET = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
CSV_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

LOCAL_SEARCH_PATHS =[CSV_FILENAME,os.path.join("data",CSV_FILENAME)]
Plot_Dir = "Plots"
os.makedirs(Plot_Dir, exist_ok=True)

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def get_dataset_path():
    for path in LOCAL_SEARCH_PATHS:
       if os.path.exists(path): 
        print(f"Found local dataset copy: {path}")
        return path
    print ("No local CSV found — attempting Kaggle auto-download via kagglehub ...")

    try:
       dataset_dir = kagglehub.dataset_download(KAGGELE_DATASET)
       csv_matches = glob.glob(os.path.join(dataset_dir, "*.csv"))
       if not csv_matches:
           raise FileNotFoundError(F"No CSV found in kaggle dataset.")
       print(f"Downloaded to: {dataset_dir}")
       return csv_matches[0]
    except Exception as e:
       raise FileNotFoundError(
            f"Could not find a local CSV and Kaggle auto-download failed ({e}). "
            f"Either place '{CSV_FILENAME}' in this folder / ./data/, or set up "
            "a Kaggle access token (see get_dataset_path docstring)."
        )

Data_path = get_dataset_path()

# ---------------------------------------------------------------------------
# TASK 1: Load data
# ---------------------------------------------------------------------------

section("Task1 : Load data")
df = pd.read_csv(Data_path)

print("First 5 rows of the dataset:")
print(df.head(5))

print("\nColums dtypes:")
print(df.dtypes)

print(f"\nDataframe shape :{df.shape}")

# ---------------------------------------------------------------------------
# TASK 2: Null value analysis
# ---------------------------------------------------------------------------

section("Task2 : Null value analysis")
null_count = df.isnull().sum()
null_pct = (null_count / df.shape[0]) * 100

null_table = pd.DataFrame({"null_count": null_count, "null_pct" : null_pct})
print(null_table.columns)
null_table = null_table.sort_values("null_pct", ascending=False)
print(null_table)


high_null_cols = null_table[null_table["null_pct"]>20].index.tolist()
print(f"\ncolumns with >20% null values:{high_null_cols if high_null_cols else 'None'}")

# Fill numeric columns below 20% null threshold with the median
numeric_cols_all = df.select_dtypes(include =[np.number]).columns.tolist()
for col in numeric_cols_all:
   if col in high_null_cols and df[col].isnull().sum()>0:
      median_val = df[col].median()
      df[col] = df[col].fillna(median_val)
      print(f"filled column '{col}' with median value {median_val}")

# ---------------------------------------------------------------------------
# TASK 3: Duplicate detection and removal
# ---------------------------------------------------------------------------

section("TASK 3: Duplicate detection and removal")
dup_count = df.duplicated().sum()
print(f"Duplicate rows found :{dup_count}")

null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
df = df.drop_duplicates()
null_pct_after = (df.isnull().sum()/ df.shape[0])*100

print(f"Rows removed :{dup_count}")
print(f"shape after dup removed : {df.shape}")
changed = (null_pct_before.reindex(null_pct_after.index) - null_pct_after).abs().sum()
print(f"Total change in null percentage after dup removed : {changed:.4f}")

# ---------------------------------------------------------------------------
# TASK 4: Data type correction
# ---------------------------------------------------------------------------

section("TASK 4: Data type correction")
mem_before = df.memory_usage(deep=True).sum()
print(f"Memory usage before conversion: {mem_before/1024:.2f} KB")

for col in df.columns:
    if df[col].dtype == object:
       coerced = pd.to_numeric(df[col],errors='coerce')
       if coerced.notnull().sum() == df[col].notnull().sum() and coerced.notnull().any():
          df[col] = coerced
          print(f"Converted numeric-looking object column '{col}' to numeric.")


categorical_candidates = [
    c for c in df.select_dtypes(include="object").columns
    if df[c].nunique() < 30
]

for col in categorical_candidates:
   df[col] = df[col].astype("category")

print(f"\nConverted to category dtype: {categorical_candidates}")

mem_after = df.memory_usage(deep=True).sum()
print(f"\nMemory usage AFTER conversion: {mem_after / 1024:.2f} KB")
print(f"Memory reduction: {(mem_before - mem_after) / 1024:.2f} KB "
      f"({(1 - mem_after / mem_before) * 100:.1f}% smaller)")

# ---------------------------------------------------------------------------
# TASK 5: Descriptive statistics and skewness
# ---------------------------------------------------------------------------

section("TASK 5: Descriptive statistics and skewness")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].describe())

skew_vals = df[numeric_cols].skew().sort_values(key= lambda s:s.abs(),ascending=False)
print("\nSkewness per numeric column (sorted by |skew|):")
print(skew_vals)

top_skew_col = skew_vals.index[0]
print(f"\nMost skewed column: '{top_skew_col}' (skew = {skew_vals.iloc[0]:.3f})")

# ---------------------------------------------------------------------------
# TASK 6: Outlier detection with IQR
# ---------------------------------------------------------------------------

section("TASK 6: Outlier detection with IQR")

outlier_check_cols = [top_skew_col]

second_col_candidates = [c for c in numeric_cols if c != top_skew_col and df[c].nunique() > 5]
outlier_check_cols.append(second_col_candidates[0])

outlier_summary = {}
for col in outlier_check_cols:
   Q1 = df[col].quantile(0.25)
   Q3 = df[col].quantile(0.75)
   IQR = Q3 - Q1
   lower = Q1 - 1.5 * IQR
   upper = Q3 + 1.5 * IQR
   n_outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0] 

   outlier_summary = {
      "Q1": Q1,
      "Q3": Q3,
      "IQR": IQR,
      "Lower": lower,
      "Upper": upper,
      "n_outliers": n_outliers,
   }

   print(f"\nColumn: {col}")
   print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
   print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
   print(f"  Outlier rows: {n_outliers}")   

# ---------------------------------------------------------------------------
# TASK 7: Visualizations
# ---------------------------------------------------------------------------

section("TASK 7: Visualizations")
num_cols_for_line = "MonthlyIncome" if "MonthlyIncome" in df.columns else numeric_cols[0]
cat_col_for_bar = "Department" if "Department" in df.columns else df.select_dtypes(include="category").columns[0]
num_cols_for_bar = "MonthlyIncome" if "MonthlyIncome" in df.columns else numeric_cols[0]
scatter_x, scatter_y = (
   ("YearsAtCompany","MonthlyIncome")
   if {"YearsAtCompany","MonthlyIncome"}.issubset(df.columns)
   else (numeric_cols[0], numeric_cols[1])
)
box_num_col = "MonthlyIncome" if "MonthlyIncome" in df.columns else numeric_cols[0]
box_cat_col = "Attrition" if "Attrition" in df.columns else df.select_dtypes(include="category").columns[0]

#line plot
plt.figure()
plt.plot(df[num_cols_for_line].reset_index(drop=True))
plt.title(f"Line Plot of {num_cols_for_line} by Row Index")
plt.xlabel("Row Index")
plt.ylabel(num_cols_for_line)
plt.tight_layout()
plt.savefig(f"plots/1_line_plot.png")
plt.close()
print(f"Saved line plot: plots/1_line_plot.png")

# 2. Bar chart: mean of numeric col across categories
plt.figure()
df.groupby(cat_col_for_bar, observed=True)[num_cols_for_bar].mean().plot.bar()
plt.title(f"Mean of {num_cols_for_bar} by {cat_col_for_bar}")
plt.xlabel(cat_col_for_bar)
plt.ylabel(f"Mean of {num_cols_for_bar}")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"plots/2_bar_chart.png")
plt.close()
print(f"Saved bar chart: plots/2_bar_chart.png")

# 3. Histogram of most skewed column
plt.figure()
sns.histplot(df[top_skew_col], bins=20, kde=True)
plt.title(f"Histogram of Most Skewed Column: {top_skew_col}")
plt.xlabel(top_skew_col)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"plots/3_histogram.png")
plt.close()
print(f"Saved histogram: plots/3_histogram.png")

# 4. Scatter plot
plt.figure()
sns.scatterplot(data=df,x=scatter_x, y=scatter_y)
plt.title(f"Scatter Plot: {scatter_x} vs {scatter_y}")
plt.xlabel(scatter_x)
plt.ylabel(scatter_y)
plt.tight_layout()
plt.savefig(f"plots/4_scatter_plot.png")
plt.close()
print(f"Saved scatter plot: plots/4_scatter_plot.png")

# 5. Box plot
plt.figure()
sns.boxplot(data=df, x=box_cat_col, y=box_num_col)
plt.title(f"Box Plot: {box_num_col} by {box_cat_col}")
plt.xlabel(box_cat_col)
plt.ylabel(box_num_col)
plt.tight_layout()
plt.savefig(f"plots/5_box_plot.png")
plt.close()
print(f"Saved box plot: plots/5_box_plot.png")

# 6. Correlation heat map
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(14, 11))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size":10})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"plots/6_correlation_heatmap.png")
plt.close()
print(f"Saved correlation heatmap: plots/6_correlation_heatmap.png")

corr_unstacked = corr_matrix.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1.0]
top_pair = corr_unstacked.sort_values(ascending=False).index[0]
top_corr_val = corr_matrix.loc[top_pair[0], top_pair[1]]
print(f"\nHighest |correlation| pair: {top_pair} = {top_corr_val:.3f}")

# ---------------------------------------------------------------------------
# TASK 8a: Imputation strategy comparison (mean vs median) on top-2 skewed cols
# ---------------------------------------------------------------------------

section("TASK 8a: Imputation strategy comparison (mean vs median) on top-2 skewed cols")
top2_skew_cols = skew_vals.index[:2].to_list()

for col in top2_skew_cols:
   mean = df[col].mean()
   median = df[col].median()
   print(f"Column: {col}, Mean: {mean:.2f}, Median: {median:.2f}"
         f"| skew = {skew_vals[col]:.3f}")

# Apply chosen strategy: median (robust to skew) for any remaining nulls
if df[col].isnull().sum() > 0:
      df[col] = df[col].fillna(median)
print(f"  -> Nulls remaining in '{col}' after imputation: {df[col].isnull().sum()}")

 
# ---------------------------------------------------------------------------
# TASK 8b: Spearman vs Pearson correlation
# ---------------------------------------------------------------------------
section("TASK 8b: SPEARMAN VS PEARSON CORRELATION")
pearson_corr = df[numeric_cols].corr(method="pearson")
spearman_corr = df[numeric_cols].corr(method="spearman")
 
diff = (spearman_corr - pearson_corr).abs()
diff_unstacked = diff.unstack()
diff_unstacked = diff_unstacked[diff_unstacked.index.get_level_values(0)
                                 != diff_unstacked.index.get_level_values(1)]
# each pair appears twice (a,b) and (b,a) -> dedupe
diff_unstacked = diff_unstacked.reset_index()
diff_unstacked.columns = ["col1", "col2", "abs_diff"]
diff_unstacked["pair_key"] = diff_unstacked.apply(
    lambda r: tuple(sorted([r["col1"], r["col2"]])), axis=1
)
diff_unstacked = diff_unstacked.drop_duplicates("pair_key")
top3_diff_pairs = diff_unstacked.sort_values("abs_diff", ascending=False).head(3)
 
print("Pearson correlation matrix (head):")
print(pearson_corr.iloc[:5, :5])
print("\nSpearman correlation matrix (head):")
print(spearman_corr.iloc[:5, :5])
 
print("\nTop 3 pairs with largest |Spearman - Pearson| difference:")
print(top3_diff_pairs[["col1", "col2", "abs_diff"]])

# ---------------------------------------------------------------------------
# TASK 8c: Grouped aggregation
# ---------------------------------------------------------------------------
section("TASK 8c: GROUPED AGGREGATION")
group_cat_col = "JobRole" if "JobRole" in df.columns else box_cat_col
group_num_col = "MonthlyIncome" if "MonthlyIncome" in df.columns else numeric_cols[0]
 
grouped_stats = df.groupby(group_cat_col, observed=True)[group_num_col].agg(
    ["mean", "std", "count"]
)
print(grouped_stats)
 
highest_mean_group = grouped_stats["mean"].idxmax()
highest_std_group = grouped_stats["std"].idxmax()
mean_ratio = grouped_stats["mean"].max() / grouped_stats["mean"].min()
 
print(f"\nGroup with highest mean {group_num_col}: {highest_mean_group}")
print(f"Group with highest std {group_num_col}: {highest_std_group}")
print(f"Ratio of highest group mean to lowest group mean: {mean_ratio:.2f}")

# ---------------------------------------------------------------------------
# TASK 9: Save cleaned dataset
# ---------------------------------------------------------------------------
section("TASK 9: SAVE CLEANED DATASET")

df.to_csv("Cleaned_data.CSV", index=False)
print("cleaned Csv saved - shape:", df.shape)
