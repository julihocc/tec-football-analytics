# Chapter 2: Python code
# Copyright: Clive Beggs 6th March 2023 (Python translation May 31, 2025)
#
# This script demonstrates basic data analysis and visualization techniques using Python.
# It is organized into sections (cells) for interactive execution in VS Code or Jupyter-like environments.
# Each section is documented with comments explaining its purpose and logic.

# %%
# Import required libraries for data manipulation, visualization, and statistics
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
# Section 1: Basic Data Creation and Summary Statistics (Example 2.1)
# Create match data and compute basic statistics for goals scored, conceded, and goal difference
match_id = list(range(1, 11))
print("Match IDs:", match_id)

goals_for = [0, 2, 4, 1, 3, 0, 2, 2, 3, 1]
print("Goals For:", goals_for)

goals_against = [1, 1, 3, 3, 0, 0, 1, 1, 1, 0]
print("Goals Against:", goals_against)

goal_diff = [x - y for x, y in zip(goals_for, goals_against)]
print("Goal Difference:", goal_diff)

# Goals scored statistics
print("\nGoals scored statistics:")
print(f"Mean: {np.mean(goals_for)}")
print(f"Median: {np.median(goals_for)}")
print(f"Standard deviation: {np.std(goals_for, ddof=1)}")  # ddof=1 for sample std
print(f"Variance: {np.var(goals_for, ddof=1)}")  # ddof=1 for sample variance

# Goals conceded statistics
print("\nGoals conceded statistics:")
print(f"Mean: {np.mean(goals_against):.3f}")
print(f"Median: {np.median(goals_against):.3f}")
print(f"Standard deviation: {np.std(goals_against, ddof=1):.3f}")
print(f"Variance: {np.var(goals_against, ddof=1):.3f}")

# Goal difference statistics
print("\nGoal difference statistics:")
print(f"Mean: {np.mean(goal_diff):.3f}")
print(f"Median: {np.median(goal_diff):.3f}")
print(f"Standard deviation: {np.std(goal_diff, ddof=1):.3f}")
print(f"Variance: {np.var(goal_diff, ddof=1):.3f}")

# Compile DataFrame
goals_dat = pd.DataFrame({
    'MatchID': match_id,
    'GoalsFor': goals_for,
    'GoalsAgainst': goals_against,
    'GoalDiff': goal_diff
})
print("\nComplete dataset:")
print(goals_dat)

# %%
# Section 2: DataFrame Operations and Access (Example 2.2)
# Explore DataFrame structure, access columns and rows, and display summary info
print("\nColumn names:")
print(goals_dat.columns)

print("\nFirst few rows:")
print(goals_dat.head())

print("\nFirst 8 rows:")
print(goals_dat.head(8))

print("\nLast few rows:")
print(goals_dat.tail())

print("\nDataFrame dimensions:")
print(f"Number of rows: {len(goals_dat)}")
print(f"Number of columns: {len(goals_dat.columns)}")
print(f"Shape: {goals_dat.shape}")

print("\nDataFrame info:")
print(goals_dat.info())

# Different ways to access data
print("\nAccessing GoalsFor column:")
print(goals_dat['GoalsFor'])  # Method 1: Using column name
print(goals_dat.iloc[:, 1])   # Method 2: Using integer location

print("\nAccessing GoalsFor and GoalsAgainst columns:")
print(goals_dat[['GoalsFor', 'GoalsAgainst']])

print("\nAccessing rows 3-5:")
print(goals_dat.iloc[2:5])

# %%
# Section 3: Conditional Logic and Adding Columns (Example 2.3)
# Use conditional logic to determine match outcomes and add results to the DataFrame
# Method 1: Using numpy.where (equivalent to R's ifelse)
outcome1 = np.where(goals_dat['GoalsFor'] > goals_dat['GoalsAgainst'], 
                   "Win", "Did not win")
print("\nOutcome method 1:")
print(outcome1)

# Method 2: Using list comprehension (equivalent to R's for loop)
def get_result(row):
    if row['GoalsFor'] > row['GoalsAgainst']:
        return "Win"
    elif row['GoalsFor'] < row['GoalsAgainst']:
        return "Lose"
    else:
        return "Draw"

outcome2 = [get_result(row) for _, row in goals_dat.iterrows()]
print("\nOutcome method 2:")
print(outcome2)

# Add results to DataFrame
match_dat = goals_dat.copy()
match_dat['Result'] = outcome2
print("\nUpdated DataFrame with results:")
print(match_dat)

# Find all matches where the team won
win_results = match_dat[match_dat['Result'] == "Win"]
print("\nMatches won:")
print(win_results)

# %%
# Section 4: Descriptive Statistics (Example 2.4)
# Generate summary and detailed statistics for the match data
print("\nDataFrame summary:")
print(match_dat.describe())

# More detailed statistics (similar to R's psych package)
def describe_detailed(df):
    stats_df = pd.DataFrame({
        'n': df.count(),
        'mean': df.mean(),
        'sd': df.std(),
        'median': df.median(),
        'trimmed': df.apply(lambda x: stats.trim_mean(x, 0.1)),
        'mad': df.apply(lambda x: stats.median_abs_deviation(x)),
        'min': df.min(),
        'max': df.max(),
        'range': df.max() - df.min(),
        'skew': df.skew(),
        'kurtosis': df.kurtosis()
    })
    return stats_df.round(3)

# Calculate detailed statistics for numeric columns
numeric_cols = match_dat.select_dtypes(include=[np.number])
detailed_stats = describe_detailed(numeric_cols)
print("\nDetailed statistics:")
print(detailed_stats)

# %%
# Section 5: Reading and Summarizing External Data (Example 2.5)
# Read Arsenal-Chelsea comparison data and compute detailed statistics
data_path = Path("../data/Arsenal_Chelsea_comparison.csv")  # Use correct relative path from Chapter02 folder
if not data_path.exists():
    data_path = Path("data/Arsenal_Chelsea_comparison.csv")  # fallback if running from project root
dat = pd.read_csv(data_path)
print("\nArsenal-Chelsea comparison data:")
print(dat)

# Calculate detailed statistics
numeric_cols = dat.select_dtypes(include=[np.number])
detailed_stats = describe_detailed(numeric_cols)
print("\nDetailed statistics:")
print(detailed_stats)

# Export results if needed
# detailed_stats.to_csv('descriptive_results.csv')

# %%
# Section 6: Visualization - Line Plots (Example 2.6)
# Plot goals for and against for Arsenal and Chelsea across seasons
seasons = list(range(2011, 2021))

# Ensure dat columns are numpy arrays for plotting
arsenal_gf = np.array(dat['Arsenal_GF'])
chelsea_gf = np.array(dat['Chelsea_GF'])
arsenal_ga = np.array(dat['Arsenal_GA'])
chelsea_ga = np.array(dat['Chelsea_GA'])

plt.figure(figsize=(10, 6))
plt.plot(seasons, arsenal_gf, 'ko-', label='Arsenal goals for')
plt.plot(seasons, chelsea_gf, 'ko--', label='Chelsea goals for')
plt.plot(seasons, arsenal_ga, 'k^-', label='Arsenal goals against')
plt.plot(seasons, chelsea_ga, 'k^--', label='Chelsea goals against')

plt.ylim(0, 140)
plt.xlabel('Season')
plt.ylabel('Goals')
plt.title('Arsenal and Chelsea comparison')
plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.show()

# %%
# Section 7: Visualization - Box Plots (Example 2.7)
# Create box plots to compare goals for and against for both teams
plt.figure(figsize=(10, 6))
goals_data = [dat['Arsenal_GF'], dat['Arsenal_GA'], 
              dat['Chelsea_GF'], dat['Chelsea_GA']]
plt.boxplot(goals_data, labels=['Arsenal GF', 'Arsenal GA', 
                               'Chelsea GF', 'Chelsea GA'])
plt.ylabel('Goals')
plt.title('Goals Distribution Comparison')
plt.show()

# Summary statistics
print("\nSummary statistics:")
print(dat[['Arsenal_GF', 'Arsenal_GA', 'Chelsea_GF', 'Chelsea_GA']].describe())

# %%
# Section 8: Visualization - Scatter and Regression Plots (Example 2.8)
# Scatter plots and regression lines for goals conceded vs points for both teams
plt.figure(figsize=(10, 6))
plt.scatter(dat['Chelsea_GA'], dat['Chelsea_points'], marker='o', 
           color='black', label='Chelsea goals conceded')
plt.scatter(dat['Arsenal_GA'], dat['Arsenal_points'], marker='^', 
           color='black', label='Arsenal goals conceded')

# Regression lines
arsenal_fit = np.polyfit(dat['Arsenal_GA'], dat['Arsenal_points'], 1)
chelsea_fit = np.polyfit(dat['Chelsea_GA'], dat['Chelsea_points'], 1)

x_range = np.array([0, 60])
plt.plot(x_range, np.polyval(arsenal_fit, x_range), 'k--', 
         label='Arsenal bestfit line')
plt.plot(x_range, np.polyval(chelsea_fit, x_range), 'k-', 
         label='Chelsea bestfit line')

plt.xlim(0, 60)
plt.ylim(0, 100)
plt.xlabel('Goals conceded')
plt.ylabel('Points')
plt.legend(bbox_to_anchor=(0.05, 0.2), loc='lower left', frameon=False)
plt.tight_layout()
plt.show()

# %%
# Section 9: Statistical Tests - t-tests and Correlations (Example 2.9)
# Perform paired t-tests and Pearson correlation tests for goals and points
# Paired t-tests
gf_ttest = stats.ttest_rel(dat['Arsenal_GF'], dat['Chelsea_GF'])
ga_ttest = stats.ttest_rel(dat['Arsenal_GA'], dat['Chelsea_GA'])

print("\nPaired t-test results:")
print("Goals For - Arsenal vs Chelsea:")
print(f"t-statistic: {gf_ttest.statistic:.4f}")
print(f"p-value: {gf_ttest.pvalue:.4f}")

print("\nGoals Against - Arsenal vs Chelsea:")
print(f"t-statistic: {ga_ttest.statistic:.4f}")
print(f"p-value: {ga_ttest.pvalue:.4f}")

# Pearson correlation tests
arsenal_corr = stats.pearsonr(dat['Arsenal_GA'], dat['Arsenal_points'])
chelsea_corr = stats.pearsonr(dat['Chelsea_GA'], dat['Chelsea_points'])

print("\nPearson correlation test results:")
print("Arsenal GA vs Points:")
print(f"correlation: {arsenal_corr[0]:.4f}")
print(f"p-value: {arsenal_corr[1]:.4f}")

print("\nChelsea GA vs Points:")
print(f"correlation: {chelsea_corr[0]:.4f}")
print(f"p-value: {chelsea_corr[1]:.4f}")

# %%
# Section 10: Regression Modeling and Prediction (Example 2.10)
# Build linear regression models for Arsenal and Chelsea, make predictions, and plot results
from sklearn.linear_model import LinearRegression

# Arsenal model
X_arsenal = dat[['Arsenal_GA', 'Arsenal_GF']]
y_arsenal = dat['Arsenal_points']
arsenal_model = LinearRegression()
arsenal_model.fit(X_arsenal, y_arsenal)

# Chelsea model
X_chelsea = dat[['Chelsea_GA', 'Chelsea_GF']]
y_chelsea = dat['Chelsea_points']
chelsea_model = LinearRegression()
chelsea_model.fit(X_chelsea, y_chelsea)

# Print model summaries
print("\nArsenal Model Summary:")
print(f"R² Score: {arsenal_model.score(X_arsenal, y_arsenal):.4f}")
print("Coefficients:")
print(f"Goals Against: {arsenal_model.coef_[0]:.4f}")
print(f"Goals For: {arsenal_model.coef_[1]:.4f}")
print(f"Intercept: {arsenal_model.intercept_:.4f}")

print("\nChelsea Model Summary:")
print(f"R² Score: {chelsea_model.score(X_chelsea, y_chelsea):.4f}")
print("Coefficients:")
print(f"Goals Against: {chelsea_model.coef_[0]:.4f}")
print(f"Goals For: {chelsea_model.coef_[1]:.4f}")
print(f"Intercept: {chelsea_model.intercept_:.4f}")

# Making predictions
arsenal_pred = arsenal_model.predict(X_arsenal)
chelsea_pred = chelsea_model.predict(X_chelsea)

# Create DataFrame with predictions
new_dat = dat.copy()
new_dat['Arsenal_pred'] = np.round(arsenal_pred, 1)
new_dat['Chelsea_pred'] = np.round(chelsea_pred, 1)
print("\nData with predictions:")
print(new_dat)

# Plotting predictions vs observed values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Arsenal plot
ax1.scatter(arsenal_pred, dat['Arsenal_points'], color='black', s=30)
ax1.plot([dat['Arsenal_points'].min(), dat['Arsenal_points'].max()],
         [dat['Arsenal_points'].min(), dat['Arsenal_points'].max()],
         'k--', alpha=0.5)
ax1.set_xlabel('Predicted points')
ax1.set_ylabel('Observed points')
ax1.set_title('Arsenal: Predicted vs Observed Points')

# Chelsea plot
ax2.scatter(chelsea_pred, dat['Chelsea_points'], color='black', s=30)
ax2.plot([dat['Chelsea_points'].min(), dat['Chelsea_points'].max()],
         [dat['Chelsea_points'].min(), dat['Chelsea_points'].max()],
         'k--', alpha=0.5)
ax2.set_xlabel('Predicted points')
ax2.set_ylabel('Observed points')
ax2.set_title('Chelsea: Predicted vs Observed Points')

plt.tight_layout()
plt.show()
