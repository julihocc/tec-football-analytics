# Chapter 2: Python code (ggplot version)
# Copyright: Clive Beggs 6th March 2023 (Python translation May 31, 2025)
#
# This script demonstrates basic data analysis and visualization techniques using Python, using plotnine (ggplot for Python) for all visualizations.
# It is organized into sections (cells) for interactive execution in VS Code or Jupyter-like environments.
# Each section is documented with comments explaining its purpose and logic.

# %%
# Import required libraries for data manipulation, visualization, and statistics
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from plotnine import *

# %%
# Section 1: Basic Data Creation and Summary Statistics (Example 2.1)
match_id = list(range(1, 11))
goals_for = [0, 2, 4, 1, 3, 0, 2, 2, 3, 1]
goals_against = [1, 1, 3, 3, 0, 0, 1, 1, 1, 0]
goal_diff = [x - y for x, y in zip(goals_for, goals_against)]

goals_dat = pd.DataFrame({
    'MatchID': match_id,
    'GoalsFor': goals_for,
    'GoalsAgainst': goals_against,
    'GoalDiff': goal_diff
})
print(goals_dat)

# %%
# Section 2: DataFrame Operations and Access (Example 2.2)
print(goals_dat.columns)
print(goals_dat.head())
print(goals_dat.tail())
print(goals_dat.shape)
print(goals_dat.info())

# %%
# Section 3: Conditional Logic and Adding Columns (Example 2.3)
def get_result(row):
    if row['GoalsFor'] > row['GoalsAgainst']:
        return "Win"
    elif row['GoalsFor'] < row['GoalsAgainst']:
        return "Lose"
    else:
        return "Draw"

goals_dat['Result'] = goals_dat.apply(get_result, axis=1)
print(goals_dat)

# %%
# Section 4: Descriptive Statistics (Example 2.4)
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

print(describe_detailed(goals_dat.select_dtypes(include=[np.number])))

# %%
# Section 5: Reading and Summarizing External Data (Example 2.5)
data_path = Path("../data/Arsenal_Chelsea_comparison.csv")
if not data_path.exists():
    data_path = Path("data/Arsenal_Chelsea_comparison.csv")
dat = pd.read_csv(data_path)
print(dat)
print(describe_detailed(dat.select_dtypes(include=[np.number])))

# %%
# Section 6: Visualization - Line Plots (Example 2.6, ggplot version)
dat_long = pd.melt(dat, id_vars=['Season'], value_vars=['Arsenal_GF', 'Chelsea_GF', 'Arsenal_GA', 'Chelsea_GA'],
                   var_name='TeamStat', value_name='Goals')
dat_long['Type'] = dat_long['TeamStat'].apply(lambda x: 'GF' if 'GF' in x else 'GA')
dat_long['Team'] = dat_long['TeamStat'].apply(lambda x: 'Arsenal' if 'Arsenal' in x else 'Chelsea')

p = (
    ggplot(dat_long, aes(x='Season', y='Goals', color='Team', linetype='Type', group='TeamStat')) +
    geom_line(size=1) +
    geom_point() +
    labs(title='Arsenal and Chelsea comparison', x='Season', y='Goals') +
    theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(10, 6))
)
print(p)

# %%
# Section 7: Visualization - Box Plots (Example 2.7, ggplot version)
dat_box = pd.melt(dat, value_vars=['Arsenal_GF', 'Arsenal_GA', 'Chelsea_GF', 'Chelsea_GA'],
                  var_name='Stat', value_name='Goals')
p_box = (
    ggplot(dat_box, aes(x='Stat', y='Goals')) +
    geom_boxplot() +
    labs(title='Goals Distribution Comparison', y='Goals') +
    theme(figure_size=(10, 6))
)
print(p_box)

# %%
# Section 8: Visualization - Scatter and Regression Plots (Example 2.8, ggplot version)
dat_scatter = pd.DataFrame({
    'Arsenal_GA': dat['Arsenal_GA'],
    'Arsenal_points': dat['Arsenal_points'],
    'Chelsea_GA': dat['Chelsea_GA'],
    'Chelsea_points': dat['Chelsea_points']
})

dat_arsenal = dat_scatter[['Arsenal_GA', 'Arsenal_points']].rename(columns={'Arsenal_GA': 'GA', 'Arsenal_points': 'Points'})
dat_arsenal['Team'] = 'Arsenal'
dat_chelsea = dat_scatter[['Chelsea_GA', 'Chelsea_points']].rename(columns={'Chelsea_GA': 'GA', 'Chelsea_points': 'Points'})
dat_chelsea['Team'] = 'Chelsea'
dat_combined = pd.concat([dat_arsenal, dat_chelsea], ignore_index=True)

p_scatter = (
    ggplot(dat_combined, aes(x='GA', y='Points', shape='Team', color='Team')) +
    geom_point(size=3) +
    geom_smooth(method='lm', se=False, linetype='dashed') +
    labs(title='Goals Conceded vs Points', x='Goals conceded', y='Points') +
    theme(figure_size=(10, 6))
)
print(p_scatter)

# %%
# Section 9: Statistical Tests - t-tests and Correlations (Example 2.9)
gf_ttest = stats.ttest_rel(dat['Arsenal_GF'], dat['Chelsea_GF'])
ga_ttest = stats.ttest_rel(dat['Arsenal_GA'], dat['Chelsea_GA'])
print("\nPaired t-test results:")
print("Goals For - Arsenal vs Chelsea:")
print(f"t-statistic: {gf_ttest.statistic:.4f}")
print(f"p-value: {gf_ttest.pvalue:.4f}")
print("\nGoals Against - Arsenal vs Chelsea:")
print(f"t-statistic: {ga_ttest.statistic:.4f}")
print(f"p-value: {ga_ttest.pvalue:.4f}")
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
from sklearn.linear_model import LinearRegression
X_arsenal = dat[['Arsenal_GA', 'Arsenal_GF']]
y_arsenal = dat['Arsenal_points']
arsenal_model = LinearRegression()
arsenal_model.fit(X_arsenal, y_arsenal)
X_chelsea = dat[['Chelsea_GA', 'Chelsea_GF']]
y_chelsea = dat['Chelsea_points']
chelsea_model = LinearRegression()
chelsea_model.fit(X_chelsea, y_chelsea)
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
arsenal_pred = arsenal_model.predict(X_arsenal)
chelsea_pred = chelsea_model.predict(X_chelsea)
new_dat = dat.copy()
new_dat['Arsenal_pred'] = np.round(arsenal_pred, 1)
new_dat['Chelsea_pred'] = np.round(chelsea_pred, 1)
print("\nData with predictions:")
print(new_dat)

# Plotting predictions vs observed values using plotnine
dat_pred_arsenal = pd.DataFrame({'Predicted': arsenal_pred, 'Observed': dat['Arsenal_points'], 'Team': 'Arsenal'})
dat_pred_chelsea = pd.DataFrame({'Predicted': chelsea_pred, 'Observed': dat['Chelsea_points'], 'Team': 'Chelsea'})
dat_pred = pd.concat([dat_pred_arsenal, dat_pred_chelsea], ignore_index=True)
p_pred = (
    ggplot(dat_pred, aes(x='Predicted', y='Observed', color='Team')) +
    geom_point(size=3) +
    geom_abline(linetype='dashed') +
    labs(title='Predicted vs Observed Points', x='Predicted points', y='Observed points') +
    theme(figure_size=(10, 5))
)
print(p_pred)
