# %%
# Chapter 11: Python code
# Copyright: Clive Beggs - 6th April 2023
# Converted from R to Python - Data Manipulation, Random Forest, and Statistical Power Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHAPTER 11: ADVANCED STATISTICAL ANALYSIS")
print("="*60)
print("This chapter demonstrates:")
print("- Data manipulation and factor handling")
print("- Random Forest machine learning")
print("- Statistical power analysis and simulation")
print("="*60)

# %%
# Example 11.1: Data Manipulation and Factors
print("EXAMPLE 11.1: Data Manipulation and Factor Handling")
print("="*50)

# Create data
players = ["Peter","Jane","John","Paul","Anne","Sarah","Lucy","Tom","Sean","David"]
sex = ["Male","Female","Male","Male","Female","Female","Female","Male","Male","Male"]
sex_id = [1, 2, 1, 1, 2, 2, 2, 1, 1, 1]  # Numerical classifier of gender
age = [19, 21, 22, 24, 19, 21, 27, 25, 20, 18]
height = [1.85, 1.64, 1.76, 1.83, 1.62, 1.57, 1.69, 1.80, 1.75, 1.81]
weight = [80.4, 67.1, 75.4, 81.2, 65.2, 63.7, 66.3, 77.5, 73.4, 81.2]

# Compute BMI and round to 2 decimal places
bmi = np.round(np.array(weight) / (np.array(height)**2), 2)

# Create DataFrame
dat = pd.DataFrame({
    'Player': players,
    'Sex': sex,
    'SexID': sex_id,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'BMI': bmi
})

print("Initial DataFrame:")
print(dat)

# %%
# Inspect data
print("\nDataFrame info:")
print(dat.info())

print("\nDataFrame description:")
print(dat.describe())

print("\nData types:")
print(dat.dtypes)

# %%
# Convert to categorical (factor equivalent)
print("Converting Sex and SexID to categorical variables...")

# Create categorical variables (equivalent to R factors)
dat['Sex'] = pd.Categorical(dat['Sex'])
dat['SexID'] = pd.Categorical(dat['SexID'])

print("\nData types after conversion:")
print(dat.dtypes)

print("\nUpdated DataFrame description:")
print(dat.describe(include='all'))

# %%
# Display categorical summaries
print("\nSex categories:")
print(dat['Sex'].value_counts())

print("\nSexID categories:")
print(dat['SexID'].value_counts())

print("\nCategorical variable details:")
print(f"Sex categories: {dat['Sex'].cat.categories.tolist()}")
print(f"SexID categories: {dat['SexID'].cat.categories.tolist()}")

# %%
# Example 11.2: Random Forest Analysis
print("\nEXAMPLE 11.2: Random Forest Machine Learning")
print("="*50)

# Load data
print("Loading EPL regression data...")
mydata = pd.read_csv('../data/EPL_regression_data_2020_2021.csv')
rfdata = mydata.iloc[:, 3:14].copy()  # Columns 4-14 (0-indexed: 3-13)

print("Data loaded successfully!")
print(f"Dataset shape: {rfdata.shape}")
print("\nFirst 6 rows:")
print(rfdata.head(6))

print(f"\nColumns for Random Forest analysis:")
print(rfdata.columns.tolist())

# %%
# Build Random Forest model (full model)
print("Building Random Forest models...")

# Prepare features and target
X = rfdata.drop('Points', axis=1)
y = rfdata['Points']

print(f"Features: {X.columns.tolist()}")
print(f"Target: Points")

# Random Forest with all predictors
np.random.seed(123)  # Set seed for reproducibility
rf_mod1 = RandomForestRegressor(n_estimators=500, random_state=123)
rf_mod1.fit(X, y)

print(f"\nRandom Forest Model 1 (Full Model):")
print(f"Number of features: {rf_mod1.n_features_in_}")
print(f"Number of trees: {rf_mod1.n_estimators}")
print(f"Out-of-bag score: {rf_mod1.oob_score_:.4f}" if hasattr(rf_mod1, 'oob_score_') else "OOB score not available")

# %%
# Compute feature importance
print("\nFeature importance analysis:")

importance_scores = rf_mod1.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print("Feature importance (sorted):")
print(feature_importance.round(4))

# %%
# Visualize feature importance (equivalent to varImpPlot)
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['Importance'])
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%
# Build refined model with top 4 features
print("Building refined Random Forest model...")

# Select top 4 features based on importance
top_features = feature_importance.head(4)['Feature'].tolist()
print(f"Top 4 features selected: {top_features}")

X_refined = rfdata[top_features]

# Build refined model
np.random.seed(123)
rf_mod2 = RandomForestRegressor(n_estimators=500, random_state=123)
rf_mod2.fit(X_refined, y)

print(f"\nRandom Forest Model 2 (Refined Model):")
print(f"Selected features: {top_features}")
print(f"Number of features: {rf_mod2.n_features_in_}")

# %%
# Predictions using full model
print("Making predictions with full model...")

mod1_pred = rf_mod1.predict(X)
obs_mod1 = y
mod1_dat = pd.DataFrame({
    'Observed': obs_mod1,
    'Predicted': mod1_pred
})

# Calculate metrics
mod1_r2 = r2_score(obs_mod1, mod1_pred)
mod1_mae = mean_absolute_error(obs_mod1, mod1_pred)

print(f"Full Model Performance:")
print(f"R²: {mod1_r2:.4f}")
print(f"Mean Absolute Error: {mod1_mae:.4f}")

# %%
# Scatter plot for full model
plt.figure(figsize=(8, 6))
plt.scatter(mod1_pred, obs_mod1, alpha=0.7, edgecolors='black')
plt.plot([obs_mod1.min(), obs_mod1.max()], [obs_mod1.min(), obs_mod1.max()], 'r--', alpha=0.8)
plt.xlabel('Predicted Points')
plt.ylabel('Observed Points')
plt.title(f'Full Random Forest Model\nR² = {mod1_r2:.4f}, MAE = {mod1_mae:.4f}')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(mod1_pred, obs_mod1, 1)
p = np.poly1d(z)
plt.plot(mod1_pred, p(mod1_pred), "b-", alpha=0.8, linewidth=2)
plt.show()

print(f"Mean Absolute Error (Full Model): {mod1_mae:.4f}")

# %%
# Predictions using refined model
print("Making predictions with refined model...")

mod2_pred = rf_mod2.predict(X_refined)
obs_mod2 = y
mod2_dat = pd.DataFrame({
    'Observed': obs_mod2,
    'Predicted': mod2_pred
})

# Calculate metrics
mod2_r2 = r2_score(obs_mod2, mod2_pred)
mod2_mae = mean_absolute_error(obs_mod2, mod2_pred)

print(f"Refined Model Performance:")
print(f"R²: {mod2_r2:.4f}")
print(f"Mean Absolute Error: {mod2_mae:.4f}")

# %%
# Scatter plot for refined model
plt.figure(figsize=(8, 6))
plt.scatter(mod2_pred, obs_mod2, alpha=0.7, edgecolors='black')
plt.plot([obs_mod2.min(), obs_mod2.max()], [obs_mod2.min(), obs_mod2.max()], 'r--', alpha=0.8)
plt.xlabel('Predicted Points')
plt.ylabel('Observed Points')
plt.title(f'Refined Random Forest Model\nR² = {mod2_r2:.4f}, MAE = {mod2_mae:.4f}')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(mod2_pred, obs_mod2, 1)
p = np.poly1d(z)
plt.plot(mod2_pred, p(mod2_pred), "b-", alpha=0.8, linewidth=2)
plt.show()

print(f"Mean Absolute Error (Refined Model): {mod2_mae:.4f}")

# %%
# Model comparison
print("Model Comparison:")
comparison = pd.DataFrame({
    'Model': ['Full Model', 'Refined Model'],
    'Features': [len(X.columns), len(top_features)],
    'R²': [mod1_r2, mod2_r2],
    'MAE': [mod1_mae, mod2_mae]
})
print(comparison.round(4))

# %%
# Example 11.3: Statistical Power Analysis and Simulation
print("\nEXAMPLE 11.3: Statistical Power Analysis and Simulation")
print("="*55)

# Population parameters
print("Setting up population parameters...")

# Country A
mean_A = 178  # Height in cm
sd_A = 7      # Standard deviation in cm

# Country B  
mean_B = 180  # Height in cm
sd_B = 7      # Standard deviation in cm

npop = 1000000  # Population size

print(f"Country A: μ = {mean_A} cm, σ = {sd_A} cm")
print(f"Country B: μ = {mean_B} cm, σ = {sd_B} cm")
print(f"Population size: {npop:,}")

# %%
# Create populations
print("Generating populations...")

# Set seed for reproducibility
np.random.seed(123)

# Create two populations
A = np.random.normal(mean_A, sd_A, npop)
B = np.random.normal(mean_B, sd_B, npop)

# Create descriptive statistics table
results = pd.DataFrame({
    'CountryA': [np.mean(A), np.std(A, ddof=1)],
    'CountryB': [np.mean(B), np.std(B, ddof=1)]
}, index=['Mean', 'SD'])

print("\nPopulation descriptive statistics:")
print(results.round(1))

# %%
# Power analysis simulation
print("Conducting power analysis simulation...")

# Simulation parameters
nsamp = 30   # Sample size
nsim = 1000  # Number of simulations

print(f"Sample size per group: {nsamp}")
print(f"Number of simulations: {nsim}")

# Initialize results matrix
pval_results = np.zeros((nsim, 4))

# Cohen's d calculation function
def cohens_d(x, y):
    """Calculate Cohen's d effect size"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

# Run simulations
np.random.seed(123)
print("Running simulations...")

for i in range(nsim):
    # Sample from populations
    samp_A = np.random.choice(A, nsamp, replace=False)
    samp_B = np.random.choice(B, nsamp, replace=False)
    
    # Calculate difference in means
    diff_means = np.mean(samp_B) - np.mean(samp_A)
    
    # Calculate Cohen's d effect size
    effect_size = cohens_d(samp_A, samp_B)
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(samp_A, samp_B, equal_var=True)
    
    # Store results
    pval_results[i, 0] = round(diff_means, 2)
    pval_results[i, 1] = round(p_value, 3)
    pval_results[i, 2] = 1 if p_value < 0.05 else 0  # Significant flag
    pval_results[i, 3] = round(effect_size, 2)

# %%
# Create results DataFrame
print("Analyzing simulation results...")

pval_df = pd.DataFrame(pval_results, columns=['DiffMeans', 'p-value', 'Significant', 'CohensD'])

print("First 6 simulation results:")
print(pval_df.head(6))

print("\nOverall simulation summary:")
print(pval_df.describe().round(3))

# %%
# Calculate key statistics
print("Key simulation statistics:")

mean_diff = np.mean(pval_df['DiffMeans'])
mean_pval = np.mean(pval_df['p-value'])
mean_cohens_d = np.mean(pval_df['CohensD'])
sig_percentage = (np.sum(pval_df['Significant']) / nsim) * 100

print(f"Mean difference between sample means: {mean_diff:.3f}")
print(f"Mean computed p-value: {mean_pval:.3f}")
print(f"Mean effect size (Cohen's d): {mean_cohens_d:.3f}")
print(f"Percentage of significant results: {sig_percentage:.1f}%")

# %%
# Visualize p-value distribution
print("Creating p-value distribution histogram...")

plt.figure(figsize=(10, 6))
plt.hist(pval_df['p-value'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title(f'Distribution of p-values from {nsim} Simulations\n'
          f'Power = {sig_percentage:.1f}% (% of p < 0.05)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Effect size distribution
plt.figure(figsize=(10, 6))
plt.hist(pval_df['CohensD'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(x=mean_cohens_d, color='red', linestyle='--', linewidth=2, 
            label=f'Mean Cohen\'s d = {mean_cohens_d:.3f}')
plt.xlabel('Cohen\'s d')
plt.ylabel('Frequency')
plt.title(f'Distribution of Effect Sizes (Cohen\'s d) from {nsim} Simulations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Power analysis summary
print("Statistical Power Analysis Summary:")
print("="*50)

# Theoretical Cohen's d
theoretical_cohens_d = (mean_B - mean_A) / sd_A
print(f"Theoretical effect size (Cohen's d): {theoretical_cohens_d:.3f}")
print(f"Observed mean effect size: {mean_cohens_d:.3f}")

# Effect size interpretation
if abs(theoretical_cohens_d) < 0.2:
    effect_interpretation = "Small effect"
elif abs(theoretical_cohens_d) < 0.5:
    effect_interpretation = "Small to medium effect"
elif abs(theoretical_cohens_d) < 0.8:
    effect_interpretation = "Medium to large effect"
else:
    effect_interpretation = "Large effect"

print(f"Effect size interpretation: {effect_interpretation}")
print(f"Statistical power: {sig_percentage:.1f}%")

# Power interpretation
if sig_percentage >= 80:
    power_interpretation = "Adequate power (≥80%)"
elif sig_percentage >= 70:
    power_interpretation = "Moderate power (70-79%)"
else:
    power_interpretation = "Low power (<70%)"

print(f"Power interpretation: {power_interpretation}")

# %%
# Additional analysis: Power vs sample size
print("Power analysis across different sample sizes...")

sample_sizes = [10, 15, 20, 25, 30, 40, 50, 75, 100]
power_results = []

for n in sample_sizes:
    np.random.seed(123)
    significant_count = 0
    
    for i in range(200):  # Fewer simulations for speed
        samp_A = np.random.choice(A, n, replace=False)
        samp_B = np.random.choice(B, n, replace=False)
        _, p_value = stats.ttest_ind(samp_A, samp_B, equal_var=True)
        if p_value < 0.05:
            significant_count += 1
    
    power = (significant_count / 200) * 100
    power_results.append(power)

# Create power curve
power_analysis = pd.DataFrame({
    'Sample_Size': sample_sizes,
    'Power_%': power_results
})

print("\nPower analysis by sample size:")
print(power_analysis)

# %%
# Plot power curve
plt.figure(figsize=(10, 6))
plt.plot(power_analysis['Sample_Size'], power_analysis['Power_%'], 'b-o', linewidth=2, markersize=6)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Power Threshold')
plt.xlabel('Sample Size per Group')
plt.ylabel('Statistical Power (%)')
plt.title('Power Analysis: Effect of Sample Size on Statistical Power\n'
          f'Population difference = {mean_B - mean_A} cm (Cohen\'s d ≈ {theoretical_cohens_d:.3f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 100)
plt.show()

# %%
print("\n" + "="*60)
print("CHAPTER 11: ADVANCED STATISTICAL ANALYSIS - COMPLETE")
print("="*60)
print("Key concepts covered:")
print("✓ Data manipulation and categorical variables")
print("✓ Random Forest machine learning")
print("✓ Feature importance analysis")
print("✓ Model comparison and validation")
print("✓ Statistical power analysis")
print("✓ Monte Carlo simulation")
print("✓ Effect size calculation (Cohen's d)")
print("✓ Power curve analysis")
print("="*60)
