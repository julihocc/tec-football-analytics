# %%
# Chapter 10: Python code
# Copyright: Clive Beggs - 3rd April 2023
# Converted from R to Python - Linear Regression Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHAPTER 10: LINEAR REGRESSION ANALYSIS")
print("="*60)
print("This chapter demonstrates:")
print("- Descriptive statistics and t-tests")
print("- Correlation analysis")
print("- Multiple linear regression")
print("- Model diagnostics and validation")
print("="*60)

# %%
# Example 10.1: Descriptive Statistics and t-tests
print("EXAMPLE 10.1: Descriptive Statistics and Independent t-tests")
print("="*50)

# Load data
regdata = pd.read_csv('../data/EPL_regression_data_2020_2021.csv')
print("Data loaded successfully!")
print(f"Dataset shape: {regdata.shape}")
print("\nFirst 6 rows:")
print(regdata.head(6))

# %%
# Split data into 2020-21 and 2021-22 seasons
season1 = regdata[regdata['Season'] == 2020].copy()  # 2020-21
season2 = regdata[regdata['Season'] == 2021].copy()  # 2021-22

print(f"Season 2020-21: {len(season1)} teams")
print(f"Season 2021-22: {len(season2)} teams")

# Remove the first three columns (League, Season, Team)
s1 = season1.iloc[:, 3:14].copy()  # Columns 4-14 in R (0-indexed in Python: 3-13)
s2 = season2.iloc[:, 3:14].copy()

print(f"\nColumns selected for analysis:")
print(s1.columns.tolist())

# %%
# Season 2020-21 descriptive statistics
print("Computing descriptive statistics for both seasons...")

s1_stats = pd.DataFrame({
    'S1.n': s1.count(),
    'S1.mean': s1.mean(),
    'S1.SD': s1.std()
})

# Season 2021-22 descriptive statistics  
s2_stats = pd.DataFrame({
    'S2.n': s2.count(),
    'S2.mean': s2.mean(),
    'S2.SD': s2.std()
})

print("\nSeason 2020-21 descriptive statistics:")
print(s1_stats.round(1))
print("\nSeason 2021-22 descriptive statistics:")
print(s2_stats.round(1))

# %%
# Perform independent t-tests
print("Performing independent t-tests between seasons...")

# Perform t-tests for each variable
pval_list = []
variables = s1.columns.tolist()

for col in variables:
    # Independent t-test (equal variances not assumed)
    t_stat, p_val = stats.ttest_ind(s1[col], s2[col], equal_var=False)
    pval_list.append(float(round(p_val, 3)))

# Compile descriptive statistics results
stats_combined = pd.concat([s1_stats, s2_stats], axis=1)
stats_combined = stats_combined.round(1)
stats_combined['pval'] = pval_list

# Set variable names as index
stats_combined.index.name = 'Variables'
stats_res = stats_combined.reset_index()

print("\nCombined descriptive statistics with t-test p-values:")
print(stats_res)

# %%
# Example 10.2: Correlation Analysis
print("\nEXAMPLE 10.2: Correlation Analysis")
print("="*40)

# Season 2020-21 correlation matrix
s1_corr = s1.corr()
print("Season 2020-21 correlation matrix:")
print(s1_corr.round(3))

# %%
# Season 2021-22 correlation matrix
s2_corr = s2.corr()
print("\nSeason 2021-22 correlation matrix:")
print(s2_corr.round(3))

# %%
# Visualize correlation matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Season 2020-21 heatmap
sns.heatmap(s1_corr, annot=True, cmap='RdBu_r', center=0, 
            square=True, ax=ax1, cbar_kws={"shrink": .8})
ax1.set_title('Season 2020-21 Correlation Matrix', fontsize=12, fontweight='bold')

# Season 2021-22 heatmap  
sns.heatmap(s2_corr, annot=True, cmap='RdBu_r', center=0,
            square=True, ax=ax2, cbar_kws={"shrink": .8})
ax2.set_title('Season 2021-22 Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Build simple linear regression models for season 2020-21
print("Building simple linear regression models...")

# Using Shots
X_shots = sm.add_constant(s1['Shots'])
shots_model = sm.OLS(s1['Points'], X_shots).fit()
print("\nRegression model using Shots:")
print(shots_model.summary())
print(f"AIC: {shots_model.aic:.2f}")

# %%
# Using Shots on Target (SoT)
X_sot = sm.add_constant(s1['SoT'])
sot_model = sm.OLS(s1['Points'], X_sot).fit()
print("\nRegression model using Shots on Target (SoT):")
print(sot_model.summary())
print(f"AIC: {sot_model.aic:.2f}")

# %%
# Scatter plot with best-fit lines
plt.figure(figsize=(10, 6))

# Plot Shots
plt.scatter(s1['Shots'], s1['Points'], marker='x', color='black', s=50, label='Shots')
# Shots regression line
shots_line = shots_model.params[0] + shots_model.params[1] * s1['Shots']
plt.plot(s1['Shots'], shots_line, 'k-', linewidth=2, label='Shots best-fit line')

# Plot SoT
plt.scatter(s1['SoT'], s1['Points'], marker='o', color='red', s=50, alpha=0.7, label='SoT')
# SoT regression line  
sot_line = sot_model.params[0] + sot_model.params[1] * s1['SoT']
plt.plot(s1['SoT'], sot_line, 'r--', linewidth=2, label='SoT best-fit line')

plt.xlim(0, 800)
plt.ylim(-40, 100)
plt.xlabel('Shots & SoT')
plt.ylabel('Points')
plt.title('Scatter Plot with Best-Fit Lines')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Example 10.3: Multiple Linear Regression
print("\nEXAMPLE 10.3: Multiple Linear Regression")
print("="*45)

# Remove Shots variable from datasets (use SoT instead)
build_dat = s1.drop(['Shots'], axis=1).copy()
print(f"Variables for model building:")
print(build_dat.columns.tolist())

# %%
# Create multiple linear regression model for season 2020-21
print("Building multiple linear regression models...")

# Base model (full model)
X_full = sm.add_constant(build_dat.drop(['Points'], axis=1))
s1_lm1 = sm.OLS(build_dat['Points'], X_full).fit()

print("\nBase model (all variables):")
print(s1_lm1.summary())

# %%
# Manual backward elimination based on p-values
print("Performing manual backward elimination...")

# Remove 'PassComp' (highest p-value > 0.05)
predictors = ['SoT', 'ShotDist', 'Dribbles', 'Tackles', 'Crosses', 'Intercepts', 'AerialWon', 'AerialLost']
X2 = sm.add_constant(build_dat[predictors])
s1_lm2 = sm.OLS(build_dat['Points'], X2).fit()
print(f"\nModel 2 (removed PassComp) - AIC: {s1_lm2.aic:.2f}")
print("Significant variables (p < 0.05):")
pvals = s1_lm2.pvalues[1:]  # Exclude constant
sig_vars = pvals[pvals < 0.05]
print(sig_vars.round(4))

# %%
# Continue elimination - remove 'AerialWon'
predictors = ['SoT', 'ShotDist', 'Dribbles', 'Tackles', 'Crosses', 'Intercepts', 'AerialLost']
X3 = sm.add_constant(build_dat[predictors])
s1_lm3 = sm.OLS(build_dat['Points'], X3).fit()
print(f"\nModel 3 (removed AerialWon) - AIC: {s1_lm3.aic:.2f}")

# Remove 'Intercepts'
predictors = ['SoT', 'ShotDist', 'Dribbles', 'Tackles', 'Crosses', 'AerialLost']
X4 = sm.add_constant(build_dat[predictors])
s1_lm4 = sm.OLS(build_dat['Points'], X4).fit()
print(f"\nModel 4 (removed Intercepts) - AIC: {s1_lm4.aic:.2f}")

# Finally remove 'ShotDist'
predictors_final = ['SoT', 'Dribbles', 'Tackles', 'Crosses', 'AerialLost']
X5 = sm.add_constant(build_dat[predictors_final])
s1_lm5 = sm.OLS(build_dat['Points'], X5).fit()

print(f"\nFinal refined model:")
print(s1_lm5.summary())
print(f"AIC: {s1_lm5.aic:.2f}")

# %%
# Compute 95% confidence intervals
print("\n95% Confidence intervals for coefficients:")
conf_int = s1_lm5.conf_int()
conf_int.columns = ['2.5%', '97.5%']
print(conf_int.round(4))

# %%
# Compare models using F-test
print("\nModel comparison (F-test):")
print("Comparing full model vs refined model...")

# F-test for nested models
f_test = s1_lm1.compare_f_test(s1_lm5)
print(f"F-statistic: {f_test[0]:.4f}")
print(f"p-value: {f_test[1]:.4f}")
print(f"Decision: {'Refined model is adequate' if f_test[1] > 0.05 else 'Full model is better'}")

# %%
# Automatic backward elimination using AIC
print("\nAutomatic backward elimination using AIC...")

def backward_elimination_aic(X, y, threshold_out=0.05):
    """
    Perform backward elimination based on AIC
    """
    included = list(X.columns)
    best_aic = float('inf')
    
    while True:
        changed = False
        excluded = []
        
        # Try removing each variable
        for var in included:
            if var == 'const':
                continue
                
            test_vars = [v for v in included if v != var]
            X_test = X[test_vars]
            model_test = sm.OLS(y, X_test).fit()
            
            if model_test.aic < best_aic:
                best_aic = model_test.aic
                excluded = [var]
                changed = True
        
        if changed:
            included = [v for v in included if v not in excluded]
            print(f"Removed: {excluded}, AIC: {best_aic:.2f}")
        else:
            break
    
    return included

# Perform automatic selection
X_auto = sm.add_constant(build_dat.drop(['Points'], axis=1))
selected_vars = backward_elimination_aic(X_auto, build_dat['Points'])
X_auto_final = X_auto[selected_vars]
s1_lm6 = sm.OLS(build_dat['Points'], X_auto_final).fit()

print(f"\nAutomatic backward elimination result:")
print(f"Selected variables: {[v for v in selected_vars if v != 'const']}")
print(f"AIC: {s1_lm6.aic:.2f}")

# %%
# Example 10.4: Relative Importance Analysis
print("\nEXAMPLE 10.4: Relative Importance Analysis")
print("="*45)

def relative_weights(model, X):
    """
    Calculate relative weights (similar to R's relweights function)
    Based on Johnson (2000) relative weights analysis
    """
    # Get correlation matrix
    R = np.corrcoef(X.T)
    
    # Get correlations with dependent variable (first column)
    rxy = R[1:, 0]  # Correlations with DV
    rxx = R[1:, 1:]  # Predictor intercorrelations
    
    # Eigen decomposition
    eigenvals, eigenvecs = np.linalg.eig(rxx)
    
    # Create lambda matrix
    lambda_mat = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
    
    # Calculate beta weights
    try:
        beta = np.linalg.solve(lambda_mat, rxy)
    except:
        beta = np.linalg.lstsq(lambda_mat, rxy, rcond=None)[0]
    
    # Calculate relative weights
    lambda_sq = lambda_mat ** 2
    raw_weights = lambda_sq @ (beta ** 2)
    rsquare = np.sum(beta ** 2)
    rel_weights = (raw_weights / rsquare) * 100
    
    return rel_weights, rsquare

# %%
# Calculate relative weights for refined model
print("Relative weights analysis for refined model...")

# Prepare data for relative weights analysis
X_rel = build_dat[['Points'] + predictors_final].values
rel_weights_refined, rsq_refined = relative_weights(s1_lm5, X_rel)

# Create results dataframe
rel_results_refined = pd.DataFrame({
    'Variable': predictors_final,
    'Rel_Weight_%': rel_weights_refined
})
rel_results_refined = rel_results_refined.sort_values('Rel_Weight_%', ascending=False)

print("\nRelative importance (% of R-squared) - Refined Model:")
print(rel_results_refined.round(2))

# %%
# Visualize relative weights
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Refined model
ax1.bar(rel_results_refined['Variable'], rel_results_refined['Rel_Weight_%'], 
        color='skyblue', edgecolor='black')
ax1.set_title(f'Relative Importance - Refined Model\n(R² = {s1_lm5.rsquared:.3f})')
ax1.set_ylabel('% of R-Square')
ax1.set_xlabel('Predictor Variables')
ax1.tick_params(axis='x', rotation=45)

# Base model relative weights
X_rel_full = build_dat.values
rel_weights_full, rsq_full = relative_weights(s1_lm1, X_rel_full)
rel_results_full = pd.DataFrame({
    'Variable': build_dat.columns[1:],  # Exclude Points
    'Rel_Weight_%': rel_weights_full
})
rel_results_full = rel_results_full.sort_values('Rel_Weight_%', ascending=False)

ax2.bar(rel_results_full['Variable'], rel_results_full['Rel_Weight_%'], 
        color='lightcoral', edgecolor='black')
ax2.set_title(f'Relative Importance - Full Model\n(R² = {s1_lm1.rsquared:.3f})')
ax2.set_ylabel('% of R-Square')
ax2.set_xlabel('Predictor Variables')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print(f"\nRelative importance (% of R-squared) - Full Model:")
print(rel_results_full.round(2))

# %%
# Correlation analysis of strongly correlated variables
print("Correlation analysis of key variables:")

key_vars = ['Points', 'SoT', 'PassComp', 'AerialLost']
cor_matrix = build_dat[key_vars].corr()
print(cor_matrix.round(3))

# Single predictor model using PassComp
X_passcomp = sm.add_constant(build_dat['PassComp'])
passcomp_model = sm.OLS(build_dat['Points'], X_passcomp).fit()
print(f"\nRegression model using PassComp only:")
print(f"R²: {passcomp_model.rsquared:.3f}")
print(f"AIC: {passcomp_model.aic:.2f}")

# %%
# Variance Inflation Factor (VIF) analysis
print("Variance Inflation Factor (VIF) analysis:")

def calculate_vif(df, features):
    """Calculate VIF for features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) 
                       for i in range(len(features))]
    return vif_data

# VIF for full model
full_features = [col for col in build_dat.columns if col != 'Points']
vif_full = calculate_vif(build_dat, full_features)
print("\nVIF values for full model:")
print(vif_full.round(2))

# VIF for refined model  
vif_refined = calculate_vif(build_dat, predictors_final)
print("\nVIF values for refined model:")
print(vif_refined.round(2))

print("\nVIF Interpretation:")
print("VIF = 1: No multicollinearity")
print("VIF = 1-5: Moderate multicollinearity") 
print("VIF > 5: High multicollinearity (concerning)")
print("VIF > 10: Severe multicollinearity (problematic)")

# %%
# Example 10.5: Model Prediction and Validation
print("\nEXAMPLE 10.5: Model Prediction and Validation")
print("="*50)

# Season 2020-21 predictions (training data)
s1_pred = s1_lm5.predict()
obs_s1 = build_dat['Points']

print("Season 2020-21 predictions (training data):")
pred_results_s1 = pd.DataFrame({
    'Team': season1['Team'].values,
    'Observed': obs_s1.values,
    'Predicted': s1_pred,
    'Residual': (obs_s1 - s1_pred).values
})
print(pred_results_s1.round(1))

# %%
# Model fit metrics for season 2020-21
s1_r2 = r2_score(obs_s1, s1_pred)
s1_mae = mean_absolute_error(obs_s1, s1_pred)

print(f"\nSeason 2020-21 model fit metrics:")
print(f"R²: {s1_r2:.3f}")
print(f"Mean Absolute Error: {s1_mae:.2f}")

# Scatter plot for season 2020-21
plt.figure(figsize=(8, 6))
plt.scatter(s1_pred, obs_s1, color='black', s=50)
plt.plot([20, 90], [20, 90], 'r--', alpha=0.8)  # Perfect prediction line
plt.xlim(20, 90)
plt.ylim(0, 100)
plt.xlabel('Predicted Points')
plt.ylabel('Observed Points')
plt.title('Season 2020-21: Predicted vs Observed Points')
plt.grid(True, alpha=0.3)

# Add R² text
plt.text(25, 85, f'R² = {s1_r2:.3f}\nMAE = {s1_mae:.1f}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.show()

# %%
# Test predictive ability on season 2021-22
print("Testing predictive ability on season 2021-22...")

# Prepare test data (remove Shots column to match training data)
test_dat = s2.drop(['Shots'], axis=1).copy()

# Make predictions using the model trained on 2020-21 data
X_test = sm.add_constant(test_dat[predictors_final])
s2_pred = s1_lm5.predict(X_test)
obs_s2 = test_dat['Points']

print("Season 2021-22 predictions (test data):")
pred_results_s2 = pd.DataFrame({
    'Team': season2['Team'].values,
    'Observed': obs_s2.values,
    'Predicted': s2_pred,
    'Residual': (obs_s2 - s2_pred).values
})
print(pred_results_s2.round(1))

# %%
# Model fit metrics for season 2021-22
s2_r2 = r2_score(obs_s2, s2_pred)
s2_mae = mean_absolute_error(obs_s2, s2_pred)

print(f"\nSeason 2021-22 model fit metrics:")
print(f"R²: {s2_r2:.3f}")
print(f"Mean Absolute Error: {s2_mae:.2f}")

# Scatter plot for season 2021-22
plt.figure(figsize=(8, 6))
plt.scatter(s2_pred, obs_s2, color='red', s=50, alpha=0.7)
plt.plot([20, 90], [20, 90], 'k--', alpha=0.8)  # Perfect prediction line
plt.xlim(20, 90)
plt.ylim(0, 100)
plt.xlabel('Predicted Points')
plt.ylabel('Observed Points')
plt.title('Season 2021-22: Predicted vs Observed Points (Test Data)')
plt.grid(True, alpha=0.3)

# Add R² text
plt.text(25, 85, f'R² = {s2_r2:.3f}\nMAE = {s2_mae:.1f}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.show()

# %%
# Compare model performance
print("Model performance comparison:")
performance_comparison = pd.DataFrame({
    'Dataset': ['Season 2020-21 (Training)', 'Season 2021-22 (Test)'],
    'R²': [s1_r2, s2_r2],
    'MAE': [s1_mae, s2_mae]
})
print(performance_comparison.round(3))

# %%
# Example 10.6: Model Diagnostics
print("\nEXAMPLE 10.6: Model Diagnostics")
print("="*35)

# Breusch-Pagan test for heteroscedasticity
print("1. Breusch-Pagan test for heteroscedasticity:")
bp_stat, bp_pval, _, _ = het_breuschpagan(s1_lm5.resid, X5)
print(f"Breusch-Pagan statistic: {bp_stat:.4f}")
print(f"p-value: {bp_pval:.4f}")
print(f"Interpretation: {'Homoscedastic' if bp_pval > 0.05 else 'Heteroscedastic'} (α = 0.05)")

# %%
# Shapiro-Wilk test for normality of residuals
print("\n2. Shapiro-Wilk test for normality of residuals:")
residuals = obs_s1 - s1_pred
sw_stat, sw_pval = shapiro(residuals)
print(f"Shapiro-Wilk statistic: {sw_stat:.4f}")
print(f"p-value: {sw_pval:.4f}")
print(f"Interpretation: {'Normal' if sw_pval > 0.05 else 'Non-normal'} residuals (α = 0.05)")

# %%
# Durbin-Watson test for autocorrelation
print("\n3. Durbin-Watson test for autocorrelation:")
dw_stat = durbin_watson(s1_lm5.resid)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")
print("Interpretation:")
print("- DW ≈ 2: No autocorrelation")
print("- DW < 2: Positive autocorrelation") 
print("- DW > 2: Negative autocorrelation")
print(f"- Current value suggests: {'No significant autocorrelation' if 1.5 < dw_stat < 2.5 else 'Potential autocorrelation'}")

# %%
# Diagnostic plots
print("\n4. Diagnostic plots:")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Residuals vs Fitted
fitted_values = s1_lm5.fittedvalues
residuals = s1_lm5.resid

ax1.scatter(fitted_values, residuals, alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted')
ax1.grid(True, alpha=0.3)

# 2. Q-Q plot
from scipy.stats import probplot
probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Normal Q-Q Plot')
ax2.grid(True, alpha=0.3)

# 3. Scale-Location plot
standardized_resid = residuals / np.sqrt(np.var(residuals))
ax3.scatter(fitted_values, np.abs(standardized_resid), alpha=0.7)
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('√|Standardized Residuals|')
ax3.set_title('Scale-Location')
ax3.grid(True, alpha=0.3)

# 4. Residuals vs Leverage
# Calculate leverage (hat values)
leverage = s1_lm5.get_influence().hat_matrix_diag
ax4.scatter(leverage, standardized_resid, alpha=0.7)
ax4.axhline(y=0, color='red', linestyle='--')
ax4.set_xlabel('Leverage')
ax4.set_ylabel('Standardized Residuals')
ax4.set_title('Residuals vs Leverage')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Additional diagnostic information
print("\n5. Additional diagnostic information:")

# Influential observations
influence = s1_lm5.get_influence()
cooks_d = influence.cooks_distance[0]

print("Top 5 teams by Cook's Distance (influence):")
teams_influence = pd.DataFrame({
    'Team': season1['Team'].values,
    'Cooks_Distance': cooks_d,
    'Leverage': leverage,
    'Residual': residuals
})
top_influence = teams_influence.nlargest(5, 'Cooks_Distance')
print(top_influence.round(4))

# Summary of diagnostic tests
print(f"\nDiagnostic Tests Summary:")
print(f"{'='*40}")
print(f"Breusch-Pagan (Heteroscedasticity): {'PASS' if bp_pval > 0.05 else 'FAIL'} (p = {bp_pval:.4f})")
print(f"Shapiro-Wilk (Normality): {'PASS' if sw_pval > 0.05 else 'FAIL'} (p = {sw_pval:.4f})")
print(f"Durbin-Watson (Autocorrelation): {'PASS' if 1.5 < dw_stat < 2.5 else 'FAIL'} (DW = {dw_stat:.4f})")
print(f"Max VIF: {vif_refined['VIF'].max():.2f} ({'PASS' if vif_refined['VIF'].max() < 5 else 'FAIL'} if < 5)")

# %%
print("\n" + "="*60)
print("CHAPTER 10: LINEAR REGRESSION ANALYSIS - COMPLETE")
print("="*60)
print("Key concepts covered:")
print("✓ Descriptive statistics and t-tests")
print("✓ Correlation analysis and visualization")
print("✓ Simple and multiple linear regression")
print("✓ Model selection and backward elimination")
print("✓ Relative importance analysis")
print("✓ Multicollinearity assessment (VIF)")
print("✓ Model validation and prediction")
print("✓ Comprehensive model diagnostics")
print("✓ Residual analysis and assumption testing")
print("="*60)
