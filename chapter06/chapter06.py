# %%
"""
Chapter 6: Python code
Converted from R script by Clive Beggs 15th March 2023
Python conversion date: June 28, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# %%
# Example 6.1
# Import historical match data
url = "https://www.football-data.co.uk/mmz4281/1819/E0.csv"
mydata = pd.read_csv(url).head(380)

# Inspect data
print(mydata.columns)

# Select variables for inclusion in a working data frame
dat = mydata[
    ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA"]
]

dat = dat.copy()
# Rename column names
dat.rename(
    columns={
        "FTHG": "Hgoals",
        "FTAG": "Agoals",
        "FTR": "Result",
        "PSH": "HWodds",
        "PSD": "Dodds",
        "PSA": "AWodds",
    },
    inplace=True,
)

# Display the first six rows of dat
print(dat.head())

# Produce table of home and away goal frequencies
n = len(dat)
location = ["Home"] * n + ["Away"] * n
goals = list(dat["Hgoals"]) + list(dat["Agoals"])
goal_dat = pd.DataFrame({"location": location, "goals": goals})
print(goal_dat.value_counts())

# Plot home and away goal frequencies
hg_freq = dat["Hgoals"].value_counts(normalize=True).sort_index()
ag_freq = dat["Agoals"].value_counts(normalize=True).sort_index()

# Combined plot
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(hg_freq))
ax.bar(index, hg_freq, bar_width, label="Home team", color="darkgray")
ax.bar(index + bar_width, ag_freq, bar_width, label="Away team", color="lightgray")
ax.set_xlabel("Goals scored")
ax.set_ylabel("Fraction")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(hg_freq.index)
ax.legend()
plt.show()

# Compute and display the expected goals scored
hgoals_mean = dat["Hgoals"].mean()
agoals_mean = dat["Agoals"].mean()
print("Home goals mean:", hgoals_mean)
print("Away goals mean:", agoals_mean)

# Plot Poisson distributions
scale = np.arange(0, 7)
h_poisson = poisson.pmf(scale, hgoals_mean)
a_poisson = poisson.pmf(scale, agoals_mean)
plt.plot(scale, h_poisson, marker="o", label="Home team")
plt.plot(scale, a_poisson, marker="o", linestyle="--", label="Away team")
plt.xlabel("Goals")
plt.ylabel("Fraction")
plt.legend()
plt.show()

# Correlation with home and away team goals
h_corr = np.corrcoef(hg_freq, h_poisson[: len(hg_freq)])[0, 1]
a_corr = np.corrcoef(ag_freq, a_poisson[: len(ag_freq)])[0, 1]
print("Correlation with home team goals:", h_corr)
print("Correlation with away team goals:", a_corr)

# %%
# Example 6.2
# Select variables for inclusion in model
build_dat = dat.iloc[:370, [1, 2, 3, 4]]
build_dat.rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"}, inplace=True)

# Put data into long-form
long_dat = pd.concat(
    [
        pd.DataFrame(
            {
                "Home": 1,
                "Team": build_dat["Home"],
                "Opponent": build_dat["Away"],
                "Goals": build_dat["Hgoals"],
            }
        ),
        pd.DataFrame(
            {
                "Home": 0,
                "Team": build_dat["Away"],
                "Opponent": build_dat["Home"],
                "Goals": build_dat["Agoals"],
            }
        ),
    ]
)

# Inspect long-form data
print(long_dat.head())
print(long_dat.tail())

# Build Poisson model
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Poisson

pois_mod = glm("Goals ~ Home + Team + Opponent", data=long_dat, family=Poisson()).fit()
print(pois_mod.summary())

# Review remaining matches
remain_matches = dat.iloc[370:, :5]
print(remain_matches)

# Predict expected (average) goals
HTeam, ATeam = "Burnley", "Arsenal"
Hgoals_exp = pois_mod.predict(
    pd.DataFrame({"Home": [1], "Team": [HTeam], "Opponent": [ATeam]})
)
Agoals_exp = pois_mod.predict(
    pd.DataFrame({"Home": [0], "Team": [ATeam], "Opponent": [HTeam]})
)
print("Expected home goals:", Hgoals_exp.values[0])
print("Expected away goals:", Agoals_exp.values[0])

# Compute probability matrix
max_goals = 8
prob_mat = np.outer(
    poisson.pmf(range(max_goals + 1), Hgoals_exp.values[0]),
    poisson.pmf(range(max_goals + 1), Agoals_exp.values[0]),
)
print("Probability matrix:")
print(np.round(prob_mat, 4))

# Compute probabilities
home_win_prob = np.sum(np.tril(prob_mat, -1))
draw_prob = np.sum(np.diag(prob_mat))
away_win_prob = np.sum(np.triu(prob_mat, 1))
print("Home win probability:", home_win_prob)
print("Draw probability:", draw_prob)
print("Away win probability:", away_win_prob)

# %%
# Example 6.3
# Fit model
expected = pois_mod.fittedvalues

# Compile and display fitted results
exp_dat = long_dat.copy()
exp_dat["Expected"] = expected
print(exp_dat.head())
print(exp_dat.tail())

# Create home.exp and away.exp vectors
home_exp = expected[: len(build_dat)]
away_exp = expected[len(build_dat) :]

# Inspect these vectors
print(home_exp.head())
print(away_exp.tail())


# Construct user-defined function for Tau
def tau(x, y, lambda_, mu, rho):
    x, y = np.asarray(x), np.asarray(y)
    lambda_, mu = np.broadcast_to(lambda_, x.shape), np.broadcast_to(mu, y.shape)
    result = np.ones_like(x, dtype=float)
    mask_00 = (x == 0) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_10 = (x == 1) & (y == 0)
    mask_11 = (x == 1) & (y == 1)

    result[mask_00] = 1 - (lambda_[mask_00] * mu[mask_00] * rho)
    result[mask_01] = 1 + (lambda_[mask_01] * rho)
    result[mask_10] = 1 + (mu[mask_10] * rho)
    result[mask_11] = 1 - rho

    # Clip values to avoid invalid probabilities
    return np.clip(result, 1e-10, None)


# Construct user-defined function for log-likelihood
def log_like(y1, y2, lambda_, mu, rho=0):
    return np.sum(
        np.log(tau(y1, y2, lambda_, mu, rho))
        + np.log(poisson.pmf(y1, lambda_))
        + np.log(poisson.pmf(y2, mu))
    )


# Construct user-defined function for optimization
def opt_rho(par):
    rho = par[0]
    return log_like(build_dat["Hgoals"], build_dat["Agoals"], home_exp, away_exp, rho)


# Run optimization process
from scipy.optimize import minimize

res = minimize(lambda x: -opt_rho(x), [0.1], method="BFGS")
Rho = res.x[0]
print("Optimized Rho:", Rho)

# Adjust to the match probability matrices
lambda_ = Hgoals_exp.values[0]
mu = Agoals_exp.values[0]
prob_mat1 = np.outer(
    poisson.pmf(range(max_goals + 1), lambda_), poisson.pmf(range(max_goals + 1), mu)
)

# Apply Dixon-Coles adjustment
scale_mat = np.array(
    [
        [tau(0, 0, lambda_, mu, Rho), tau(0, 1, lambda_, mu, Rho)],
        [tau(1, 0, lambda_, mu, Rho), tau(1, 1, lambda_, mu, Rho)],
    ]
)
prob_mat2 = prob_mat1.copy()
prob_mat2[:2, :2] *= scale_mat
print("Adjusted probability matrix:")
print(np.round(prob_mat2, 4))

# Compute probabilities of match outcomes
home_win_prob_adj = np.sum(np.tril(prob_mat2, -1))
draw_prob_adj = np.sum(np.diag(prob_mat2))
away_win_prob_adj = np.sum(np.triu(prob_mat2, 1))
print("Adjusted home win probability:", home_win_prob_adj)
print("Adjusted draw probability:", draw_prob_adj)
print("Adjusted away win probability:", away_win_prob_adj)

# %%
# Example 6.4
# Inspect data
print(dat.head())

# Select variables for inclusion in random forest model
rf_dat = dat[["Result", "HWodds", "Dodds", "AWodds"]]
rf_dat = rf_dat.copy()
rf_dat["Result"] = rf_dat["Result"].astype("category")

# Split data into training and testing sets
rf_train_dat = rf_dat.iloc[:370]
rf_test_dat = rf_dat.iloc[370:]

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(rf_train_dat.drop("Result", axis=1), rf_train_dat["Result"])

# Assess the relative importance of predictor variables
print("Feature importances:", rf_model.feature_importances_)

# Make predictions
rf_pred = rf_model.predict(rf_test_dat.drop("Result", axis=1))

# Compare predictions with actual results
print(confusion_matrix(rf_test_dat["Result"], rf_pred))
print(classification_report(rf_test_dat["Result"], rf_pred))
