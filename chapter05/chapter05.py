# %%
"""
Chapter 5: Python code
Converted from R script by Clive Beggs 13th March 2023
Python conversion date: June 28, 2025
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Dynamically determine the file path based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
file_path = os.path.join(data_dir, "EPL_final_table_2021.csv")

# %%
# Example 5.1
# Load data
tabdat = pd.read_csv(file_path)

# Compute average points per match for each team
tabdat["avPPM"] = round(tabdat["PTS"] / tabdat["PLD"], 4)

# Display data
print(tabdat.head())

# Compute the number of matches and teams in the league
m = tabdat["PLD"].mean()  # Number of matches played per team
n = len(tabdat)  # Number of teams in league

# Compute the observed points average and variance
LPPM_av = tabdat["avPPM"].mean()  # Average points per match for the league
LPTS_av = tabdat["PTS"].mean()  # Average team points total over the season
LPTS_error = tabdat["PTS"] - LPTS_av
LPTS_var = (1 / n) * sum(LPTS_error**2)  # Variance in team points

# Specify probabilities of outcomes
pH, pA, pD = 0.46, 0.29, 0.25

# Compile league statistics
league_stats = {
    "Pld": round(m, 3),
    "HWprob": round(pH, 3),
    "AWprob": round(pA, 3),
    "Dprob": round(pD, 3),
    "Pts.tot": round(tabdat["PTS"].sum(), 3),
    "Pts.av": round(LPTS_av, 3),
    "Pts.var": round(LPTS_var, 3),
    "Ptspm.av": round(LPPM_av, 3),
}
print(league_stats)

# Expected average points total per team
exP = (m / 2) * (3 - pD)

# Expected points variance for all teams in the league
exVar = (m / 2) * (9 - (7 * pD) - ((3 * pH + pD) ** 2) - ((3 * pA + pD) ** 2))

# Proportion of league points variance due to chance
fchance = exVar / LPTS_var

# Display results
print(exP, exVar, LPTS_var, fchance)

# %%
# Example 5.2
# Load data
file_path = os.path.join(data_dir, "EPL_standings_2013_14.csv")
download_dat = pd.read_csv(file_path)
standings = download_dat.iloc[:, 1:39]
standings.index = download_dat.iloc[:, 0]

# Perform Spearman correlation analysis
cor_r, cor_p = spearmanr(standings, axis=0)
round38_r = np.round(cor_r[:, -1], 3)
round38_p = np.round(cor_p[:, -1], 3)

# Display results
print(round38_r)
print(round38_p)

# Plot results
plt.plot(cor_r[:, -1], color="black")
plt.ylim(0, 1)
plt.xlabel("Round of competition")
plt.ylabel("Spearman correlation (r value)")
plt.show()

# %%
# Example 5.3
# Load data
file_path = os.path.join(data_dir, "EPL_final_table_2021.csv")
EoStab = pd.read_csv(file_path)

# Specify coefficients
a, b, c, d = 2.78, 1.24, 1.24, 1.25

# Compute key metrics
pygFrac = (EoStab["GF"] ** b) / ((EoStab["GF"] ** c) + (EoStab["GA"] ** d))
pygPts = a * pygFrac * EoStab["PLD"]
pygDiff = EoStab["PTS"] - pygPts

# Round to specified decimal places
pygFrac = pygFrac.round(3)
pygPts = pygPts.round(2)
pygDiff = pygDiff.round(2)

# Compile table
pygtab = pd.concat([EoStab, pygFrac, pygPts, pygDiff], axis=1)
print(pygtab)

# Evaluate correlation between Pythagorean expected points and actual points
correlation = pygPts.corr(EoStab["PTS"])
print("Correlation r value:", correlation)

# Produce scatter plot
plt.scatter(EoStab["PTS"], pygPts, color="black")
plt.xlabel("EPL points")
plt.ylabel("Pythagorean points")
plt.plot(
    np.unique(EoStab["PTS"]),
    np.poly1d(np.polyfit(EoStab["PTS"], pygPts, 1))(np.unique(EoStab["PTS"])),
    linestyle="--",
)
plt.show()


# %%
# Example 5.4
# Pythagorean expected points function
def pythag_pred(PLD, GF, GA, PTS, nGames):
    a, b, c, d = 2.78, 1.24, 1.24, 1.25
    pythag_frac = (GF**b) / ((GF**c) + (GA**d))
    pythag_pts = a * pythag_frac * PLD
    pythag_diff = PTS - pythag_pts
    points_avail = (nGames - PLD) * 3
    pred_pts = pythag_frac * a * (nGames - PLD)
    pred_total = PTS + pred_pts
    pythag_total = pythag_pts + pred_pts
    return [
        PLD,
        GF,
        GA,
        PTS,
        pythag_frac,
        pythag_pts,
        pythag_diff,
        points_avail,
        pred_pts,
        pred_total,
        pythag_total,
    ]


# Specify input parameters
nTeams = 20
nGames = (nTeams - 1) * 2
PLD, GF, GA, PTS = 10, 22, 17, 21

# Apply the function
pred_res = pythag_pred(PLD, GF, GA, PTS, nGames)
var_names = [
    "PLD",
    "GF",
    "GA",
    "PTS",
    "PythagFrac",
    "PythagPTS",
    "PythagDiff",
    "AvailPTS",
    "PredPTS",
    "PredTot",
    "PythagTot",
]
Liv_pred10 = pd.DataFrame([pred_res], columns=var_names)
print(Liv_pred10)

# %%
# Example 5.5
# Load data
file_path = os.path.join(data_dir, "EPL_after_98_matches_2021.csv")
part_tab = pd.read_csv(file_path)

# Create empty matrix to store results
predPTS = []
for i in range(len(part_tab)):
    predPTS.append(
        pythag_pred(
            part_tab.loc[i, "PLD"],
            part_tab.loc[i, "GF"],
            part_tab.loc[i, "GA"],
            part_tab.loc[i, "PTS"],
            nGames,
        )
    )

# Compile results
Pred_EoSPTS = pd.DataFrame(predPTS, columns=var_names)
Pred_EoSPTS.insert(0, "Club", part_tab["Club"])
print(Pred_EoSPTS.head())

# Load and sort EoS table
file_path = os.path.join(data_dir, "EPL_final_table_2021.csv")
EoS_tab = pd.read_csv(file_path)
sortedEoS = EoS_tab.sort_values("Club")
sortedPred = Pred_EoSPTS.sort_values("Club")

# Compile EoS table with predicted points
EoS_predtab = sortedEoS.copy()
EoS_predtab["Pred"] = sortedPred["PredTot"].values
print(EoS_predtab)

# Evaluate accuracy of predictions
correlation = sortedEoS["PTS"].corr(sortedPred["PredTot"])
mae = np.mean(np.abs(sortedEoS["PTS"] - sortedPred["PredTot"]))
print("Correlation r value:", correlation)
print("Mean Absolute Error:", mae)

# Scatter plot
plt.scatter(sortedEoS["PTS"], sortedPred["PredTot"], color="black")
plt.xlabel("Actual EoS points")
plt.ylabel("Predicted EoS points")
plt.plot(
    np.unique(sortedEoS["PTS"]),
    np.poly1d(np.polyfit(sortedEoS["PTS"], sortedPred["PredTot"], 1))(
        np.unique(sortedEoS["PTS"])
    ),
    linestyle="--",
)
plt.show()
