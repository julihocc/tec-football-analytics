# Chapter 1: Python code
# Copyright: Clive Beggs 6th March 2023 (Python translation May 31, 2025)
#
# This script demonstrates basic data analysis and visualization techniques using Python.
# It is organized into sections (cells) for interactive execution in VS Code or Jupyter-like environments.
# Each section is documented with comments explaining its purpose and logic.

# %%
# Import required libraries for data manipulation, visualization, and statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# %%
# Section 1: EPL Champions - DataFrame creation and visualization
# Create a DataFrame of EPL champions and their number of titles (1992-2021)
clubs = ["Arsenal", "Blackburn Rovers", "Chelsea", "Leicester City", "Liverpool", "Man City", "Man United"]
titles = [3, 1, 5, 1, 1, 6, 13]
epl_dat = pd.DataFrame({'Clubs': clubs, 'Titles': titles})
print(epl_dat)

# Bar plot of EPL champions
plt.figure()
plt.bar(epl_dat['Clubs'], epl_dat['Titles'])
plt.ylim(0, 14)
plt.title("Bar chart of EPL champions 1992-2021")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart of EPL champions
plt.figure()
colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(epl_dat['Clubs'])))
plt.pie(titles, labels=clubs, colors=colors)
plt.title("Pie Chart of EPL champions 1992-2021")
plt.show()

# %%
# Section 2: Manchester United Managers - Contingency Table and Chi-Square Test
# Create a DataFrame of match results for different managers
manager = ["Ferguson", "Moyes", "van Gaal", "Mourinho", "Solskjaer", "Rangnick"]
wins = [895, 27, 54, 84, 91, 11]
draws = [338, 9, 25, 32, 37, 10]
losses = [267, 15, 24, 28, 40, 8]

mu_record = pd.DataFrame({
    'Manager': manager,
    'Wins': wins,
    'Draws': draws,
    'Losses': losses
})
print(mu_record)

# Select two managers for comparison
man1 = "Ferguson"
man2 = "Rangnick"

# Extract win/draw/loss data for the selected managers
manager1 = mu_record[mu_record['Manager'] == man1].iloc[0, 1:].values
manager2 = mu_record[mu_record['Manager'] == man2].iloc[0, 1:].values

# Create a contingency table for chi-square test
cont_tab = np.array([manager1, manager2], dtype=float)  # Convert to float type
print("\nContingency Table:")
print(pd.DataFrame(
    cont_tab,
    index=['Manager 1', 'Manager 2'],
    columns=['Wins', 'Draws', 'Loses']
))

# Perform chi-square test to compare the managers' records
chi2, p_value, dof, expected = chi2_contingency(cont_tab)
print("\nChi-square test results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")

# %%
# Section 3: Regression Analysis - Points vs. Shots
# Analyze the relationship between points and shots for EPL teams in 2020-21
points = [61, 55, 41, 39, 67, 44, 59, 28, 59, 66, 69, 86, 74, 45, 23, 43, 62, 26, 65, 45]
shots = [455, 518, 476, 383, 553, 346, 395, 440, 524, 472, 600, 590, 517, 387, 319, 417, 442, 336, 462, 462]

dat2020 = pd.DataFrame({'Points': points, 'Shots': shots})
print("\nFirst 6 rows of the dataset:")
print(dat2020.head(6))

# Build OLS regression model for season 2020-21
from sklearn.linear_model import LinearRegression

X = dat2020['Shots'].values.reshape(-1, 1)
y = dat2020['Points'].values

model = LinearRegression()
model.fit(X, y)

# Print summary statistics for the regression
from sklearn.metrics import r2_score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("\nRegression Results:")
print(f"R-squared: {r2:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")

# Scatter plot with best-fit regression line
plt.figure()
plt.scatter(shots, points, color='black', marker='o')
plt.plot(X, y_pred, color='red')
plt.xlim(0, 800)
plt.ylim(0, 100)
plt.xlabel('Shots')
plt.ylabel('Points')
plt.show()

# Make predictions for specific teams based on their number of shots
print("\nPredictions:")
print(f"Chelsea (585 shots): {model.predict([[585]])[0]:.2f} points")
print(f"Manchester City (704 shots): {model.predict([[704]])[0]:.2f} points")
print(f"Norwich City (374 shots): {model.predict([[374]])[0]:.2f} points")

# %%
# Section 4: Betting Odds and Implied Probabilities
# Compare odds and implied probabilities from two bookmakers
# William Hill match odds
wh_hwodds = 2.15  # Odds offered by bookmaker for a home win
wh_dodds = 3.30   # Odds offered by bookmaker for a draw
wh_awodds = 3.50  # Odds offered by bookmaker for a away win

# Pinnacle match odds
p_hwodds = 2.13   # Odds offered by bookmaker for a home win
p_dodds = 3.61    # Odds offered by bookmaker for a draw
p_awodds = 3.64   # Odds offered by bookmaker for a away win

# Compile the data frame of odds
bet_dat = pd.DataFrame({
    'WH_odds': [wh_hwodds, wh_dodds, wh_awodds],
    'Pin_odds': [p_hwodds, p_dodds, p_awodds]
}, index=['Home win', 'Draw', 'Away win'])

# Compute implied probabilities for each outcome
bet_dat['WH_prob'] = round(1/bet_dat['WH_odds'], 3)
bet_dat['Pin_prob'] = round(1/bet_dat['Pin_odds'], 3)
print("\nBetting odds and probabilities:")
print(bet_dat)

# Calculate the over-round (bookmaker's margin) for each bookmaker
wh_or = bet_dat['WH_prob'].sum() - 1
print(f"\nWilliam Hill's over-round: {wh_or:.3f}")
pin_or = bet_dat['Pin_prob'].sum() - 1
print(f"Pinnacle's over-round: {pin_or:.3f}")

# Calculate profit from a sample wager
wager = 10  # £10 wager with WH on Tottenham to win
profit = wager * (wh_hwodds - 1)
print(f"\nProfit from £{wager} wager: £{profit:.2f}")

# %%
# Section 5: Random Cup Draw Simulation
# Simulate a random cup draw for 16 teams
teams = list(range(1, 17))
print("\nTeams:", teams)

# Create DataFrame to store draw results
cup_draw = pd.DataFrame(columns=['HomeTeam', 'AwayTeam'])

# Randomly sample eight home teams
np.random.seed(123)  # This makes the draw results repeatable
samp_ht = np.random.choice(teams, size=8, replace=False)
remaining_teams = [t for t in teams if t not in samp_ht]
samp_at = np.random.choice(remaining_teams, size=8, replace=False)

cup_draw['HomeTeam'] = samp_ht
cup_draw['AwayTeam'] = samp_at
print("\nCup Draw:")
print(cup_draw)
