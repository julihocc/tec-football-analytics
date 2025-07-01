# %%
"""
Chapter 7: Python code
Converted from R script by Clive Beggs 21st March 2023
Python conversion date: June 28, 2025
"""

import numpy as np
import pandas as pd

# %%
# Example 7.1

# Specify number of players
nplayers = 10  # Number of players playing roulette
nspins = 10  # Number of spins per player
wager = 10  # Wager per spin (pounds)

# Simulate the roulette wheel
# Create empty matrix to store results
outcome = np.zeros((nplayers, nspins), dtype=int)
wheel = [1] * 18 + [-1] * 19  # European roulette wheel (18 black, 18 red, 1 green)
print(wheel)  # This displays the roulette wheel vector.

# Simulate random spins of the roulette wheel for 10 players
np.random.seed(234)  # This sets the seed so that the results are reproducible.
for i in range(nplayers):
    for j in range(nspins):
        outcome[i, j] = np.random.choice(
            wheel
        )  # Randomly sample one value from the wheel vector.

# Display the results for each player
results = outcome * wager  # Compute the value of the wins and losses
print(results)  # Negative values represent losses and positive values represent wins

# Compute total profit for each player
profit = results.sum(axis=0)
print(profit)

# Compute profit for the casino
house_profit = -1 * results.sum()  # Convert the negative values into positive ones.
print(house_profit)

# %%
# Example 7.2

b_odds = 2.93  # Odds offered by bookmaker
t_odds = 2.50  # Assumed 'true' odds

# Compute probabilities
# Bookmaker's estimate
b_prob = 1 / b_odds
print(b_prob)

# Assumed true probability
t_prob = 1 / t_odds
print(t_prob)

# Create a virtual roulette wheel to simulate the match outcomes
n = round(t_prob * 1000)  # Total number of segments required on the virtual wheel
sims = 10  # Number of simulations (spins of the wheel)
vir_wheel = [1] * n + [-1] * (1000 - n)  # Virtual roulette wheel
bet = 10  # Amount wagered (£10)

np.random.seed(234)
res = [np.random.choice(vir_wheel) for _ in range(sims)]

# Display random match outcomes
print(res)

# Compute the potential winnings
winnings = []
for r in res:
    if r < 0:
        winnings.append(-1 * bet)
    else:
        winnings.append(bet * (b_odds - 1))

# Display winnings
print(winnings)

# Compute expected profit over ten matches
prof = sum(winnings)
print(prof)

# Compute expected winnings per match
exp_winnings = np.mean(winnings)
print(exp_winnings)

# %%
# Example 7.3

# Load data
url = "https://www.football-data.co.uk/mmz4281/1819/E0.csv"
mydata = pd.read_csv(url).head(380)

# Select variables of interest
dat = mydata[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "B365H",
        "B365D",
        "B365A",
        "BWH",
        "BWD",
        "BWA",
        "IWH",
        "IWD",
        "IWA",
        "PSH",
        "PSD",
        "PSA",
        "WHH",
        "WHD",
        "WHA",
        "VCH",
        "VCD",
        "VCA",
    ]
]

# Group odds
hwin = dat[["B365H", "BWH", "IWH", "PSH", "WHH", "VCH"]]
draw = dat[["B365D", "BWD", "IWD", "PSD", "WHD", "VCD"]]
awin = dat[["B365A", "BWA", "IWA", "PSA", "WHA", "VCA"]]

# Create value vectors
val = np.zeros(len(dat))  # Value indicator
odds = np.zeros(len(dat))  # Bet odds
OC = np.full(len(dat), -1)  # Outcome indicator
W = np.zeros(len(dat))  # Wins
L = np.zeros(len(dat))  # Losses

# Select the appropriate data set (e.g., hwin, draw, or awin)
data = hwin  # Select when evaluating home win bets

# Identify matches with potential value
for i in range(len(dat)):
    odds[i] = data.iloc[i].max()
    if dat.loc[i, "PSH"] < odds[i]:
        val[i] = 1  # Select for home win

# Specify wager value on each bet
stake = 10  # £10 wagered on the bet

# Compute winnings and losses
for i in range(len(dat)):
    if dat.loc[i, "FTR"] == "H" and val[i] > 0:
        OC[i] = 1  # Select for home win
    if OC[i] > 0:
        W[i] = stake * (odds[i] - 1)
    elif val[i] > 0 and OC[i] < 0:
        L[i] = stake

# Compute how much won and lost and total profit
Profit = W.sum() - L.sum()
print(Profit)

# %%
# Example 7.4

# Adjust hmax, dmax, and amax to store column names for the best odds
hmax = pd.DataFrame(index=dat.index, columns=["Hprob", "Hbm"])
dmax = pd.DataFrame(index=dat.index, columns=["Dprob", "Dbm"])
amax = pd.DataFrame(index=dat.index, columns=["Aprob", "Abm"])
prob = np.zeros(len(dat))  # Sum of implied probabilities
arb = np.zeros(len(dat))  # Arbitrage opportunity indicator

# Identify best odds offered on each match
for i in range(len(dat)):
    hmax.loc[i, "Hprob"] = round(1 / hwin.iloc[i].max(), 3)  # Best home win odds
    hmax.loc[i, "Hbm"] = hwin.iloc[i].idxmax()
    dmax.loc[i, "Dprob"] = round(1 / draw.iloc[i].max(), 3)  # Best draw odds
    dmax.loc[i, "Dbm"] = draw.iloc[i].idxmax()
    amax.loc[i, "Aprob"] = round(1 / awin.iloc[i].max(), 3)  # Best away win odds
    amax.loc[i, "Abm"] = awin.iloc[i].idxmax()
    prob[i] = float(hmax.loc[i, "Hprob"] + dmax.loc[i, "Dprob"] + amax.loc[i, "Aprob"])
    if prob[i] < 1:
        arb[i] = 1

# Compile arbitrage opportunity data frame
arb_temp = pd.concat([hmax, dmax, amax], axis=1)
arb_temp["Prob"] = prob
arb_temp["Arb"] = arb
arb_ops = pd.concat([dat[["Date", "HomeTeam", "AwayTeam", "FTR"]], arb_temp], axis=1)

# Ensure probability columns are numeric
arb_ops["Hprob"] = pd.to_numeric(arb_ops["Hprob"], errors="coerce")
arb_ops["Dprob"] = pd.to_numeric(arb_ops["Dprob"], errors="coerce")
arb_ops["Aprob"] = pd.to_numeric(arb_ops["Aprob"], errors="coerce")

# Compute odds from probabilities
arb_bets = arb_ops.copy()
arb_bets["Hodds"] = round(1 / arb_ops["Hprob"], 3)
arb_bets["Dodds"] = round(1 / arb_ops["Dprob"], 3)
arb_bets["Aodds"] = round(1 / arb_ops["Aprob"], 3)

# Specify nominal wager on each bet
nom_wager = 1000  # £1000

# Compute stakes to be placed on each bet
Hstake = arb_ops["Hprob"] * nom_wager
Dstake = arb_ops["Dprob"] * nom_wager
Astake = arb_ops["Aprob"] * nom_wager
wager = Hstake + Dstake + Astake

# Compute profits from arbitrage bets
profit = np.zeros(len(dat))
for i in range(len(dat)):
    if arb_bets.loc[i, "Arb"] > 0:
        if arb_bets.loc[i, "FTR"] == "H":
            profit[i] = (
                (Hstake[i] * (arb_bets.loc[i, "Hodds"] - 1)) - Dstake[i] - Astake[i]
            )
        elif arb_bets.loc[i, "FTR"] == "D":
            profit[i] = (
                (Dstake[i] * (arb_bets.loc[i, "Dodds"] - 1)) - Hstake[i] - Astake[i]
            )
        elif arb_bets.loc[i, "FTR"] == "A":
            profit[i] = (
                (Astake[i] * (arb_bets.loc[i, "Aodds"] - 1)) - Hstake[i] - Dstake[i]
            )

# Compute number of arbitrage bets made and profit for the season
nbets = arb_bets["Arb"].sum()
print(nbets)

tot_profit = profit.sum()
print(tot_profit)

# Average yield
approx_yield = round((100 * tot_profit / (nom_wager * nbets)), 2) if nbets > 0 else 0
print(approx_yield)
