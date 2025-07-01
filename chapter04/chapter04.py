# %%
"""
Chapter 4: Python code
Converted from R script by Clive Beggs 7th March 2023
Python conversion date: June 28, 2025
"""

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# %%
# Ensure we're in the correct directory for relative paths to work.
current_dir = os.getcwd()
if os.path.basename(current_dir) != "chapter04":
    possible_dirs = [
        "chapter04",
        os.path.join("migrating-to-python", "chapter04"),
        os.path.join("..", "chapter04"),
        os.path.join(".", "chapter04"),
    ]
    for dir_path in possible_dirs:
        if os.path.exists(dir_path) and os.path.exists(
            os.path.join(dir_path, "chapter04.py")
        ):
            os.chdir(dir_path)
            break

# %%
# Load data
url = "https://www.football-data.co.uk/mmz4281/2021/E0.csv"
EPL2020_data = pd.read_csv(url).iloc[:380, :16]

# %%
# Inspect data
print(EPL2020_data.columns)

# %%
# Create a working data frame
dat = EPL2020_data[
    [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
    ]
]
print(dat.head())

# %%
# Create new variables populated with NaNs and zeros
dat["GD"] = np.nan
dat["TG"] = np.nan
dat["HTSR"] = np.nan
dat["ATSR"] = np.nan
dat["HPts"] = 0
dat["APts"] = 0

# %%
# Populate the variables
dat["GD"] = dat["FTHG"] - dat["FTAG"]
dat["TG"] = dat["FTHG"] + dat["FTAG"]
dat["HTSR"] = round(dat["HS"] / (dat["HS"] + dat["AS"]), 3)
dat["ATSR"] = round(dat["AS"] / (dat["AS"] + dat["HS"]), 3)

# %%
# Compute home and away points awarded per match
for i in range(len(dat)):
    if dat.loc[i, "FTR"] == "H":
        dat.loc[i, "HPts"] = 3
    elif dat.loc[i, "FTR"] == "A":
        dat.loc[i, "APts"] = 3
    elif dat.loc[i, "FTR"] == "D":
        dat.loc[i, "HPts"] = 1
        dat.loc[i, "APts"] = 1

# %%
# Rename variables
dat.rename(columns={"FTHG": "HG", "FTAG": "AG", "FTR": "Result"}, inplace=True)

# %%
# Display augmented data frame
print(dat.head())

# %%
# Specify target team
study_team = "Liverpool"

# Extract data for the target team
home_matches = dat[dat["HomeTeam"] == study_team]
away_matches = dat[dat["AwayTeam"] == study_team]

# Add a 'status' variable
home_matches = home_matches.assign(status="home")
away_matches = away_matches.assign(status="away")

# Combine home and away matches
target_team_matches = pd.concat([home_matches, away_matches])
print(target_team_matches.head())
