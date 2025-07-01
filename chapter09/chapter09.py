# %%
# Chapter 9: Python code
# Copyright: Clive Beggs - 31st March 2023
# Converted from R to Python - Team Rating Systems

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import pinv
import requests
from io import StringIO

print("="*60)
print("CHAPTER 9: TEAM RATING SYSTEMS")
print("="*60)
print("This chapter demonstrates various team rating systems:")
print("- Colley rating system")
print("- Massey rating system") 
print("- Elo rating system")
print("="*60)

# %%
# Example 9.1: Mini-soccer league data setup
print("EXAMPLE 9.1: Goals-Based Network Analysis")
print("="*40)

# First we create the results for a mini-soccer league competition.
matches = [
    ["Team A", "Team E", 3, 2, "H"],  # Team A v Team E (score: 3-2)
    ["Team B", "Team F", 1, 1, "D"],  # Team B v Team F (score: 1-1)
    ["Team C", "Team G", 5, 2, "H"],  # Team C v Team G (score: 5-2)
    ["Team D", "Team H", 0, 1, "A"],  # Team D v Team H (score: 0-1)
    ["Team E", "Team D", 2, 3, "A"],  # Team E v Team D (score: 2-3)
    ["Team F", "Team C", 2, 1, "H"],  # Team F v Team C (score: 2-1)
    ["Team G", "Team B", 0, 0, "D"],  # Team G v Team B (score: 0-0)
    ["Team H", "Team A", 1, 3, "A"],  # Team H v Team A (score: 1-3)
    ["Team A", "Team F", 4, 2, "H"],  # Team A v Team F (score: 4-2)
    ["Team G", "Team D", 2, 2, "D"]   # Team G v Team D (score: 2-2)
]

mini = pd.DataFrame(matches, columns=["HomeTeam", "AwayTeam", "HG", "AG", "Results"])
mini["HG"] = pd.to_numeric(mini["HG"])  # Convert to integers
mini["AG"] = pd.to_numeric(mini["AG"])  # Convert to integers
print("Mini-soccer league results:")
print(mini)

# %%
# Setup team mappings and adjacency matrix
# Assign numerical values to individual teams
teams = sorted(mini["HomeTeam"].unique())
n = len(teams)
print(f"\nTeams in the league: {teams}")
print(f"Number of teams: {n}")

# Create team mapping
team_to_id = {team: i for i, team in enumerate(teams)}
HT = mini["HomeTeam"].map(team_to_id)
AT = mini["AwayTeam"].map(team_to_id)

# Create new matrix
X = np.column_stack([HT.values, AT.values, mini["HG"].values, mini["AG"].values])

# Populate adjacency matrix with weights
adj1 = np.zeros((n, n))
adj2 = np.zeros((n, n))
p = len(mini)

for k in range(p):
    i = X[k, 0]
    j = X[k, 1]
    if adj1[i, j] == 0:
        adj1[i, j] = X[k, 2]
    if adj2[j, i] == 0:
        adj2[j, i] = X[k, 3]

adj_mat = adj1 + adj2
print("\nAdjacency matrix (goals scored):")
print(pd.DataFrame(adj_mat, index=teams, columns=teams))

# %%
# Visualize the goals network
G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
mapping = {i: teams[i] for i in range(n)}
G = nx.relabel_nodes(G, mapping)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.title("Goals Scored Network", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# %%

# Example 9.2: Colley Rating System
print("EXAMPLE 9.2: Colley Rating System")
print("="*40)
print("Computing Colley ratings based on win-loss records...")

# Create adjacency matrix for who-played-who
WPWadj_mat = np.zeros((n, n))

for k in range(p):
    i = X[k, 0]
    j = X[k, 1]
    if WPWadj_mat[j, i] == 0:
        WPWadj_mat[j, i] = 1
    else:
        WPWadj_mat[j, i] += 1
    if WPWadj_mat[i, j] == 0:
        WPWadj_mat[i, j] = 1
    else:
        WPWadj_mat[i, j] += 1

print("Who-played-who adjacency matrix:")
print(pd.DataFrame(WPWadj_mat, index=teams, columns=teams))

# %%
# Produce the Colley matrix
sumrowplus2 = WPWadj_mat.sum(axis=1) + 2  # Compute the sum of the rows
d = np.diag(sumrowplus2)
C = (-1 * WPWadj_mat) + d  # This is the Colley matrix
print("\nColley matrix:")
print(pd.DataFrame(C, index=teams, columns=teams))

# %%
# Create win-lose vector
print("Creating win-lose vectors...")

# Count home results
home_results = mini.groupby(['HomeTeam', 'Results']).size().unstack(fill_value=0)
if 'A' not in home_results.columns:
    home_results['A'] = 0
if 'D' not in home_results.columns:
    home_results['D'] = 0
if 'H' not in home_results.columns:
    home_results['H'] = 0
home_results = home_results[['A', 'D', 'H']]  # Reorder columns
home_results.columns = ['HL', 'HD', 'HW']
print("\nHome results:")
print(home_results)

# Count away results  
away_results = mini.groupby(['AwayTeam', 'Results']).size().unstack(fill_value=0)
if 'A' not in away_results.columns:
    away_results['A'] = 0
if 'D' not in away_results.columns:
    away_results['D'] = 0
if 'H' not in away_results.columns:
    away_results['H'] = 0
away_results = away_results[['H', 'D', 'A']]  # Reorder columns (away perspective)
away_results.columns = ['AW', 'AD', 'AL']
print("\nAway results:")
print(away_results)

# %%
# Solve Colley system
# Ensure all teams are represented
all_teams_df = pd.DataFrame(index=teams)
home_results = all_teams_df.join(home_results, how='left').fillna(0)
away_results = all_teams_df.join(away_results, how='left').fillna(0)

# Compute win-lose vector
w = home_results['HW'] + away_results['AW']
l = home_results['HL'] + away_results['AL']
e = np.ones((n, 1))
v = e + 0.5 * (w.values - l.values).reshape(-1, 1)
print("Win-lose vector:")
print(v.flatten())

# Solve equation using pseudoinverse
Colley_r = pinv(C) @ v

# Compile team ratings
Colley_rat = pd.DataFrame({
    'Team': teams,
    'Rating': np.round(Colley_r.flatten(), 3)
})
print("\nColley ratings:")
print(Colley_rat)

# Rank teams according to Colley rating score
Colley_rank = Colley_rat.sort_values('Rating', ascending=False).reset_index(drop=True)
print("\nColley rankings:")
print(Colley_rank)

# %%
# Visualize Colley ratings
plt.figure(figsize=(10, 6))
plt.barh(range(len(Colley_rank)), Colley_rank['Rating'])
plt.yticks(range(len(Colley_rank)), Colley_rank['Team'])
plt.xlabel('Rating')
plt.title('Colley Ratings of Teams')
plt.xlim(0, 0.8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%

# Example 9.3: Massey Rating System
print("EXAMPLE 9.3: Massey Rating System")
print("="*40)
print("Computing Massey ratings based on goal differences...")

# Create identity matrix x2
i2 = np.eye(n) * 2

# Create Massey matrix
M = C - i2
print("Massey matrix:")
print(pd.DataFrame(M, index=teams, columns=teams))

# %%
# Compute overall goal differences for each club
y = np.zeros((n, 1))

for i in range(n):
    team = teams[i]
    tempH = mini[mini['HomeTeam'] == team]
    tempA = mini[mini['AwayTeam'] == team]
    GFH = tempH['HG'].sum()
    GFA = tempA['AG'].sum()
    GAH = tempH['AG'].sum()
    GAA = tempA['HG'].sum()
    GF = GFH + GFA
    GA = GAH + GAA
    y[i] = GF - GA

print("Goal differences vector:")
print(y.flatten())

# Compute Massey ratings
Massey_r = pinv(M) @ y

# Compile team ratings
Massey_rat = pd.DataFrame({
    'Team': teams,
    'Rating': np.round(Massey_r.flatten(), 3)
})
print("\nMassey ratings:")
print(Massey_rat)

# Rank teams according to Massey rating score
Massey_rank = Massey_rat.sort_values('Rating', ascending=False).reset_index(drop=True)
print("\nMassey rankings:")
print(Massey_rank)

# %%
# Visualize Massey ratings
plt.figure(figsize=(10, 6))
plt.barh(range(len(Massey_rank)), Massey_rank['Rating'])
plt.yticks(range(len(Massey_rank)), Massey_rank['Team'])
plt.xlabel('Rating')
plt.title('Massey Ratings of Teams')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%

# Example 9.4: Elo Rating System (Manual Implementation)
print("EXAMPLE 9.4: Elo Rating System - Manual Implementation")
print("="*40)
print("Computing dynamic Elo ratings match by match...")

Elo_dat = mini.copy()
nobs = len(Elo_dat)

# Create new columns for Elo points
Elo_dat['HTEP'] = np.nan
Elo_dat['ATEP'] = np.nan

# Populate Elo points based on results
for i in range(nobs):
    if Elo_dat.iloc[i]['Results'] == 'H':
        Elo_dat.iloc[i, Elo_dat.columns.get_loc('HTEP')] = 1
        Elo_dat.iloc[i, Elo_dat.columns.get_loc('ATEP')] = 0
    elif Elo_dat.iloc[i]['Results'] == 'D':
        Elo_dat.iloc[i, Elo_dat.columns.get_loc('HTEP')] = 0.5
        Elo_dat.iloc[i, Elo_dat.columns.get_loc('ATEP')] = 0.5
    else:
        Elo_dat.iloc[i, Elo_dat.columns.get_loc('HTEP')] = 0
        Elo_dat.iloc[i, Elo_dat.columns.get_loc('ATEP')] = 1

print("Elo data with match outcomes as points:")
print(Elo_dat)

# %%
# Setup Elo rating system
# Create empty matrix to store Elo rank results
ranks = np.zeros((n, nobs + 1))
ranks[:, 0] = 1200  # Initial Elo score of 1200
m = ranks.shape[1]

# Specify K-factor
K = 18.5

# Create new columns to store Elo results
Elo_dat['HomeProb'] = np.nan
Elo_dat['AwayProb'] = np.nan
Elo_dat['HomeRating'] = np.nan
Elo_dat['AwayRating'] = np.nan

print(f"Initial Elo ratings for all teams: {ranks[:, 0]}")
print(f"K-factor: {K}")

# %%
# Execute Elo algorithm match by match
print("Processing matches with Elo algorithm...")

# Populate the ranks matrix using the Elo algorithm
for i in range(nobs):
    home_team = Elo_dat.iloc[i]['HomeTeam']
    away_team = Elo_dat.iloc[i]['AwayTeam']
    fA = teams.index(home_team)  # Home team index
    fB = teams.index(away_team)  # Away team index
    
    if ranks[fA, i] == ranks[fB, i]:
        expA = 0.5  # Expected probability of home team
        expB = 0.5  # Expected probability of away team
    else:
        expA = 1 / (1 + 10**(-((ranks[fA, i] - ranks[fB, i]) / 400)))  # Expected probability of A
        expB = 1 / (1 + 10**(-((ranks[fB, i] - ranks[fA, i]) / 400)))  # Expected probability of B
    
    rA = ranks[fA, i] + K * (Elo_dat.iloc[i]['HTEP'] - expA)  # Elo algorithm applied to home team
    rB = ranks[fB, i] + K * (Elo_dat.iloc[i]['ATEP'] - expB)  # Elo algorithm applied to away team
    
    ranks[:, i + 1] = ranks[:, i]
    ranks[fA, i + 1] = rA
    ranks[fB, i + 1] = rB
    
    Elo_dat.iloc[i, Elo_dat.columns.get_loc('HomeProb')] = round(expA, 3)
    Elo_dat.iloc[i, Elo_dat.columns.get_loc('AwayProb')] = round(expB, 3)
    Elo_dat.iloc[i, Elo_dat.columns.get_loc('HomeRating')] = rA
    Elo_dat.iloc[i, Elo_dat.columns.get_loc('AwayRating')] = rB

# Display results
print("\nElo data with probabilities and updated ratings:")
print(Elo_dat)

# %%
# Analyze Elo rating evolution
print("Elo rating evolution across all matches:")
ranks_df = pd.DataFrame(np.round(ranks, 3), index=teams)
print(ranks_df)

# Compile final results
er = pd.DataFrame({
    'Team': teams,
    'Elo_Rating': ranks[:, -1]
})

# Rank teams in descending order
Elo_rank = er.sort_values('Elo_Rating', ascending=False).reset_index(drop=True)
print("\nFinal Elo rankings:")
print(Elo_rank)

# %%
# Visualize Elo rating evolution
plt.figure(figsize=(12, 8))
for i, team in enumerate(teams):
    plt.plot(range(nobs + 1), ranks[i, :], marker='o', label=team, linewidth=2)

plt.xlabel('Match Number')
plt.ylabel('Elo Rating')
plt.title('Elo Rating Evolution Throughout Season')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

# Example 9.5: Custom Elo Implementation Class
print("EXAMPLE 9.5: Elo System Class Implementation")
print("="*40)
print("Creating reusable Elo rating system...")

# Note: Python doesn't have an exact equivalent to R's elo package,
# so we'll implement a comprehensive version

class EloSystem:
    def __init__(self, k=18.5, initial_rating=1200):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}
        
    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, team_a, team_b, score_a, score_b):
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        # Convert match result to score (1 for win, 0.5 for draw, 0 for loss)
        if score_a > score_b:
            actual_a, actual_b = 1, 0
        elif score_a < score_b:
            actual_a, actual_b = 0, 1
        else:
            actual_a, actual_b = 0.5, 0.5
        
        new_rating_a = rating_a + self.k * (actual_a - expected_a)
        new_rating_b = rating_b + self.k * (actual_b - expected_b)
        
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
        
        return new_rating_a, new_rating_b

# Initialize Elo system
elo_system = EloSystem(k=18.5, initial_rating=1200)

# Process all matches
print("Processing matches with Elo class...")
for _, match in Elo_dat.iterrows():
    elo_system.update_ratings(
        match['HomeTeam'], 
        match['AwayTeam'], 
        match['HG'], 
        match['AG']
    )

# Get final ratings
elo_results = pd.DataFrame([
    {'Team': team, 'Rating': rating} 
    for team, rating in elo_system.ratings.items()
])
elo_results = elo_results.sort_values('Rating', ascending=False).reset_index(drop=True)
print("\nElo system final rankings:")
print(elo_results)

# %%
# Compare rating systems
print("RATING SYSTEMS COMPARISON")
print("="*40)

# Combine all rating systems for comparison
comparison_df = pd.DataFrame({
    'Team': teams,
    'Colley': Colley_rank.set_index('Team').loc[teams, 'Rating'].values,
    'Massey': Massey_rank.set_index('Team').loc[teams, 'Rating'].values,
    'Elo_Manual': Elo_rank.set_index('Team').loc[teams, 'Elo_Rating'].values,
    'Elo_Class': [elo_system.get_rating(team) for team in teams]
})

print("Comparison of all rating systems:")
print(comparison_df.round(3))

# %%
# Visualize rating systems comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Colley ratings
axes[0,0].barh(range(len(teams)), [elo_system.get_rating(team) for team in teams])
axes[0,0].set_yticks(range(len(teams)))
axes[0,0].set_yticklabels(teams)
axes[0,0].set_title('Elo Class Ratings')
axes[0,0].invert_yaxis()

# Massey ratings  
massey_vals = Massey_rank.set_index('Team').loc[teams, 'Rating'].values
axes[0,1].barh(range(len(teams)), massey_vals)
axes[0,1].set_yticks(range(len(teams)))
axes[0,1].set_yticklabels(teams)
axes[0,1].set_title('Massey Ratings')
axes[0,1].invert_yaxis()

# Colley ratings
colley_vals = Colley_rank.set_index('Team').loc[teams, 'Rating'].values
axes[1,0].barh(range(len(teams)), colley_vals)
axes[1,0].set_yticks(range(len(teams)))
axes[1,0].set_yticklabels(teams)
axes[1,0].set_title('Colley Ratings')
axes[1,0].invert_yaxis()

# Elo manual ratings
elo_manual_vals = Elo_rank.set_index('Team').loc[teams, 'Elo_Rating'].values
axes[1,1].barh(range(len(teams)), elo_manual_vals)
axes[1,1].set_yticks(range(len(teams)))
axes[1,1].set_yticklabels(teams)
axes[1,1].set_title('Elo Manual Ratings')
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.show()

# %%
# Example 9.6: Real Premier League Data Analysis
print("EXAMPLE 9.6: Premier League Elo Prediction System")
print("="*40)
print("Loading and analyzing real Premier League data...")

# Load data from URL
url = 'https://www.football-data.co.uk/mmz4281/2021/E0.csv'
response = requests.get(url)
mydata = pd.read_csv(StringIO(response.text)).head(380).iloc[:, :11]
print("Column names:", mydata.columns.tolist())

# Select relevant columns (adjust indices as needed)
dat = mydata.iloc[:, [3, 4, 5, 6, 7]].copy()  # HomeTeam, AwayTeam, HG, AG, Results
dat.columns = ["HomeTeam", "AwayTeam", "HG", "AG", "Results"]

# Create Elo points columns
dat['HTEP'] = np.nan
dat['ATEP'] = np.nan
ng = len(dat)

print(f"Total matches loaded: {ng}")
print(f"Unique teams: {dat['HomeTeam'].nunique()}")

# %%
# Prepare data for Elo analysis
# Populate empty columns
for i in range(ng):
    if dat.iloc[i]['Results'] == 'H':
        dat.iloc[i, dat.columns.get_loc('HTEP')] = 1
        dat.iloc[i, dat.columns.get_loc('ATEP')] = 0
    elif dat.iloc[i]['Results'] == 'D':
        dat.iloc[i, dat.columns.get_loc('HTEP')] = 0.5
        dat.iloc[i, dat.columns.get_loc('ATEP')] = 0.5
    else:
        dat.iloc[i, dat.columns.get_loc('HTEP')] = 0
        dat.iloc[i, dat.columns.get_loc('ATEP')] = 1

print("\nFirst 10 rows of Premier League data:")
print(dat.head(10))

# Divide data into training and testing sets
dat_train = dat.iloc[:210].copy()
dat_test = dat.iloc[210:230].copy()

print(f"Training matches: {len(dat_train)}")
print(f"Test matches: {len(dat_test)}")

# %%
# Build Elo model with home advantage
class EloSystemWithHomeAdvantage(EloSystem):
    def __init__(self, k=18.5, initial_rating=1200, home_advantage=30):
        super().__init__(k, initial_rating)
        self.home_advantage = home_advantage
    
    def expected_score_with_home_advantage(self, home_rating, away_rating):
        adjusted_home_rating = home_rating + self.home_advantage
        return 1 / (1 + 10**((away_rating - adjusted_home_rating) / 400))
    
    def predict_match(self, home_team, away_team):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        return self.expected_score_with_home_advantage(home_rating, away_rating)

# Initialize Elo system with home advantage
epl_elo = EloSystemWithHomeAdvantage(k=18.5, initial_rating=1200, home_advantage=30)

print(f"Elo system initialized with home advantage: {epl_elo.home_advantage}")

# %%
# Train the model and collect predictions
print("Training Elo model on first 210 matches...")

predictions_train = []
for _, match in dat_train.iterrows():
    # Make prediction before updating ratings
    pred = epl_elo.predict_match(match['HomeTeam'], match['AwayTeam'])
    predictions_train.append(pred)
    
    # Update ratings
    epl_elo.update_ratings(
        match['HomeTeam'], 
        match['AwayTeam'], 
        match['HG'], 
        match['AG']
    )

print(f"First 15 training predictions: {predictions_train[:15]}")

# %%
# Evaluate training performance
# Set thresholds
p_win = 0.6
p_lose = 0.4

# Compile match prediction results for training data
train_pred = dat_train.iloc[:, :5].copy()
train_pred['pred_train'] = predictions_train
train_pred['Prediction'] = ''  # Initialize as string
train_pred['Outcome'] = 0

for i in range(len(train_pred)):
    if train_pred.iloc[i]['pred_train'] >= p_win:
        train_pred.iloc[i, train_pred.columns.get_loc('Prediction')] = 'H'
    elif train_pred.iloc[i]['pred_train'] <= p_lose:
        train_pred.iloc[i, train_pred.columns.get_loc('Prediction')] = 'A'
    else:
        train_pred.iloc[i, train_pred.columns.get_loc('Prediction')] = 'D'
    
    if train_pred.iloc[i]['Results'] == train_pred.iloc[i]['Prediction']:
        train_pred.iloc[i, train_pred.columns.get_loc('Outcome')] = 1
    else:
        train_pred.iloc[i, train_pred.columns.get_loc('Outcome')] = 0

print("\nFirst 10 training predictions:")
print(train_pred.head(10))
print("\nLast 10 training predictions:")
print(train_pred.tail(10))

# Check performance of the model
train_check = train_pred[train_pred['Prediction'] != 'D'].copy()  # Remove draws
n_pred = len(train_check)
n_correct = train_check['Outcome'].sum()
train_accuracy = n_correct / n_pred if n_pred > 0 else 0
print(f"\nTraining accuracy (excluding draws): {train_accuracy:.3f}")

# %%
# Test the model
print("Testing model on matches 211-230...")

# Make predictions on test data
predictions_test = []
for _, match in dat_test.iterrows():
    pred = epl_elo.predict_match(match['HomeTeam'], match['AwayTeam'])
    predictions_test.append(pred)

# Compile test predictions
test_pred = dat_test.iloc[:, :5].copy()
test_pred['pred_test'] = predictions_test
test_pred['Prediction'] = ''  # Initialize as string
test_pred['Outcome'] = 0

for i in range(len(test_pred)):
    if test_pred.iloc[i]['pred_test'] >= p_win:
        test_pred.iloc[i, test_pred.columns.get_loc('Prediction')] = 'H'
    elif test_pred.iloc[i]['pred_test'] <= p_lose:
        test_pred.iloc[i, test_pred.columns.get_loc('Prediction')] = 'A'
    else:
        test_pred.iloc[i, test_pred.columns.get_loc('Prediction')] = 'D'
    
    if test_pred.iloc[i]['Results'] == test_pred.iloc[i]['Prediction']:
        test_pred.iloc[i, test_pred.columns.get_loc('Outcome')] = 1
    else:
        test_pred.iloc[i, test_pred.columns.get_loc('Outcome')] = 0

print("\nTest data predictions:")
print(test_pred)

# Calculate test accuracy
test_check = test_pred[test_pred['Prediction'] != 'D'].copy()
test_n_pred = len(test_check)
test_n_correct = test_check['Outcome'].sum()
test_accuracy = test_n_correct / test_n_pred if test_n_pred > 0 else 0
print(f"\nTest accuracy (excluding draws): {test_accuracy:.3f}")

# %%
# Visualize prediction performance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training vs Test Accuracy
accuracies = [train_accuracy, test_accuracy]
ax1.bar(['Training', 'Test'], accuracies, color=['lightblue', 'lightcoral'])
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim(0, 1)

# Prediction distribution
train_pred_dist = train_pred['Prediction'].value_counts()
ax2.pie(train_pred_dist.values, labels=train_pred_dist.index, autopct='%1.1f%%')
ax2.set_title('Training Prediction Distribution')

plt.tight_layout()
plt.show()

# %%
print("\n" + "="*60)
print("CHAPTER 9: TEAM RATING SYSTEMS - COMPLETE")
print("="*60)
print("Key concepts covered:")
print("✓ Colley rating system (win-loss based)")
print("✓ Massey rating system (goal difference based)")
print("✓ Elo rating system (dynamic, match-by-match)")
print("✓ Custom Elo implementation with classes")
print("✓ Real Premier League data analysis")
print("✓ Model training and testing procedures")
print("✓ Prediction accuracy evaluation")
print("✓ Home advantage incorporation")
print("="*60)
