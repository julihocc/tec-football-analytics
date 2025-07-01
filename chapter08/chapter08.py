# Chapter 8: Python code
# Copyright: Clive Beggs - 31st March 2023

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Example 8.1: Mini-soccer competition results
match_data = [
    ("Midtown", "Halton", 3, 2, "H"),
    ("Oldbury", "Newtown", 3, 1, "H"),
    ("Longbury", "Scotsway", 4, 2, "H"),
    ("Tilcome", "Oldbury", 0, 1, "A"),
    ("Scotsway", "Tilcome", 3, 3, "D"),
    ("Midtown", "Longbury", 2, 2, "D"),
    ("Halton", "Midtown", 1, 2, "A"),
    ("Newtown", "Longbury", 1, 3, "A"),
    ("Halton", "Scotsway", 4, 2, "H"),
    ("Oldbury", "Midtown", 4, 1, "H")
]

columns = ["HomeTeam", "AwayTeam", "HG", "AG", "Results"]
mini = pd.DataFrame(match_data, columns=columns)
print("Mini-soccer competition results:")
print(mini)

# Harvest team names
teams = sorted(mini['HomeTeam'].unique())
n = len(teams)
print(f"\nTeams: {teams}")

# Create adjacency matrix for who-played-who
adj_tab1 = pd.crosstab(mini['HomeTeam'], mini['AwayTeam'])
print("\nWho-played-who table:")
print(adj_tab1)

# Convert to numpy matrix
adj_mat1 = adj_tab1.values
print("\nAdjacency matrix (who-played-who):")
print(adj_mat1)

# Example 8.2: Goals-based adjacency matrix
# Create numerical team mappings
team_to_num = {team: i for i, team in enumerate(teams)}
home_team_nums = mini['HomeTeam'].map(team_to_num)
away_team_nums = mini['AwayTeam'].map(team_to_num)

# Create adjacency matrix with goal weights
adj1 = np.zeros((n, n))
adj2 = np.zeros((n, n))

for idx in mini.index:
    i = home_team_nums[idx]
    j = away_team_nums[idx] 
    if adj1[i, j] == 0:
        adj1[i, j] = mini.loc[idx, 'HG']
    if adj2[j, i] == 0:
        adj2[j, i] = mini.loc[idx, 'AG']

adj_mat2 = adj1 + adj2
adj_mat2_df = pd.DataFrame(adj_mat2, index=teams, columns=teams)
print("\nGoals-based adjacency matrix:")
print(adj_mat2_df)

# Example 8.3: Create sample adjacency matrices for demonstration
# (In practice, these would be loaded from CSV files)
np.random.seed(42)  # For reproducible results
adj_T1 = np.random.randint(0, 5, size=(14, 14))  # Spain's adjacency matrix
adj_T2 = np.random.randint(0, 5, size=(14, 14))  # Netherlands' adjacency matrix

# Make them have some structure
for i in range(14):
    adj_T1[i, i] = 0  # No self-loops
    adj_T2[i, i] = 0

# Create graphs using modern NetworkX syntax
G_T1 = nx.from_numpy_array(adj_T1, create_using=nx.DiGraph)
G_T2 = nx.from_numpy_array(adj_T2, create_using=nx.DiGraph)

# Plot graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
pos1 = nx.kamada_kawai_layout(G_T1)
nx.draw(G_T1, pos1, with_labels=True, node_color="white", 
        edge_color="black", node_size=300, font_size=8)
plt.title("Spain")

plt.subplot(1, 2, 2)
pos2 = nx.kamada_kawai_layout(G_T2)
nx.draw(G_T2, pos2, with_labels=True, node_color="white", 
        edge_color="black", node_size=300, font_size=8)
plt.title("Netherlands")

plt.tight_layout()
plt.savefig('passing_networks.png', dpi=150, bbox_inches='tight')
plt.close()
print("Passing networks saved as 'passing_networks.png'")

# Example 8.4: Graph statistics
print("\n=== Graph Statistics ===")

# Spain
nv_T1 = G_T1.number_of_nodes()
ne_T1 = G_T1.number_of_edges()
den_T1 = nx.density(G_T1)
try:
    if nx.is_weakly_connected(G_T1):
        dia_T1 = nx.diameter(G_T1.to_undirected())
        apl_T1 = nx.average_shortest_path_length(G_T1.to_undirected())
    else:
        dia_T1 = "N/A (not connected)"
        apl_T1 = "N/A (not connected)"
except:
    dia_T1 = "N/A"
    apl_T1 = "N/A"
res_T1 = [nv_T1, ne_T1, den_T1, dia_T1, apl_T1]

# Netherlands
nv_T2 = G_T2.number_of_nodes()
ne_T2 = G_T2.number_of_edges()
den_T2 = nx.density(G_T2)
try:
    if nx.is_weakly_connected(G_T2):
        dia_T2 = nx.diameter(G_T2.to_undirected())
        apl_T2 = nx.average_shortest_path_length(G_T2.to_undirected())
    else:
        dia_T2 = "N/A (not connected)"
        apl_T2 = "N/A (not connected)"
except:
    dia_T2 = "N/A"
    apl_T2 = "N/A"
res_T2 = [nv_T2, ne_T2, den_T2, dia_T2, apl_T2]

# Compile descriptive statistics
results = pd.DataFrame({"Spain": res_T1, "Netherlands": res_T2},
                        index=["Number of vertices", "Number of edges", "Graph density", "Graph diameter", "Average path length"])
print("\nGraph statistics:")
print(results)

# Player statistics using PageRank and HITS
print("\n=== Player Statistics ===")

# Spain
pr_T1 = nx.pagerank(G_T1)
try:
    hub_T1, auth_T1 = nx.hits(G_T1)
    players_T1 = pd.DataFrame({
        "PageRank": list(pr_T1.values()), 
        "Authority": list(auth_T1.values())
    }).round(3)
    players_T1.index = list(pr_T1.keys())
    print("\nSpain player statistics:")
    print(players_T1)
except Exception as e:
    print(f"Could not compute HITS algorithm for Spain: {e}")
    players_T1 = pd.DataFrame({"PageRank": list(pr_T1.values())}).round(3)
    players_T1.index = list(pr_T1.keys())
    print("\nSpain PageRank only:")
    print(players_T1)

# Netherlands
pr_T2 = nx.pagerank(G_T2)
try:
    hub_T2, auth_T2 = nx.hits(G_T2)
    players_T2 = pd.DataFrame({
        "PageRank": list(pr_T2.values()), 
        "Authority": list(auth_T2.values())
    }).round(3)
    players_T2.index = list(pr_T2.keys())
    print("\nNetherlands player statistics:")
    print(players_T2)
except Exception as e:
    print(f"Could not compute HITS algorithm for Netherlands: {e}")
    players_T2 = pd.DataFrame({"PageRank": list(pr_T2.values())}).round(3)
    players_T2.index = list(pr_T2.keys())
    print("\nNetherlands PageRank only:")
    print(players_T2)

# Example 8.5: Relationships between managers and clubs
print("\n=== Manager-Club Relationships ===")

clubs = ["A", "A", "A", "B", "B", "B", "C", "D", "D", "E", "F"]
managers = ["m1", "m2", "m7", "m3", "m4", "m5", "m5", "m6", "m7", "m2", "m6"]

# Create a DataFrame
relationships = pd.DataFrame({"Club": clubs, "Manager": managers})
print("\nManager-Club relationships:")
print(relationships)

# Convert to graph object
g = nx.Graph()
for _, row in relationships.iterrows():
    g.add_edge(row["Club"], row["Manager"])

# Set node attributes for type (1 for clubs, 2 for managers)
unique_clubs = list(set(clubs))
unique_managers = list(set(managers))

for node in g.nodes:
    if node in unique_clubs:
        g.nodes[node]["type"] = 1
    else:
        g.nodes[node]["type"] = 2

# Define color mappings
node_colors = []
for node in g.nodes:
    if g.nodes[node]["type"] == 1:
        node_colors.append("lightgray")
    else:
        node_colors.append("white")

# Plot the bipartite graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(g, seed=42)
nx.draw(g, pos, with_labels=True, node_color=node_colors, 
        edge_color="black", node_size=1000, font_size=10, 
        font_weight="bold")
plt.title("Manager-Club Relationship Network")
plt.savefig('manager_club_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("Manager-Club network saved as 'manager_club_network.png'")

print("\nScript completed successfully!")
