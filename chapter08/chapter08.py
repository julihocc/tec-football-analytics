# %%
# Chapter 8: Python code
# Copyright: Clive Beggs - 31st March 2023
# Network Analysis in Football

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("CHAPTER 8: NETWORK ANALYSIS IN FOOTBALL")
print("="*60)

# %%
# Example 8.1: Mini-soccer competition results
print("EXAMPLE 8.1: Who-Played-Who Network")
print("="*40)

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

# %%
# Harvest team names and create adjacency matrix
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

# %%
# Visualize who-played-who network
G1 = nx.from_numpy_array(adj_mat1, create_using=nx.DiGraph)
mapping = {i: teams[i] for i in range(len(teams))}
G1 = nx.relabel_nodes(G1, mapping)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G1, seed=42)
nx.draw(G1, pos, with_labels=True, node_color='lightblue', 
        node_size=1500, font_size=10, arrows=True, edge_color='black')
plt.title("Who-Played-Who Network", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# %%

# Example 8.2: Goals-based adjacency matrix
print("EXAMPLE 8.2: Goals-Based Network")
print("="*40)

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

# %%
# Visualize goals-based network
G2 = nx.from_numpy_array(adj_mat2, create_using=nx.DiGraph)
mapping = {i: teams[i] for i in range(len(teams))}
G2 = nx.relabel_nodes(G2, mapping)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G2, seed=42)
nx.draw(G2, pos, with_labels=True, node_color='lightcoral', 
        node_size=1500, font_size=10, arrows=True, edge_color='black')

# Add edge labels for goals
edge_labels = {}
for (u, v, d) in G2.edges(data=True):
    if d['weight'] > 0:
        edge_labels[(u, v)] = int(d['weight'])

nx.draw_networkx_edge_labels(G2, pos, edge_labels, font_size=8)
plt.title("Goals Scored Network", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# %%

# Example 8.3: Create sample passing networks for demonstration
print("EXAMPLE 8.3: Passing Networks Analysis")
print("="*40)
print("Simulating passing data for Spain vs Netherlands...")

# Create more realistic passing matrices
np.random.seed(42)  # For reproducible results

# Spain - possession-based style (11 players)
adj_T1 = np.random.poisson(2, size=(11, 11))
for i in range(11):
    adj_T1[i, i] = 0  # No self-passes
    # Add structure - midfielders pass more
    if i in [4, 6, 8]:  # Midfield positions
        adj_T1[i, :] = adj_T1[i, :] + np.random.poisson(1, 11)
        adj_T1[i, i] = 0

# Netherlands - more direct style (11 players)
adj_T2 = np.random.poisson(1.5, size=(11, 11))
for i in range(11):
    adj_T2[i, i] = 0  # No self-passes
    # Add structure - more direct passing
    if i in [2, 3, 9, 10]:  # Defensive and attacking positions
        adj_T2[i, :] = adj_T2[i, :] + np.random.poisson(0.8, 11)
        adj_T2[i, i] = 0

print(f"Spain total passes: {adj_T1.sum()}")
print(f"Netherlands total passes: {adj_T2.sum()}")

# Create graphs
G_T1 = nx.from_numpy_array(adj_T1, create_using=nx.DiGraph)
G_T2 = nx.from_numpy_array(adj_T2, create_using=nx.DiGraph)

# Assign player labels
spain_players = [f"ESP_{i+1}" for i in range(11)]
netherlands_players = [f"NED_{i+1}" for i in range(11)]

G_T1 = nx.relabel_nodes(G_T1, {i: spain_players[i] for i in range(11)})
G_T2 = nx.relabel_nodes(G_T2, {i: netherlands_players[i] for i in range(11)})

# %%
# Visualize passing networks
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Spain network
plt.sca(ax1)
pos1 = nx.kamada_kawai_layout(G_T1)
nx.draw(G_T1, pos1, with_labels=True, node_color="red", 
        edge_color="darkred", node_size=500, font_size=8,
        alpha=0.8, arrows=True, ax=ax1)
ax1.set_title("Spain Passing Network", fontsize=12, fontweight='bold')

# Netherlands network
plt.sca(ax2)
pos2 = nx.kamada_kawai_layout(G_T2)
nx.draw(G_T2, pos2, with_labels=True, node_color="orange", 
        edge_color="darkorange", node_size=500, font_size=8,
        alpha=0.8, arrows=True, ax=ax2)
ax2.set_title("Netherlands Passing Network", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %%

# Example 8.4: Graph statistics
print("EXAMPLE 8.4: Network Statistics Analysis")
print("="*40)

# Calculate network metrics for Spain
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
res_T1 = [nv_T1, ne_T1, round(den_T1, 3), dia_T1, apl_T1]

# Calculate network metrics for Netherlands
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
res_T2 = [nv_T2, ne_T2, round(den_T2, 3), dia_T2, apl_T2]

# Compile comparative statistics
results = pd.DataFrame({"Spain": res_T1, "Netherlands": res_T2},
                        index=["Number of vertices", "Number of edges", "Graph density", 
                               "Graph diameter", "Average path length"])
print("\nComparative Graph Statistics:")
print(results)

# %%
# Visualize key network metrics
metrics_to_plot = ["Number of edges", "Graph density"]
spain_values = [res_T1[1], res_T1[2]]
netherlands_values = [res_T2[1], res_T2[2]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Number of edges comparison
ax1.bar(['Spain', 'Netherlands'], [spain_values[0], netherlands_values[0]], 
        color=['red', 'orange'], alpha=0.7)
ax1.set_title('Number of Edges Comparison')
ax1.set_ylabel('Number of Edges')

# Graph density comparison
ax2.bar(['Spain', 'Netherlands'], [spain_values[1], netherlands_values[1]], 
        color=['red', 'orange'], alpha=0.7)
ax2.set_title('Graph Density Comparison')
ax2.set_ylabel('Density')

plt.tight_layout()
plt.show()

# %%

# Player statistics using PageRank and HITS
print("EXAMPLE 8.4b: Player Centrality Analysis")
print("="*40)

# Spain player analysis
print("Analyzing Spain players...")
pr_T1 = nx.pagerank(G_T1)
try:
    hub_T1, auth_T1 = nx.hits(G_T1)
    players_T1 = pd.DataFrame({
        "PageRank": [round(pr_T1[player], 3) for player in spain_players], 
        "Authority": [round(auth_T1[player], 3) for player in spain_players]
    })
    players_T1.index = spain_players
    print("\nSpain player centrality statistics:")
    print(players_T1)
except Exception as e:
    print(f"Could not compute HITS algorithm for Spain: {e}")
    players_T1 = pd.DataFrame({"PageRank": [round(pr_T1[player], 3) for player in spain_players]})
    players_T1.index = spain_players
    print("\nSpain PageRank statistics:")
    print(players_T1)

# %%
# Visualize Spain player centrality
if 'Authority' in players_T1.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    players_T1.sort_values('PageRank', ascending=True).plot(kind='barh', y='PageRank', 
                                                           ax=ax1, color='red', alpha=0.7)
    ax1.set_title('Spain - PageRank Centrality')
    ax1.set_xlabel('PageRank Score')
    
    players_T1.sort_values('Authority', ascending=True).plot(kind='barh', y='Authority', 
                                                            ax=ax2, color='darkred', alpha=0.7)
    ax2.set_title('Spain - Authority Centrality')
    ax2.set_xlabel('Authority Score')
    
    plt.tight_layout()
    plt.show()
else:
    plt.figure(figsize=(10, 6))
    players_T1.sort_values('PageRank', ascending=True).plot(kind='barh', color='red', alpha=0.7)
    plt.title('Spain - PageRank Centrality')
    plt.xlabel('PageRank Score')
    plt.tight_layout()
    plt.show()

# %%
# Netherlands player analysis
print("Analyzing Netherlands players...")
pr_T2 = nx.pagerank(G_T2)
try:
    hub_T2, auth_T2 = nx.hits(G_T2)
    players_T2 = pd.DataFrame({
        "PageRank": [round(pr_T2[player], 3) for player in netherlands_players], 
        "Authority": [round(auth_T2[player], 3) for player in netherlands_players]
    })
    players_T2.index = netherlands_players
    print("\nNetherlands player centrality statistics:")
    print(players_T2)
except Exception as e:
    print(f"Could not compute HITS algorithm for Netherlands: {e}")
    players_T2 = pd.DataFrame({"PageRank": [round(pr_T2[player], 3) for player in netherlands_players]})
    players_T2.index = netherlands_players
    print("\nNetherlands PageRank statistics:")
    print(players_T2)

# %%
# Visualize Netherlands player centrality
if 'Authority' in players_T2.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    players_T2.sort_values('PageRank', ascending=True).plot(kind='barh', y='PageRank', 
                                                           ax=ax1, color='orange', alpha=0.7)
    ax1.set_title('Netherlands - PageRank Centrality')
    ax1.set_xlabel('PageRank Score')
    
    players_T2.sort_values('Authority', ascending=True).plot(kind='barh', y='Authority', 
                                                            ax=ax2, color='darkorange', alpha=0.7)
    ax2.set_title('Netherlands - Authority Centrality')
    ax2.set_xlabel('Authority Score')
    
    plt.tight_layout()
    plt.show()
else:
    plt.figure(figsize=(10, 6))
    players_T2.sort_values('PageRank', ascending=True).plot(kind='barh', color='orange', alpha=0.7)
    plt.title('Netherlands - PageRank Centrality')
    plt.xlabel('PageRank Score')
    plt.tight_layout()
    plt.show()

# %%

# Example 8.5: Relationships between managers and clubs
print("EXAMPLE 8.5: Manager-Club Bipartite Network")
print("="*40)

clubs = ["A", "A", "A", "B", "B", "B", "C", "D", "D", "E", "F"]
managers = ["m1", "m2", "m7", "m3", "m4", "m5", "m5", "m6", "m7", "m2", "m6"]

# Create a DataFrame
relationships = pd.DataFrame({"Club": clubs, "Manager": managers})
print("\nManager-Club relationships:")
print(relationships)

# Create summary statistics
summary = relationships.groupby(['Manager', 'Club']).size().reset_index(name='Relationships')
print("\nRelationship summary:")
for _, row in summary.iterrows():
    print(f"Manager {row['Manager']} worked with Club {row['Club']}: {row['Relationships']} time(s)")

# %%
# Create and analyze the bipartite network
g = nx.Graph()
for _, row in relationships.iterrows():
    g.add_edge(row["Club"], row["Manager"])

# Set node attributes
unique_clubs = sorted(list(set(clubs)))
unique_managers = sorted(list(set(managers)))

print(f"\nClubs: {unique_clubs}")
print(f"Managers: {unique_managers}")

for node in g.nodes:
    if node in unique_clubs:
        g.nodes[node]["type"] = 1  # Club
    else:
        g.nodes[node]["type"] = 2  # Manager

# %%
# Visualize the bipartite network
plt.figure(figsize=(12, 8))

# Use bipartite layout for better visualization
try:
    pos = nx.bipartite_layout(g, unique_clubs)
except:
    pos = nx.spring_layout(g, seed=42)

# Draw clubs and managers separately
club_nodes = [node for node in g.nodes if g.nodes[node]["type"] == 1]
manager_nodes = [node for node in g.nodes if g.nodes[node]["type"] == 2]

# Draw clubs as squares
nx.draw_networkx_nodes(g, pos, nodelist=club_nodes, node_color='lightblue', 
                      node_shape='s', node_size=1200, alpha=0.8)

# Draw managers as circles  
nx.draw_networkx_nodes(g, pos, nodelist=manager_nodes, node_color='lightcoral', 
                      node_shape='o', node_size=1200, alpha=0.8)

# Draw edges and labels
nx.draw_networkx_edges(g, pos, edge_color="gray", alpha=0.6)
nx.draw_networkx_labels(g, pos, font_size=10, font_weight="bold")

plt.title("Manager-Club Relationship Network\n(Squares = Clubs, Circles = Managers)", 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# Network analysis summary
print("\nNetwork Analysis Summary:")
print(f"Number of clubs: {len(unique_clubs)}")
print(f"Number of managers: {len(unique_managers)}")
print(f"Number of relationships: {len(relationships)}")
print(f"Network density: {nx.density(g):.3f}")

# Find most connected nodes
degree_centrality = nx.degree_centrality(g)
print("\nMost connected entities:")
sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
for node, centrality in sorted_centrality[:5]:
    node_type = "Club" if node in unique_clubs else "Manager"
    print(f"{node_type} {node}: {centrality:.3f}")

# %%
print("\n" + "="*60)
print("CHAPTER 8: NETWORK ANALYSIS - COMPLETE")
print("="*60)
print("Key concepts covered:")
print("✓ Who-played-who networks from match data")
print("✓ Goal-weighted adjacency matrices")
print("✓ Passing network visualization and analysis")
print("✓ Network statistics (density, diameter, path length)")
print("✓ Player centrality measures (PageRank, Authority)")
print("✓ Bipartite manager-club relationship networks")
print("✓ Network density and centrality analysis")
print("="*60)
