# %%
# load single_task and pairwise task affinities 
import os
import numpy as np
import pandas as pd
cur_set = ["bfs", "dfs",  "bellman_ford", "mst_prim", "floyd_warshall", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "dijkstra", "dag_shortest_paths"]
# ["dijkstra", "bfs", "dfs", "bellman_ford"]
# ["bfs", "dfs", "bridges", "strongly_connected_components", "mst_kruskal", "dag_shortest_paths"]
# ["bubble_sort", "insertion_sort", "heapsort", "quicksort", "mst_prim", "mst_kruskal", "dijkstra","bfs", "dfs", "bellman_ford"]
# ["bfs", "dfs",  "bellman_ford", "mst_prim", "floyd_warshall", "topological_sort", "articulation_points", "bridges", "strongly_connected_components", "mst_kruskal", "dijkstra", "dag_shortest_paths"]

# Construct specific filename
results_dir = './results'
csv_path = os.path.join(results_dir, 'processor_edge_t_layers_5_dim_192_full_training_pair_tasks_edge_t_v2.csv')

# Load the results
df = pd.read_csv(csv_path)

# We'll focus on validation accuracy as the metric
df = df[df['split'] == 'test']
df = df[df['metric'] == 'score']

single_csv_path = os.path.join(results_dir, 'processor_gatv2_layers_3_dim_192_full_training_single_task.csv')
df_single = pd.read_csv(single_csv_path)
df_single = df_single[df_single['split'] == 'test']
df_single = df_single[df_single['metric'] == 'score']
# concatenate single task results to the pairwise dataframe
df = pd.concat([df, df_single], ignore_index=True)


# Initialize affinity matrix
n = len(cur_set)
cur_set = list(cur_set)  # Convert to list for consistent indexing
affinity_matrix = np.zeros((n, n))

# For each pair of algorithms
for i in range(n):
    for j in range(n):
        algo1, algo2 = cur_set[i], cur_set[j]
        
        # Find subsets containing both algorithms
        mask = df['training_algorithms'].apply(lambda x: algo1 in x and algo2 in x)
        relevant_results = df[mask]
        
        if len(relevant_results) > 0:
            # Average the performance when both algorithms are present
            # Average the performance when both algorithms are present
            joint_performance = relevant_results[
                (relevant_results['target_algorithm'] == algo1) 
            ]['value'].mean()
            affinity_matrix[i, j] = joint_performance


            joint_performance = relevant_results[
                (relevant_results['target_algorithm'] == algo2) 
            ]['value'].mean()
            affinity_matrix[j, i] = joint_performance

# for i in range(n):
#     for j in range(n):
#         if affinity_matrix[i, j] == 0:
#             affinity_matrix[i, j] = affinity_matrix[i, i]

#%%
# Initialize affinity matrix
n = len(cur_set)
cur_set = list(cur_set)  # Convert to list for consistent indexing
std_matrix = np.zeros((n, n))

# For each pair of algorithms
for i in range(n):
    for j in range(n):
        algo1, algo2 = cur_set[i], cur_set[j]
        
        # Find subsets containing both algorithms
        mask = df['training_algorithms'].apply(lambda x: algo1 in x and algo2 in x)
        relevant_results = df[mask]
        
        if len(relevant_results) > 0:
            # Average the performance when both algorithms are present
            # Average the performance when both algorithms are present
            joint_performance = relevant_results[
                (relevant_results['target_algorithm'] == algo1) 
            ]['std'].mean()
            std_matrix[i, j] = joint_performance


            joint_performance = relevant_results[
                (relevant_results['target_algorithm'] == algo2) 
            ]['std'].mean()
            std_matrix[j, i] = joint_performance


# %%
from clustering import run_sdp_clustering, compute_average_density, run_regularized_sdp_clustering
k = 2
_, assignment = run_sdp_clustering(affinity_matrix, k)
print(assignment)
for cluster_id in range(len(assignment)):
    cluster_members = assignment[cluster_id]
    print(f'Cluster {cluster_id}: {[cur_set[i] for i in cluster_members]}')
density = compute_average_density(affinity_matrix, assignment)
print(f'k={k}, density={density}')