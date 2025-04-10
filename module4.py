import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("team_stats_2003_2023.csv")
df['year'] = df['year'].astype(int)
df = df[df['year'] == 2022]

columns_to_use = [
    'win_loss_perc', 'points', 'points_opp', 'points_diff',
    'mov', 'total_yards', 'yds_per_play_offense', 'turnovers',
    'first_down', 'pass_yds', 'pass_td', 'pass_int',
    'rush_yds', 'rush_td', 'rush_yds_per_att', 'penalties',
    'score_pct', 'turnover_pct', 'exp_pts_tot'
]

df_cluster = df[columns_to_use].copy()
df_cluster.fillna(0, inplace=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

inertia = []
k_values = range(1, 11)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

key_metrics = ['win_loss_perc', 'points_diff']
metric_labels = ['Win Percentage', 'Points Differential']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, metric in enumerate(key_metrics):
    cluster_means = df.groupby('cluster')[metric].mean()
    
    ax = axes[i]
    bars = ax.bar(cluster_means.index, cluster_means.values, color=colors)
    
    ax.set_title(metric_labels[i], fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylim(bottom=min(0, cluster_means.min() * 1.1))
    
    ax.set_xticks(cluster_means.index)
    ax.set_xticklabels([f'{i}' for i in cluster_means.index])
    
    for bar in bars:
        height = bar.get_height()
        y_pos = max(height + 0.01, 0.02) if height >= 0 else height - 0.05
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                   xytext=(0, 3), 
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10)

fig.suptitle('Average Performance Metrics by Cluster', fontsize=16, y=1.05)

plt.tight_layout()
plt.show()

for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id} example teams:")
    print(df[df['cluster'] == cluster_id][['team', 'year', 'win_loss_perc', 'points_diff']].head(3))
