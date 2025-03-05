import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("tweets.csv")

users = df['author']
tweets = df['content']
likes = df['number_of_likes']
shares = df['number_of_shares']

G = nx.DiGraph()

for user in users.unique():
    G.add_node(user)

for i, row in df.iterrows():
    author = row['author']
    content = row['content']
    for word in content.split():
        if word.startswith('@'):
            mentioned_user = word.strip('@.,!?:;')
            if mentioned_user in users.values:
                G.add_edge(author, mentioned_user)

pagerank = nx.pagerank(G)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

important_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:3]

def plot_graph(G):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=8, edge_color='gray')
    plt.title("Twitter Network Graph")
    plt.show()

plot_graph(G)

print("Top 3 important users:", important_nodes)
