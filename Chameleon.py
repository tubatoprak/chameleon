import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import metis
import itertools

def draw(df):
    df.plot(kind='scatter', c=df['cluster'], cmap='gist_rainbow', x=0, y=1)
    plt.title("Chameleon Cluster")
    plt.show(block=False)

def knn_cluster(df, k):
    points = [p[1:] for p in df.itertuples()]
    g = nx.Graph()
    for i in range(0, len(points)):
        g.add_node(i)
    iterpoints = enumerate(points)
    for i, p in iterpoints: 
        distances = list(map(lambda x: np.linalg.norm(np.array(p) - np.array(x)), points))
        closests = np.argsort(distances)[1:k+1]  
        g.nodes[i]['pos'] = p
    g.graph['edge_weight_attr'] = 'similarity'
    return g


def part_graph(graph, k, df=None):
    edgecuts, parts = metis.part_graph(
        graph, 2, objtype='cut', ufactor=250)
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]['cluster'] = parts[i]
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph


def pre_part_graph(graph, k, df=None):
    clusters = 0
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]['cluster'] = 0
    cnts = {}
    cnts[0] = len(graph.nodes())

    while clusters < k - 1:
        maxc = -1
        maxcnt = 0
        for key, val in cnts.items():
            if val > maxcnt:
                maxcnt = val
                maxc = key
        s_nodes = [n for n in graph.nodes if graph.nodes[n]['cluster'] == maxc]
        s_graph = graph.subgraph(s_nodes)
        edgecuts, parts = metis.part_graph(
            s_graph, 2, objtype='cut', ufactor=250)
        new_part_cnt = 0
        for i, p in enumerate(s_graph.nodes()):
            if parts[i] == 1:
                graph.nodes[p]['cluster'] = clusters + 1
                new_part_cnt = new_part_cnt + 1
        cnts[maxc] = cnts[maxc] - new_part_cnt
        cnts[clusters + 1] = new_part_cnt
        clusters = clusters + 1

    edgecuts, parts = metis.part_graph(graph, k)
    if df is not None:
        df['cluster'] = nx.get_node_attributes(graph, 'cluster').values()
    return graph


def get_cluster(graph, clusters):
    nodes = [n for n in graph.nodes if graph.nodes[n]['cluster'] in clusters]
    return nodes


def connecting_edges(partitions, graph):
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set

def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]['weight'] for edge in edges]


def bisection_weights(graph, cluster):
    cluster = graph.subgraph(cluster)
    cluster = cluster.copy()
    cluster = part_graph(cluster, 2)
    partitions = get_cluster(cluster, [0]), get_cluster(cluster, [1])
    edges = connecting_edges(partitions, cluster)
    weights = get_weights(cluster, edges)
    return weights

def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))

    cluster_i = graph.subgraph(cluster_i)
    edges = cluster_i.edges()
    weights = get_weights(cluster_i, edges)
    Ci= np.sum(weights)
    
    cluster_j = graph.subgraph(cluster_j)
    edges = cluster_j.edges()
    weights = get_weights(cluster_j, edges)
    Cj= np.sum(weights)
 
    SECci =  np.mean(bisection_weights(graph, cluster_i))
    SECcj =  np.mean(bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge(graph, df, a, k):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False
    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            graph_i = get_cluster(graph, [i])
            graph_j = get_cluster(graph, [j])
            edges = connecting_edges(
                (graph_i, graph_j), graph)
            if not edges:
                continue
            edges = connecting_edges(( graph_i, graph_j), graph)
            result = np.sum(get_weights(graph, edges)) / ((np.sum(bisection_weights(graph,graph_i)) + np.sum(bisection_weights(graph,graph_j))) / 2.0)
            ms= result * np.power(relative_closeness(graph,graph_i, graph_j), a)
            if ms > max_score:
                max_score = ms
                ci, cj = i, j
    if max_score > 0:
        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]['cluster'] == cj:
                graph.nodes[p]['cluster'] = ci
    return max_score > 0


def cluster(df, k, knn=10, m=30, alpha=2.0, verbose=False):
    graph = knn_cluster(df, knn)
    graph = pre_part_graph(graph, m, df)
    iterm = enumerate(range(m - k))
    for i in iterm:
        merge(graph, df, alpha, k)
        ans = df.copy()    
    clusters = list(pd.DataFrame(df['cluster'].value_counts()).index)
    c = 1
    for i in clusters:
        ans.loc[df['cluster'] == i, 'cluster'] = c
        c = c + 1  
    return ans

if __name__ == "__main__":
    print("---Running Chameleon Cluster----k = 10")
    df = pd.read_csv('Aggregation.csv', sep=' ')
    result = cluster(df,7, knn=10, m=40, alpha=2.0)
    draw(result)

