import networkx as nx
import pandas as pd
import numpy as np
import scipy.spatial.kdtree
from collections import Counter


def initialize_graph(df):
    arr_df = [x for _, x in df.groupby('frame')]
    nodes = df[['frame', 'label']].values
    nodes = [tuple(x) for x in nodes]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edges = []
    for df1, df2 in zip(arr_df, arr_df[1:]):
        edges = get_edges(df1, df2)
        G.add_weighted_edges_from(edges)
    
    return G


def get_edges(df1, df2):
    neighboring_points = 3
    get_label = lambda x: tuple(int(y) for y in x[[2, 3]])

    x1 = df1[['i', 'j', 'frame', 'label']].values
    x2 = df2[['i', 'j', 'frame', 'label']].values
    
    kdt = scipy.spatial.kdtree.KDTree(df1[['i', 'j']])
    points = df2[['i', 'j']]

    result = kdt.query(points, neighboring_points)
    edges = []
    for i2, (ds, ns) in enumerate(zip(*result)):
        end_node = get_label(x2[i2])
        for d, i1 in zip(ds, ns):
            start_node = get_label(x1[i1])
            w = d
            edges.append((start_node, end_node, w))

    return edges


def displacement(x):
    d = np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2)
    return d


def analyze_graph(G, cutoff=100):
    """Trace a path forward from each nucleus in the starting frame. Only keep 
    the paths that reach the final frame.
    """
    start_nodes = [n for n in G.nodes if n[0] == 0]
    max_frame = max([frame for frame, _ in G.nodes])
    
    cost, path = nx.multi_source_dijkstra(G, start_nodes, cutoff=cutoff)
    cost = {k:v for k,v in cost.items() if k[0] == max_frame}
    path = {k:v for k,v in path.items() if k[0] == max_frame}
    return cost, path


def filter_paths(cost, path, threshold=35):
    """Remove intersecting paths. 
    returns list of one [(frame, label)] per trajectory
    """
    # remove intersecting paths (node in more than one path)
    node_count = Counter(sum(path.values(), []))
    bad = set(k for k,v in node_count.items() if v > 1)
    print('bad', len(bad), len(node_count))

    # remove paths with cost over threshold
    too_costly = [k for k,v in cost.items() if v > threshold]
    bad = bad | set(too_costly)
    
    relabel = [v for v in path.values() if not (set(v) & bad)]
    assert(len(relabel) > 0)
    return relabel


def relabel_nuclei(nuclei, relabel):
    nuclei_ = nuclei.copy()
    max_label = nuclei.max() + 1
    for i, nodes in enumerate(zip(*relabel)):
        labels = [n[1] for n in nodes]
        table = np.zeros(max_label).astype(int)
        table[labels] = range(len(labels))
        nuclei_[i] = table[nuclei_[i]]

    return nuclei_

