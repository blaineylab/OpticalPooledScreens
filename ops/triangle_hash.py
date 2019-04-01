import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor


def process(df):
    v, c = get_vectors(df[['i', 'j']].values)

    return (pd.concat([
        pd.DataFrame(v).rename(columns='V_{0}'.format), 
        pd.DataFrame(c).rename(columns='c_{0}'.format)], axis=1)
        .assign(magnitude=lambda x: x.eval('(V_0**2 + V_1**2)**0.5'))
    )

    return df_

def nine_edge_hash(dt, i):
    """For triangle `i` in Delaunay triangulation `dt`, extract the vector 
    displacements of the 9 edges containing to at least one vertex in the 
    triangle.

    Does not currently remove "upper" triangles, i.e., border triangles that 
    jump to points within the triangulation.

    Example:
    dt = Delaunay(X_0)
    i = 0
    segments, vector = nine_edge_hash(dt, i)
    plot_nine_edges(X_0, segments)

    """
    # indices of inner three vertices
    # already in CCW order
    a,b,c = dt.simplices[i]

    # reorder so ab is the longest
    X = dt.points
    start = np.argmax((np.diff(X[[a, b, c, a]], axis=0)**2).sum(axis=1)**0.5)
    if start == 0:
        order = [0, 1, 2]
    elif start == 1:
        order = [1, 2, 0]
    elif start == 2:
        order = [2, 0, 1]
    a,b,c = np.array([a,b,c])[order]

    # outer six vertices
    a_ix, b_ix, c_ix = dt.neighbors[i]
    inner = {a,b,c}
    outer = lambda xs: [x for x in xs if x not in inner][0]
    # should be two shared, one new; if not, probably a weird edge simplex
    # that shouldn't hash (return None)
    bc = outer(dt.simplices[dt.neighbors[i, order[0]]])
    ac = outer(dt.simplices[dt.neighbors[i, order[1]]])
    ab = outer(dt.simplices[dt.neighbors[i, order[2]]])
    
    # segments
    segments = [
     (a, b),
     (b, c),
     (c, a),
     (a, ab),
     (b, ab),
     (b, bc),
     (c, bc),
     (c, ac),
     (a, ac),
    ]

    i = X[segments, 0]
    j = X[segments, 1]
    vector = np.hstack([np.diff(i, axis=1), np.diff(j, axis=1)])
    return segments, vector

def plot_nine_edges(X, segments):
    fig, ax = plt.subplots()
    
    [(a, b),
     (b, c),
     (c, a),
     (a, ab),
     (b, ab),
     (b, bc),
     (c, bc),
     (c, ac),
     (a, ac)] = segments
    
    for i0, i1 in segments:
        ax.plot(X[[i0, i1], 0], X[[i0, i1], 1])

    d = {'a': a, 'b': b, 'c': c, 'ab': ab, 'bc': bc, 'ac': ac}
    for k,v in d.items():
        i,j = X[v]
        ax.text(i,j,k)

    ax.scatter(X[:, 0], X[:, 1])

    s = X[np.array(segments).flatten()]
    lim0 = s.min(axis=0) - 100
    lim1 = s.max(axis=0) + 100

    ax.set_xlim([lim0[0], lim1[0]])
    ax.set_ylim([lim0[1], lim1[1]])
    return ax

def get_vectors(X):
    """Get the nine edge vectors and centers for all the faces in the 
    Delaunay triangulation of point array `X`.
    """
    dt = Delaunay(X)
    # skip triangles with an edge on the outer boundary
    keep = (dt.neighbors > -1).all(axis=1)
    results = [nine_edge_hash(dt, i) for i in range(dt.simplices.shape[0])]
    vectors = np.array([v for _, v in results])
    centers = np.vstack([X[dt.simplices, 0].mean(axis=1), 
                     X[dt.simplices, 1].mean(axis=1)]).T
    return vectors.reshape(-1, 18)[keep], centers[keep]

def nearest_neighbors(V_0, V_1):
    Y = cdist(V_0, V_1)
    distances = Y.min(axis=1)
    ix_0 = np.arange(V_0.shape[0])
    ix_1 = Y.argmin(axis=1)
    return ix_0, ix_1, distances

def get_vc(df, normalize=True):
    V,c = (df.filter(like='V').values, 
            df.filter(like='c').values)
    if normalize:
        V = V / df['magnitude'].values[:, None]
    return V, c

def compare_positions(df_0, df_1):
    V_0, c_0 = get_vc(df_0)
    V_1, c_1 = get_vc(df_1)

    i0, i1, distances = nearest_neighbors(V_0, V_1)

    model = RANSACRegressor(random_state=0, min_samples=10,
                           residual_threshold=30)
    model.fit(c_0[i0], c_1[i1])
    
    translation = model.estimator_.coef_
    inliers = model.inlier_mask_
    
    # analyze inliers
    inlier_count = model.inlier_mask_.sum()
    score = model.score(c_0[i0], c_1[i1])

    guess = model.predict(c_0[i0]) 
    error = np.linalg.norm(guess - c_1[i1], axis=1)
    hits = error < 2
    
    return translation, inliers, hits

def query_points(df_query, df_target, n):
    df_0 = df_query.sample(n)
    df_1 = df_target
    translation, inliers, hits = compare_positions(df_0, df_1)
    return {'translation': translation, 'inliers': inliers.sum(), 
            'hits': hits.sum()}
