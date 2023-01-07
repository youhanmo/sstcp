from .libs import *

def cluster_by_kmeans(vecs: list, cluster_size:int) -> Tuple[int, list, np.ndarray]:

    X = np.vstack(vecs)

    if len(vecs) > cluster_size:
        cluster_res = KMeans(n_clusters=cluster_size, random_state=0).fit_predict(X)
    else:
        cluster_res = KMeans(n_clusters=1, random_state=0).fit_predict(X)

    cluster_identifiers = np.unique(cluster_res)

    return cluster_res, cluster_identifiers, cluster_size