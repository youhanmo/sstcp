import numpy as np
from typing import Tuple

try:
    import hdbscan
except:
    print("library hdbscan not found, please use\n   'pip install hdbscan'\nto download it.")

# from sklearn.decomposition import PCA


def cluster_by_hdbscan(samples: np.ndarray, min_samples: int,
                       min_cluster_size: int) -> Tuple[np.ndarray, list, int]:

    """
    Parameters:
    -----------
    - samples : np.ndaray with shape (n, m)
        n is the number of samples, m is the dimension of each sample.
    
    - min_cluster_size : int
        minimum number of samples in each cluster.

    - gen_min_span_tree : bool
        decide whether to generate minimum spaning tree to link each cluster.

    Returns:
    --------
    - labels : np.ndarray with shape (n, )
        cluster identifier for each sample.
    
    - cluster_identifiers : list
        identifier of each cluster.
    
    - n_cluster : int
        number of clusters.
    """

    if len(samples) == 1:
        return [0], [0], 1

    # n_dims = 50 if 50 < len(samples) else len(samples)
    # samples = PCA(n_components=n_dims).fit_transform(samples)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True
    )
    clusterer = clusterer.fit(samples)
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

    if soft_clusters.ndim == 1:
        soft_clusters = soft_clusters.reshape(-1, 1)
    cluster_res = np.argmax(soft_clusters, axis=1)

    cluster_identifiers = np.unique(cluster_res)

    # print("n_sample = {}, n_cluster = {}".format(len(samples), len(cluster_identifiers)))

    return cluster_res, cluster_identifiers, len(cluster_identifiers)
