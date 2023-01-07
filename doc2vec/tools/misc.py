from .libs import *


""" Read doc2vec model to memory. """
def read_doc2vec(path: str) -> Doc2Vec:
    return Doc2Vec.load(path)

""" Calculate cosine similarity between two vectors. """
def cos_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    return np.dot(vec_a, vec_b.T) / (a_norm * b_norm)