import os
import json
import re
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score