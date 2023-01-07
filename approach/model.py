from torch import nn
import torch.nn.functional as F
import torch
from typing import List
import numpy as np
import tools


class Model(nn.Module):

    def __init__(self, input_dims: int) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fc = nn.Linear(input_dims * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestSorter:

    def __init__(self, doc2vec, net, epochs, input_dims):
        self.doc2vec = tools.read_doc2vec(doc2vec)
        self.net = Model(input_dims=input_dims)
        self.net.load_state_dict(torch.load(net, map_location='cpu')['model'])
        self.net.eval()
        self.epochs = epochs

    def preprocess(self, changed_files, test_methods):
        queries = []
        for changed_file in changed_files:
            if not len(changed_file):
                continue
            queries.extend(list(map(tools.tokenize, changed_file)))

        docs = list(map(tools.tokenize, test_methods))
        return queries, docs

    def doc2vec_inference(self, docs, add_axis=False):
        vectors = []
        for doc in docs:
            vector = self.doc2vec.infer_vector(doc, epochs=self.epochs)
            if add_axis:
                vector = vector[None, :]
            vectors.append(vector)
        return vectors

    def get_sims(self, queries, docs):
        query_vectors = self.doc2vec_inference(queries, add_axis=False)
        doc_vectors = self.doc2vec_inference(docs, add_axis=False)
        pairs = []
        for doc in doc_vectors:
            max_sim = -1
            max_q = None
            for q in query_vectors:
                sim = tools.cos_sim(doc, q)
                if sim > max_sim:
                    max_sim = sim
                    max_q = q
            pair = np.concatenate((doc[None, ...], max_q[None, ...]), axis=1)
            pairs.append(pair)
        x = np.concatenate(pairs, axis=0)
        x = torch.tensor(x).float()
        output = self.net(x)
        probs = F.softmax(output, dim=1)
        return probs.detach().numpy()[:, 1].tolist()
