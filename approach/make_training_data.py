import os
import random
import numpy as np
import json
from tqdm import tqdm
import pickle as pkl
from sklearn.model_selection import train_test_split
import tools
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--index", type=int, required=True)
file_index = parser.parse_args().index

seed = file_index
random.seed(seed)


files = []
for root, dirs, fs in os.walk('data/test_case_all_vecs'):
    for f in fs:
        if not f.endswith('pkl'):
            continue
        files.append(os.path.join(root, f))


positive_pairs = []
negtive_pairs = []
for fpath in tqdm(files):
    with open(fpath, 'rb') as fp:
        dicts = pkl.load(fp)
    change_vectors = []
    test_vectors = []
    for d in dicts:
        if 'diff0' in d:
            change_vectors.extend(d['diff0'])
        else:
            if not len(d['content']):
                continue
            test_vectors.append((d['content'], d['fail']))
    if not len(change_vectors):
        print(f"error: no code change found in {fpath}")
        continue
    if not len(test_vectors):
        print(f"error: no test case found in {fpath}")
        continue

    for methods, label in test_vectors:
        max_sim = -1
        max_pair = None
        for test_func in methods:
            for cc_func in change_vectors:
                sim = tools.cos_sim(test_func, cc_func)
                if sim > max_sim:
                    max_sim = sim
                    max_pair = np.concatenate((test_func, cc_func), axis=0)
        if label == 0:
            negtive_pairs.append(max_pair)
        else:
            positive_pairs.append(max_pair)
    print(f"after {fpath}, n pos {len(positive_pairs)}, n neg {len(negtive_pairs)}")
    # if (len(positive_pairs) > 3): break

# sample
n_pos = len(positive_pairs)
negtive_pairs = random.sample(negtive_pairs, 3 * n_pos)

pairs = list(map(lambda x: x[None, :], positive_pairs + negtive_pairs))
labels = [1] * len(positive_pairs) + [0] * len(negtive_pairs)

x = np.concatenate(pairs, axis=0)
y = np.asarray(labels)
x_train, x_test, y_train, y_test = train_test_split(
                                   x, y, stratify=y, test_size=0.3, random_state=seed)

n_positive = len(positive_pairs)
n_negative = len(negtive_pairs)

print(f"size of dataset = {len(x)}")
print(f"number of positive samples = {n_positive}")
print(f"number of negative samples = {n_negative}")
print(f"train size = {len(x_train)}")
print(f"test size = {len(x_test)}")

prefix = os.path.join(os.getcwd(), "data", "train", str(file_index))
if not os.path.exists(prefix): os.makedirs(prefix)

np.save(os.path.join(prefix, "x_train"), x_train)
np.save(os.path.join(prefix, "y_train"), y_train)
np.save(os.path.join(prefix, "x_test"), x_test)
np.save(os.path.join(prefix, "y_test"), y_test)

