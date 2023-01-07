import os
import json
import pickle
from gensim.models.doc2vec import Doc2Vec
from typing import List
import re


def tokenize(code: str) -> list:
    # Remove some special charactor.
    # Add @, &, $, `
    special_word = [',','/','+',')','.','(',';','{','}','<','>','"','"', '\'','=', '@', '&', '$', '`']
    for sub in special_word:
        code = code.replace(sub,' ')
    code = code.replace('\t',' ')
    code = code.replace('\n',' ')
    code = code.replace('_', ' ') # Add tag '_' for tokenizing
    code = ' '.join(code.strip().split())
    term_list = []
    term_list = code.split(" ")
    lower_term = []
    for word in term_list:
        if len(word.strip()) != 0:
            new_word = word.strip()
            if len(new_word) > 1:
                for sub_word in re.sub(r"([A-Z])", r" \1",new_word).split():
                    lower_term.append(sub_word.lower())
            else:
                lower_term.append(new_word)
    return lower_term

# load model
print('loading doc2vec model ...')
d2v_path = 'doc2vec_model/window_8.model'
doc2vec = Doc2Vec.load(d2v_path)
epochs = 200
print('done ...')

def inference_docs(model: 'doc2vec', docs: List[str]) -> List['vector']:
    vectors = []
    for doc in docs:
        doc = tokenize(doc)
        vectors.append(model.infer_vector(doc))
    return vectors

# 2980 jobs
cnt = 0
for root, dirs, fs in os.walk('test_case_all'):
    for f in fs:
        if f.endswith('json'):
            print(f'\ninferencing vectors for {f}, total = {cnt}')
            fpath = os.path.join(root, f)
            # read json
            with open(fpath, 'r') as fp:
                dicts = json.load(fp)
            for d in dicts:
                if 'diff0' in d:
                    docs = d.get('diff0')
                    vectors = inference_docs(doc2vec, docs)
                    # replace documents by vectors
                    d['diff0'] = vectors
                else:
                    docs = d.get('content')
                    vectors = inference_docs(doc2vec, docs)
                    d['content'] = vectors

            # write json
            name = f.split('.')[0]
            to_dir = os.path.join('test_case_all_vecs', name)
            if not os.path.exists(to_dir):
                os.makedirs(to_dir)
            to_path = os.path.join(to_dir, f'{name}.pkl')
            with open(to_path, 'wb') as fp:
                pickle.dump(dicts, fp)
            print(f'save vectors to {to_path} ...')
            cnt += 1
            # exit()