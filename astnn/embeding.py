import pandas as pd
import os
import sys
import warnings
import json
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    train_ast = pd.read_pickle('./ast.pkl')
    pairs = pd.read_pickle('./train_.pkl')
    train_ids = pairs['id1'].append(pairs['id2']).unique()
    trees = train_ast.set_index('id', drop=False).loc[train_ids]

    trees = trees.drop(['id'], axis=1)
    all_ast = pd.DataFrame(columns=['code'])
    all_ast = pd.concat([all_ast, trees])
    all_ast = all_ast.reset_index(drop=True)

    testAst = pd.DataFrame(columns=['code'])
    case_idx = 0
    # This jsonfile stores the test fails of each jobid
    with open('./fails_for_each_jobid.json', 'r') as fp:
        fail_dict_all = json.load(fp)
    for jobId in os.listdir('./all_test_pkl'):
        print(jobId, '  length of all_ast: ', len(all_ast), '  length of testAst: ', len(testAst))
        for file in os.listdir(os.path.join('./all_test_pkl', str(jobId))):
            if file == 'testcases.pkl':
                continue
            file_ = os.path.join('./all_test_pkl', str(jobId), file)
            asts = pd.read_pickle(file_)
            fail_num_3 = len(fail_dict_all[jobId]) * 3
            for this_file_idx in range(len(asts)):
                if this_file_idx < fail_num_3:
                    testAst.loc[case_idx, 'code'] = asts.loc[this_file_idx, 'code']
                    case_idx += 1
                else:
                    break

    all_ast = pd.concat([all_ast, testAst])
    all_ast = all_ast.reset_index(drop=True)
    all_ast.to_pickle('./all_blocks.pkl')

    from utils import get_sequence as func


    def trans_to_sequences(ast):
        sequence = []
        func(ast, sequence)
        return sequence


    corpus = all_ast['code'].apply(trans_to_sequences)

    from gensim.models.word2vec import Word2Vec
    w2v = Word2Vec(corpus, size=128, workers=16, sg=1, max_final_vocab=6000)
    MAX_TOKENS = w2v.wv.syn0.shape[0]
    print(MAX_TOKENS)
    w2v.save('./new_node_w2v_' + str(128))
