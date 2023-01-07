import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
import torch
import os
import json

warnings.filterwarnings('ignore')


def get_apfd(fails, ranks):
    """
    calculate apfd， same as doc2vec

    Parameters:
    -----------
    - fails: list
        failed method id
    - ranks: list
        method id list

    Returns:
    --------
    - apfd: float
        apfd
    """
    # key point:
    # index can be used to represent a testcase.
    tfs = [ranks.index(fail) + 1 for fail in fails]

    nb_fail, nb_method = len(fails), len(ranks)
    if nb_fail == 0:
        print("no method fails!")
        return -1
    elif nb_method == 0:
        print("no method at all!")
        return -2
    else:
        return 1 - (sum(tfs) / (nb_fail * nb_method)) \
               + 1 / (2 * nb_method)


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1 = []
    for _, item in tmp.iterrows():
        x1.append(item['code'])
    return x1


curDir = os.getcwd()

if __name__ == '__main__':
    word2vec = Word2Vec.load("./node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    #导入sum_字典
    with open('./num2.json', "r") as fp:
        sum_ = json.load(fp)

    with open('./fails_for_each_jobid.json', 'r') as fp:
        fail_dict_all = json.load(fp)
    all_apfd = []
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    EPOCHS = 5
    BATCH_SIZE = 1
    USE_GPU = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    failed_jobids = 0
    for jobId in os.listdir('./all_test_pkl'):
        # print(jobId)
        fail_flag = 0
        for case in sum_[jobId].keys():
            if sum_[jobId][case] == 0:
                if int(case) in fail_dict_all[jobId]:
                    failed_jobids += 1
                    print(jobId)
                    print(failed_jobids)
                    fail_flag = 1
                    break
        if fail_flag:
            continue

        df1 = pd.read_pickle(os.path.join(curDir, "all_test_pkl", str(jobId), "queriesBlocks.pkl"))
        df2 = pd.read_pickle(os.path.join(curDir, "all_test_pkl", str(jobId), "testcasesBlocks.pkl"))
        train_data1 = df1
        train_data2 = df2
        i = 0
        q_this_dis = []
        while i < len(train_data1):
            batch = get_batch(train_data1, i, BATCH_SIZE)
            i += BATCH_SIZE
            train1_inputs = batch
            if USE_GPU:
                train1_inputs = train1_inputs
            idx_ = len(train1_inputs)
            model.batch_size = idx_
            model.hidden = model.init_hidden()
            output1 = model(train1_inputs)
            if USE_GPU:
                output1 = output1.cpu().detach().numpy()
            else:
                output1 = output1.detach().numpy()
            for ij in range(len(output1)):
                q_this_dis.append(output1[ij])

        j = 0
        t_this_dis = []
        while j < len(train_data2):
            batch = get_batch(train_data2, j, BATCH_SIZE)
            j += BATCH_SIZE
            train2_inputs = batch
            if USE_GPU:
                train2_inputs = train2_inputs
            idx_ = len(train2_inputs)
            model.batch_size = idx_
            model.hidden = model.init_hidden()
            output2 = model(train2_inputs)
            if USE_GPU:
                output2 = output2.cpu().detach().numpy()
            else:
                output2 = output2.detach().numpy()
            for ij in range(len(output2)):
                t_this_dis.append(output2[ij])

        distance = []
        job_num = []
        for key in sum_[jobId].keys():
            case_dis = []
            num = sum_[jobId][key]
            if num == 0:
                continue
            case_encode = t_this_dis[sum(job_num): sum(job_num) + num]
            for tes in case_encode:
                nn = []
                for que in q_this_dis:
                    nn.append(cosine(tes, que))
                case_dis.append(np.min(nn))
            m = np.array(case_dis)
            distance.append(np.min(m))
            job_num.append(num)

        reorder = np.argsort(np.array(distance)).tolist()
        this_apfd = get_apfd(fail_dict_all[jobId], reorder)
        print(jobId, ': ', this_apfd)
        all_apfd.append(this_apfd)
    mean_apfd = np.mean(np.array(all_apfd))
    medium_apfd = np.median(np.array(all_apfd))
    print('Mean apfd of all jobid: ', mean_apfd)
    print('Middium apfd of all jobid: ', medium_apfd)





