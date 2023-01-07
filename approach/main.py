import os
import json
import numpy as np
import random
import argparse
import pickle
import tools
from model import TestSorter
import tools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--doc2vec", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--testset", type=str, required=True)
    return parser.parse_args()


def check_exist(path):
    if not os.path.exists(path):
        print(f"{path} not found")
        exit()
    else:
        print(f"{path} found")


def check_job(changed_files, methods, fails):
    # sit1：no code change
    if not len(changed_files):
        print("no changed files")
        return False
    # sit2：code change exists but empty diff0
    elif not sum(map(len, changed_files)):
        print("with changed files, but all changed files are empty")
        return False
    # sit3：no method
    elif not len(methods):
        print("no test method")
        return False
    # sit4：no filaed method
    elif not len(fails):
        print("no failed method")
        return False
    else:
        return True
    

if __name__ == "__main__":
    this = os.getcwd()
    args = parse_args()
    print(f'arguments are: \n'
          f'{tools.format_args(args)}')
    check_exist(args.doc2vec)
    check_exist(args.net)

    print("loading model")
    sorter = TestSorter(args.doc2vec, args.net, args.epochs, 100)

    infos = []
    # cnt = 0
    for root, dirs, fs in os.walk(args.testset):
        for f in fs:
            if not f.endswith('json'):
                continue
            print("running on {}".format(f))
            with open(os.path.join(root, f), "r") as fp:
                job = json.load(fp)
            methods, code_changes, fails = tools.parse_job(job)

            if not check_job(code_changes, methods, fails):
                apfd = -1
                order = []
                sims = []
            else:
                queries, docs = sorter.preprocess(code_changes, methods)
                sims = sorter.get_sims(queries, docs)
                order = np.argsort(-np.array(sims)).tolist()
                apfd = tools.get_apfd(fails, order)
                # cnt += 1

            infos.append(
                {"file": f, "apfd": apfd, "order": order, "sims": sims}
            )

            array = np.array([d.get("apfd") for d in infos])
            print("apfd = {}, avg_apfd = {}\n".format(
                apfd, np.mean(array[array >= 0])
            ))
        # if cnt == 1:
        #     break

    fpath = os.path.join(this, args.tag + ".json")
    with open(fpath, "w") as fp:
        json.dump(infos, fp)

    print("done, info saved to {}".format(fpath))
