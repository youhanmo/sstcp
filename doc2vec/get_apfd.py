import os
import json
import numpy as np
import random
import argparse
import pickle
import tools


def inference(model, texts, epochs):
    """
    covert tokenize text into vectors

    Parameters:
    -----------
    - model: Doc2vec

    - texts: list
        >>> [['word', 'word', 'word', ...], ...]

    - epochs: int

    Returns:
    --------
    - vectors: list
    """
    return [
        model.infer_vector(text, epochs=epochs)
        for text in texts
    ]


def document_sims(model, methods, code_changes, epochs, concat):
    """
    calculate similarity using doc2vec
    """
    queries = []
    for diff0 in code_changes:
        if concat:
            if not len(diff0):
                continue
            query = tools.tokenize(" ".join(diff0))
            queries.append(query)
        else:
            queries.extend(list(map(tools.tokenize, diff0)))

    queries = inference(model, queries, epochs)

    sims = []
    for method in methods:
        if len(method) <= 0:
            continue
        # BUG(fixed): method = list(map(tools.tokenize, method))

        method = tools.tokenize(method)
        vector = model.infer_vector(method, epochs=epochs)
        sim = np.max([tools.cos_sim(vector, query) for query in queries])
        sims.append(sim)

    return sims


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--testset", type=str, required=True)
    parser.add_argument("--concat", type=int, required=True)

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    print("loading model...")
    doc2vec = tools.read_doc2vec(args.model)
    print("done")

    infos = []
    cnt = 1
    for root, dirs, fs in os.walk(args.testset):
        fs = list(filter(lambda f: f.endswith('json'), fs))
        # if cnt > 1:
        #     break
        for f in fs:
            print(f"running job {cnt}, file name: {f}")

            with open(os.path.join(root, f), "r") as fp:
                job = json.load(fp)
            methods, code_changes, fails = tools.parse_job(job)

            # concat all code change files
            tmp = []
            for ccfile in code_changes:
                tmp.extend(ccfile)
            code_changes = [tmp]

            # sit1: no code change
            if not len(code_changes):
                print("no diff0 ...")
                apfd = -1
                order = []
                sims = []

            # sit2: code change exists but no diff0
            elif not sum(map(len, code_changes)):
                print("with diff0, but no code change ...")
                apfd = -2
                order = []
                sims = []

            # sit3: no fault-triggering methods
            elif not len(fails):
                print("no failed method ...")
                apfd = -3
                order = []
                sims = []

            # sit4: no method
            elif not len(methods):
                print("no method ...")
                apfd = -4
                order = []
                sims = []

            # normal
            else:
                sims = document_sims(
                    model=doc2vec,
                    methods=methods,
                    code_changes=code_changes,
                    epochs=args.epochs,
                    concat=args.concat
                )
                order = np.argsort(-np.array(sims)).tolist()
                apfd = tools.get_apfd(fails, order)

            # record
            infos.append(
                {"file": os.path.join(root, f), "apfd": apfd, "order": order, "sims": list(map(str, sims)),
                "fails": list(map(int, fails))}
            )

            # cal apfd
            array = np.array([d.get("apfd") for d in infos])
            print(f"apfd = {apfd}, average_apfd = {np.mean(array[array >= 0])}\n")

            cnt += 1

    fpath = os.path.join(os.getcwd(), args.tag + ".json")
    with open(fpath, "w") as fp:
        json.dump(infos, fp)

    print("done, info saved to {}".format(fpath))
