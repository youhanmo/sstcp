import os
import json
import numpy as np
import pandas as pd
import random
import argparse
import pickle

import tools


def fmt_output(sth: str) -> None:
    print("{}{}{}".format("- " * 15, sth, " -" * 15))


def inference_vectors(model, texts, epochs):
    return [
        model.infer_vector(tools.tokenize(text), epochs=epochs)
        for text in texts
    ]


def get_sims(model, test_cases, code_changes, epochs):
    sims = []
    code_change_vecs = inference_vectors(model, code_changes, epochs)

    # Loop for each testcase: List[str](a list of method string)
    for test_case in test_cases:
        if len(test_case) == 0:
            # This tese case's content is empty.
            sims.append([])
        else:
            test_case_vecs = inference_vectors(model, test_case, epochs)
            test_case_sims = []
            for func_vec in test_case_vecs:
                func_sims = []
                for code_change_vec in code_change_vecs:
                    func_sims.append(float(tools.cos_sim(func_vec, code_change_vec)))
                test_case_sims.append(func_sims)
            sims.append(test_case_sims)

    return sims


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--job_dir", type=str, required=True)

    return parser.parse_args()


def load_doc2vec(path):
    print("loading model...")
    assert os.path.exists(path), "{} not found.".format(path)
    print("found {}.".format(path))
    doc2vec = tools.read_doc2vec(path)
    print("done")

    return doc2vec

if __name__ == "__main__":
    cur_dir = os.getcwd()

    args = parse_args()
    epochs, model = args.epochs, args.model
    fname = args.fname
    job_dir = args.job_dir

    res = {}

    print(
        "Running configuration:\n"\
        "    model_path:       {}\n"\
        "    inference epochs: {}\n"\
        "    fname:            {}\n".format(
            model, epochs, fname
        )
    )

    doc2vec = load_doc2vec(model)
    for index, job_id in enumerate(os.listdir(job_dir)):
        print("progress = {}, ".format(index + 1), end="")

        # Read this job.
        with open(os.path.join(job_dir, job_id, "{}.json".format(job_id)), "r") as fp:
            job = json.load(fp)
        """
        test_cases: List[List[string of method]]
        code_changes: List[string of method]
        """
        test_cases, code_changes, fails = tools.parse_job(job)

        # Such as job id: 436719525, diff0 = []
        if len(code_changes) == 0:
            sims = []
        else:
            sims = get_sims(
                model=doc2vec,
                test_cases=test_cases,
                code_changes=code_changes,
                epochs=epochs
            )
        fails = list(map(int, fails))
        res[job_id] = {"sims": sims, "fails": fails}
        
    fpath = os.path.join(cur_dir, fname + ".json")
    with open(fpath, "w") as fp:
        json.dump(res, fp)
    print("Done!")
