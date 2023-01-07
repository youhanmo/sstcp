import os
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

with open(args.path, "r") as fp:
    result = json.load(fp)

for i, job_result in enumerate(result):
    for j, sim in enumerate(job_result["sims"]):
        print(sim)
        if j >= 10: break
    print()
    if i >= 10:
        break
