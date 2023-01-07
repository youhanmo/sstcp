import os

for i in range(10):
    cmd = f"python -u train.py --index {i} --backup ./backup/{i} > ./logs/train/train_{i}.log"
    print(f"run {cmd}")
    os.system(cmd)