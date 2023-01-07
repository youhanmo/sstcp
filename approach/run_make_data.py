import os


for i in range(10):
    cmd = f"nohup python -u make_training_data.py --index {i} > {i}.log 2>&1 &"
    print(f"run {cmd}")
    os.system(cmd)