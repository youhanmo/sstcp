import os

for i in range(10):
    cmd = f"nohup python -u main.py --doc2vec ./doc2vec_model/window_8.model "\
          f"--net ./backup/{i}/ckpt_300.pkl --testset ./testset/ "\
          f"--tag model_data_{i}_cc_new_version > ./logs/test/{i}_cc_new_version.log 2>&1 &"
    print(f"run {cmd}")
    os.system(cmd)