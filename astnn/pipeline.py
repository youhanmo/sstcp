import pandas as pd
import os
import warnings
import javalang
import json
import copy
warnings.filterwarnings('ignore')

curDir = os.getcwd()
jobDir = os.path.join(curDir, "all_test")

# load num1
with open('./num1.json', 'r') as f:
    num1 = json.load(f)
copy_num1 = copy.deepcopy(num1)


# parse source code
def parse_source(path, file_):
    parse_fails = []
    if os.path.exists(path):
        os.remove(path)

    def parse_program(func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    source = pd.read_csv(file_, sep='\t', header=None)
    source.columns = ['code']
    source_num = len(source)
    true_idx = 0
    new_source = pd.DataFrame(columns=['code'])

    for idx in range(source_num):
        try:
            new_source.loc[true_idx, 'code'] = parse_program(source.loc[idx, 'code'])
            true_idx += 1
        except:
            parse_fails.append(idx)

    new_source.to_pickle(path)
    return parse_fails


if __name__ == '__main__':
    if not os.path.exists('./all_test_pkl'):
        os.mkdir('./all_test_pkl')
    for jobId in os.listdir(jobDir):
        # print(jobId)
        if not os.path.exists('./all_test_pkl/' + str(jobId)):
            os.mkdir('./all_test_pkl/' + str(jobId))
        all_num = num1[jobId]
        for file in os.listdir('./all_test/' + str(jobId)):
            file_ = os.path.join('./all_test', str(jobId), file)
            path = os.path.join('./all_test_pkl', str(jobId), file[:-4] + '.pkl')
            try:
                parseFails = parse_source(path, file_)
            except BaseException as e:
                print(jobId, e)
                continue

            if file == 'testcases.tsv':
                all_num_list = []
                for key in all_num.keys():
                    # print(all_num.keys())
                    # exit(0)
                    if len(all_num_list) == 0:
                        all_num_list.append(all_num[key])
                    else:
                        all_num_list.append(all_num_list[-1] + all_num[key])

                for fail_idx in parseFails:
                    for i in range(len(all_num_list)):
                        if fail_idx < all_num_list[i]:
                            copy_num1[jobId][str(i)] = copy_num1[jobId][str(i)] - 1
                            break

    json_str = json.dumps(copy_num1)
    with open('./num2.json', 'w') as json_file:
        json_file.write(json_str)


