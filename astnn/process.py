import os
import json


curDir = os.getcwd()
jobDir = os.path.join(curDir, "test_case_all")


def parseJsonFile(jsonFile: list) -> tuple:
    '''
    fetch query, testcases, failed test cases id from json,
    :param jsonFile:need to be parsed
    :return: query, testcases, fail test cases fetched
    '''
    # Indices of failed test case.
    fails = []

    testcases = [_dict for _dict in jsonFile if not ("diff0" in _dict.keys())]
    codechanges = [_dict for _dict in jsonFile if "diff0" in _dict.keys()]

    for index, testcase in enumerate(testcases):
        # Get content --> list.
        # content = testcase.get("content")
        # Check if this test case failed.
        if testcase.get("fail"):
            fails.append(index)
    contents = [testcase.get("content") for testcase in testcases]
    queries = [codechange.get("diff0") for codechange in codechanges]
    return queries, contents, fails


def main():
    # Store the length of each testcase
    sum_ = {}
    # Store all fails index of each jobid
    fails_ = {}
    # Loop for each json file(build)
    for jobId in os.listdir(jobDir):
        sum_[jobId] = {}
        # store all queries and testacses
        all_queries = []
        all_contents = []
        if not os.path.exists('./all_test/' + "{}".format(str(jobId))):
            os.mkdir('./all_test/' + "{}".format(str(jobId)))
        # Read json.
        with open(os.path.join(jobDir, jobId, "{}.json".format(jobId)), "r") as fp:
            jsonFile = json.load(fp)
            queries, testcases, fails = parseJsonFile(jsonFile)
        fails_[jobId] = fails
        # Label the testcase
        idx = 0
        for query in queries:
            all_queries.extend(query)
        for testcase in testcases:
            all_contents.extend(testcase)
            sum_[jobId][idx] = len(testcase)
            idx += 1

        # write to tsv
        with open('./all_test/' + "{}".format(str(jobId)) + '/queries.tsv', 'w', encoding='utf-8') as fp1:
            for i in range(len(all_queries)):
                if i == (len(all_queries) - 1):
                    fp1.write(all_queries[i])
                else:
                    fp1.write(all_queries[i] + '\n')

        with open('./all_test/' + "{}".format(str(jobId)) + '/testcases.tsv', 'w', encoding='utf-8') as fp2:
            for i in range(len(all_contents)):
                if i == (len(all_contents) - 1):
                    fp2.write(all_contents[i])
                else:
                    fp2.write(all_contents[i] + '\n')

    json_str = json.dumps(sum_)
    with open('./num1.json', 'w') as json_file:
        json_file.write(json_str)

    json_str_ = json.dumps(fails_)
    with open('./fails_for_each_jobid.json', 'w') as json_file:
        json_file.write(json_str_)


if __name__ == '__main__':
    if not os.path.exists('./all_test'):
        os.mkdir('./all_test')
    main()