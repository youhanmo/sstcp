import os
import stat
import shutil

#filePath
def delete_file(filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0],name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0],name))
            shutil.rmtree(filePath)
        return "delete ok"
    else:
        return "no filepath"

fail_jobid = []
with open('./queries_null.txt', 'r') as fp:
    str_ = fp.readline()
    while str_:
        fail_jobid.append(str_.strip('\n'))
        str_ = fp.readline()

print(fail_jobid)

for jobid in os.listdir(os.path.join('./all_test_pkl')):
    if jobid in fail_jobid:
        print(jobid)
        delete_file(os.path.join('./all_test_pkl', jobid))