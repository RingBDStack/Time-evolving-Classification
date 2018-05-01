import os
import numpy as np


def split_file(dir, file_name):
    file_path = os.path.join(dir, file_name)
    res = np.loadtxt(file_path)
    modelname = file_name[13:]
    np.savetxt(os.path.join(dir, 'accuracy_' + modelname), res[:, 0])
    np.savetxt(os.path.join(dir, 'time_' + modelname), res[:, 1])

dir_names = os.listdir("result")
for dir_name in dir_names:
    dir = os.path.join("result", dir_name)
    file_names = os.listdir(dir)
    for item in file_names:
        if item.startswith('accr_results_'):
            split_file(dir, item)


