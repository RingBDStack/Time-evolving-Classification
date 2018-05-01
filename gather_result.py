
import os
import numpy as np

SOURCE_PATH = '.'
TARGET_PATH = '.'




def identify_best_eval(eval_path):
    """
    返回eval_path中acc最高的eval值及ckpt数
    :param eval_path: log_eval文件所在目录
    :return: (acc, num)
    """
    acc = []
    nums = []
    with open(os.path.join(eval_path, "log_eval")) as f:
        while True:
            result_line = f.readline()
            if not result_line:
                break
            if result_line[0] == ' ':
                continue
            acc.append(float(result_line.split(" ")[3][4:].replace(",", "")))
            nums.append(float(result_line.split(" ")[2][5:].replace(":", "")))

    def intv_avg(arr, i):
        l = max(i, i-2)
        r = min(i+10, len(arr) - 1)
        return sum(arr[l:r+1])/(r-l+1)

    acc = [intv_avg(acc, i) for i in range(len(acc))]

    i, acc = max(enumerate(acc), key=lambda item: item[1])
    return acc, nums[i]

def gather_accuracy_results(source_path = SOURCE_PATH, target_path = TARGET_PATH ):
    """
    收集各个时期的测试文件夹下面的best eval result
    :param source_path: 测试文件夹所在目录
    :param target_path: result要写入的目标目录
    :return: 将result写入磁盘：'accr_results' 文件
    """
    file_names = os.listdir(source_path)
    eval_names = []
    batch_cnt = []
    for item in file_names:
        if item.startswith('eval_'):
            eval_names.append(item)
    # total_time_steps = len(eval_names)
    total_time_steps = 12
    results = np.zeros([total_time_steps, 2], dtype=np.float32)
    sum = 0.; cnt = 0
    for eval_name in eval_names:
        eval_path = os.path.join(source_path,eval_name)
        time_step = int(eval_name.split('_')[-1])
        results[time_step][0], results[time_step][1] = identify_best_eval(eval_path)
        if results[time_step][0] > 0.01:
            cnt += 1
            sum += results[time_step][0]
    if cnt == 0:
        return
    print("%s: %.4f " % (source_path.split(os.path.sep)[-1], sum/cnt))
    output_name = 'accr_results_' + source_path.split(os.path.sep)[-1]
    np.savetxt(os.path.join(target_path, output_name),results)


if __name__ == '__main__':
    target_dir = os.path.join(TARGET_PATH, 'result')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for exp_line in open('./exp_paths_list.txt','r'):
        exp = exp_line.strip()
        target_path = os.path.join(target_dir, exp)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        scan_names = []
        file_names = os.listdir("logs/"+exp)
        for item in file_names:
            log_path = os.path.join("logs/"+exp, item)
            if os.path.exists(log_path):
                gather_accuracy_results(source_path=log_path, target_path=target_path)



