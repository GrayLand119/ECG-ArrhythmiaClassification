import pandas as pd
import numpy as np
import os


def separate_type(type=None):
    if type is None:
        return

    cur_dir = os.getcwd() + "/"
    from_path = cur_dir + type + "/datas-mean.csv"
    to_training_path = cur_dir + type + "/datas-training.csv"
    to_testing_path = cur_dir + type + "/datas-testing.csv"
    to_validation_path = cur_dir + type + "/datas-validation_path.csv"

    from_df = pd.read_csv(from_path)
    from_length = len(from_df.index)

    all_indices = np.arange(from_length)  # 随机打乱索引并切分训练集与测试集
    np.random.shuffle(all_indices)

    training_df = pd.DataFrame()
    testing_df = pd.DataFrame()
    validation_df = pd.DataFrame()

    training_length = int(from_length * 0.8)
    testing_length = int(from_length * 0.1)
    validation_length = int(from_length * 0.1) + 1

    for i in range(from_length):
        t_s = from_df.iloc[all_indices[i]]
        print("Sep ", type, " Index ", i)
        if i < training_length:
            training_df = training_df.append(t_s, ignore_index=True, sort=False)
        elif i < training_length + testing_length:
            testing_df = testing_df.append(t_s, ignore_index=True, sort=False)
        else:
            validation_df = validation_df.append(t_s, ignore_index=True, sort=False)

    training_df.to_csv(to_training_path)
    testing_df.to_csv(to_testing_path)
    validation_df.to_csv(to_validation_path)


if __name__ == "__main__":
    separate_type('N')
    separate_type('S')
    separate_type('V')
    separate_type('F')
    separate_type('Q')
    pass