import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    file_path = os.getcwd() + "/datas-mean.csv"
    save_path = file_path.replace("datas-mean.csv", "datas-reduce.csv")
    all_datas = pd.read_csv(file_path)

    all_length = len(all_datas)

    reduce_datas = pd.DataFrame()
    print("all_length = ", all_length)
    for i in range(0, all_length - 100, 100):
        choice_indexes = np.random.choice(100, 10, False)
        choice_indexes = choice_indexes + i
        choice_list = choice_indexes.tolist()
        print("index = %d, choice index 0 = %d" % (i, int(choice_list[0])))
        for j in range(len(choice_list)):
            reduce_datas = reduce_datas.append(all_datas.iloc[choice_list[j]])

    reduce_length = len(reduce_datas.index)
    print("Recude Length = ", reduce_length)
    reduce_datas.to_csv(save_path)


