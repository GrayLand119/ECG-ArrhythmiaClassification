import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np


if __name__ == '__main__':
    cur_path = os.getcwd()
    typeStr = "N"
    file_path = cur_path + "/Train-DB/" + typeStr + "/datas-resample.csv"
    datas: pd.DataFrame = pd.read_csv(file_path)

    fig, ax = plt.subplots()


    for i in range(50):
        data = datas["raw_data"][i]
        data = json.loads(data)
        arr = np.array(data)
        mean = arr.mean()
        mean_arr = arr - mean
        data = mean_arr.tolist()
        ax.plot(data)

    plt.show()
