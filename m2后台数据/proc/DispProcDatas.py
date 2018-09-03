import matplotlib.pyplot as plt
import numpy as np
import os
from io import StringIO
import binascii as asc


def test():
    cur_path = os.getcwd() + "/"
    file_path = cur_path + "2018-05-23_06_42_42.txt"

    print("Loading file...")
    f = open(file_path, 'rt')
    content = ""
    disp_min = 2
    sep_length = disp_min * 60 * 250

    for i in range(sep_length * 3):
        # if i == 0:
        #     f.seek(10000, 1)
        #     f.readline()
        #     continue
        # line: bytes = f.readline()
        # line = line.decode()
        # print("Line:", line, end="")

        content += f.readline()

    # print(content)
    f.close()

    contentIO = StringIO(content)

    np_arr1: np.array = np.loadtxt(contentIO, dtype=np.float64)
    print("Load finished.")

    arr = np_arr1.tolist()

    plt.figure(figsize=(15, 8))

    ax1 = plt.subplot(311)
    ax1.plot(arr[:sep_length])

    ax2 = plt.subplot(312)
    ax2.plot(arr[sep_length:sep_length*2])

    ax3 = plt.subplot(313)
    ax3.plot(arr[sep_length*2:sep_length*3])

    print("Disp....")
    plt.show()

if __name__ == "__main__":
    test()
    pass