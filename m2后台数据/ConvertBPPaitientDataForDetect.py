import os
import pandas as pd
import numpy as np


def read_bp_data(file_name: str, archive=False) -> list:
    cur_path = os.getcwd() + "/"
    file_path = cur_path + file_name
    if not os.path.exists(file_path):
        print("file doesn't exist.", file_path)
        return

    org_arr: np.array = np.loadtxt(file_path, dtype=np.float64)
    sig_arr = org_arr / 385.
    mean_value = sig_arr.mean()

    final_arr: np.array = sig_arr - mean_value

    save_path = cur_path + "proc/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path += file_name

    if archive:
        np.savetxt(save_path, final_arr, fmt="%.18f")

    return final_arr.tolist()



def convert_bp_paitient_data_for_detect(file_name):
    # Read Data
    cur_path = os.getcwd() + "/"

if __name__ == "__main__":
    base_path = os.getcwd()
    items = os.listdir(base_path)
    txt_items = list(filter(lambda x: x.endswith(".txt"), items))

    for item in txt_items:
        print("Converting file ...", item)
        arr = read_bp_data(item, archive=True)
        print("Convert finished. Length is ", len(arr))
        