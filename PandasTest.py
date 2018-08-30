import pandas as pd
import os


if __name__ == '__main__':

    cur_dir = os.getcwd()
    save_path = cur_dir + "/AnalysisData/100.csv"

    datas = pd.read_csv(save_path, dtype={'r_peak_index': float,
                                          'raw_data': str,
                                          'symbol': str})

    print(datas[datas['r_peak_index'] > 300.])