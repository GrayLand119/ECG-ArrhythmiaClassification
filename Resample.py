import wfdb as wf
import wfdb.processing
import os
import numpy as np
import matplotlib.pyplot as plot
from BeatModel import *
import pandas as pd
import json


def test_read_db(file_name="100"):
    cur_dir = os.getcwd()
    record_dir = "/MIT-DB"
    channel = 0
    file_path = cur_dir + os.path.join(record_dir, file_name)
    record360 = wf.rdrecord(file_path, channels=[channel], sampto=1200)
    # ann360 = wf.rdann(file_path, "atr", sampto=1200, summarize_labels=True)
    ann360 = wf.rdann(file_path, "atr", summarize_labels=True)
    print(ann360.aux_note)

def resample_to_csv():
    name_lists = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                  200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234]

    cur_dir = os.getcwd()
    # 默认 channel 0
    channel = 0
    record_dir = "/MIT-DB"

    finish_lists = [234, 233, 232, 231, 230, 223, 222, 221, 220, 219, 217, 215, 214, 213, 212, 210, 209, 124, 123, 122, 121, 118, 117, 116, 115, 114, 113, 112, 111, 109, 103, 108, 107, 106, 105, 104, 102, 101, 100, 228, 119, 208, 207, 205, 203, 202, 201, 200]
    error_list = [224, 225, 226, 227, 229, 218, 216, 211, 120, 110, 206, 204]
    dif_list = []
    dif_list.extend(finish_lists)
    dif_list.extend(error_list)
    # finish_lists.extend(error_list)
    """
    finish_lists:
    [234, 233, 232, 231, 230, 223, 222, 221, 220, 219, 217, 215, 214, 213, 212, 210, 209, 124, 123, 122, 121, 118, 117, 116, 115, 114, 113, 112, 111, 109, 103, 108, 107, 106, 105, 104, 102, 101, 100, 228, 119, 208, 207, 205, 203, 202, 201, 200]
    has_error:
    {'124': 1, '122': 1, '121': 1, '117': 1, '116': 1, '115': 1, '113': 1, '112': 1, '109': 1, '103': 1, '104': 1, '102': 1, '100': 1, '208': 1, '205': 1, '202': 1, '200': 1}
    error_list:
    [224, 225, 226, 227, 229, 218, 216, 211, 120, 110, 206, 204]
    """
    has_error = {}

    name_lists = list(set(name_lists).difference(set(dif_list)))

    while len(name_lists):
        file_name = str(name_lists.pop())

        print("Remain:", len(name_lists))
        print("Processing...", file_name)

        file_path = cur_dir + os.path.join(record_dir, file_name)

        # Read record
        try:
            record360 = wf.rdrecord(file_path, channels=[channel])
            ann360 = wf.rdann(file_path, "atr")
        except Exception as e:
            print("%s error! continue..." % (file_name))
            error_list.append(int(file_name))
            continue

        sig = record360.p_signal
        sig = np.array([x[0] for x in sig])
        record250, ann250 = wf.processing.resample_singlechan(sig, ann360, record360.fs, 250)
        # record250 = wf.processing.resample_sig(adc, record360.fs, 250)[0]

        symbol = ann250.symbol
        sample = ann250.sample
        aux_note = ann250.aux_note
        record = record250.tolist()

        save_path = cur_dir + "/AnalysisData/" + file_name + ".csv"
        file_exist = os.path.exists(save_path)

        if file_exist:
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame()
        print("Writting....")
        for i in range(len(sample)):
            s = symbol[i]
            if s == '+':
                continue
            index = sample[i]
            if index < 75:
                continue

            model = BeatModel()
            model.raw_data = record[index - 75:index + 105]
            if len(model.raw_data) != 180:
                print("%d length not 180, in file %s" % (index, file_name))
                has_error[file_name] = has_error.get(file_name, 0) + 1
                continue
            model.symbol = symbol[i]
            model.r_peak_index = int(sample[i])
            model.aux_note = aux_note[i]

            d = model.toDict()
            df = df.append(d, ignore_index=True)
        finish_lists.append(int(file_name))
        print("finish_lists:\n",finish_lists)
        print("has_error:\n", has_error)
        print("error_list:\n", error_list)

        # print(df)
        print("Save CSV :", save_path)
        df.to_csv(save_path, mode='w' if file_exist else "a", index=False)

def filter_symbol(symbol='N'):

    name_lists = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122,
                  123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
                  228, 230, 231, 232, 233, 234]


    cur_dir = os.getcwd()
    file_name = "100"
    countN = 0
    hash_symbols = {}
    hash_symbols_org = {}

    while len(name_lists):
        file_name = str(name_lists.pop())
        print("Processing file ...", file_name)

        file_path = cur_dir + "/AnalysisData/" + file_name + ".csv"
        try:
            datas = pd.read_csv(file_path, dtype={'r_peak_index': float,
                                                  'raw_data': str,
                                                  'symbol': str,
                                                  'aux_note': str})
        except Exception as e:
            print("Read file %s error... skip" % (file_name))
            continue

        # df = pd.DataFrame()

        symbols = datas['symbol']
        hash_symbols_in_file = {}

        dfs = {}
        # df_N = pd.DataFrame()
        # df_S = pd.DataFrame()
        # df_V = pd.DataFrame()
        # df_F = pd.DataFrame()
        # df_Q = pd.DataFrame()

        for index in range(len(symbols)):
            symbol = symbols[index]
            s1 = datas.iloc[index]

            tempS = ""

            if isTypeN(symbol):
                tempS = 'N'
                # s1['symbol'] = tempS
            elif isTypeS(symbol):
                tempS = 'S'
                # s1['symbol'] = tempS
            elif isTypeV(symbol):
                tempS = 'V'
                # s1['symbol'] = tempS
            elif isTypeF(symbol):
                tempS = 'F'
                # s1['symbol'] = tempS
            else:
                tempS = 'Q'
                # s1['symbol'] = tempS

            tdf = dfs.get(tempS, pd.DataFrame())
            tdf = tdf.append(s1, ignore_index=True)
            dfs[tempS] = tdf

            # 统计
            hash_symbols[tempS] = hash_symbols.get(tempS, 0) + 1
            hash_symbols_org[symbol] = hash_symbols_org.get(symbol, 0) + 1
            hash_symbols_in_file[tempS] = hash_symbols_in_file.get(tempS, 0) + 1


        print("hash_symbols_in_file:\n", hash_symbols_in_file)

        for key in hash_symbols_in_file.keys():
            save_dir = cur_dir + "/Train-DB/" + key
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + '/datas-resample.csv'
            file_exist = os.path.exists(save_path)

            tdf = dfs[key]

            # Check
            sel_length = len(tdf.index)
            has_err = False
            for i in range(sel_length):
                check_datas = json.loads(tdf.iloc[i]['raw_data'])
                check_length = len(check_datas)
                if check_length != 180:
                    print("!!! %s file length not equal 180 at index %d" % (file_name, i))
                    has_err = True
                    break

            if has_err:
                print("skip file: ", file_name)
                continue

            print("Save CSV :", save_path)
            if file_exist:
                tdf.to_csv(save_path, mode='a', index=False, header=False)
            else:
                tdf.to_csv(save_path, mode='w', index=False, header=True)




    #  'N': 74535
    #  '~': 571
    #  'J': 83
    #  'V': 6902
    #  'A': 2546
    #  'F': 802
    #  '|': 131
    #  'R': 7257
    #  'j': 229
    #  '"': 437
    #  'x': 193
    #  'L': 8074
    #  'Q': 15
    #  'a': 150
    #  'E': 106
    #  'e': 16
    #  'S': 2
    #  '[': 6
    #  '!': 472
    #  ']': 6
    try:
        with open(cur_dir + "/Train-DB/info.txt", 'w') as f:
            f.write(json.dumps(hash_symbols))
            f.write("Original:\n")
            org_str = json.dumps(hash_symbols_org)
            org_str = org_str.replace(', ', '\n')
            f.write(org_str)
    except Exception as e:
        print(e)

    print(hash_symbols)
    pass

def isTypeN(type: str) -> bool:
    if type == 'N' or type == 'R' or type == 'L' or type == 'j' or type == 'e':
        return True
    return False

def isTypeS(type: str) -> bool:
    if type == 'A' or type == 'a' or type == 'J' or type == 'S':
        return True
    return False

def isTypeV(type: str) -> bool:
    if type == 'V' or type == 'E':
        return True
    return False

def isTypeF(type: str) -> bool:
    if type == 'F':
        return True
    return False

def isTypeQ(type: str) -> bool:
    # other
    pass

def mean_datas():
    cur_dir = os.getcwd()
    type_dir = os.listdir(cur_dir + "/Train-DB-Checked")
    dirs = list(filter(lambda x: os.path.isdir(cur_dir + "/Train-DB-Checked/" + x), type_dir))
    for t_dir in dirs:
        file_path = cur_dir + "/Train-DB-Checked/" + t_dir + "/datas-resample.csv"
        df: pd.DataFrame = pd.read_csv(file_path, dtype=str)
        count = len(df.index)

        save_path = cur_dir + "/Train-DB-Checked/" + t_dir + "/datas-mean.csv"
        mean_df = pd.DataFrame()
        print("count:", count)

        for i in range(count):
            print("Processing...", str(i))
            s1 = df.iloc[i]
            datas = np.array(json.loads(s1['raw_data']))
            mean_value = datas.mean()
            mean_arr = datas - mean_value
            mean_df = mean_df.append({"r_peak_index": s1['r_peak_index'],
                                      "raw_data": json.dumps(mean_arr.tolist()),
                                      "symbol": s1['symbol'],
                                      "aux_note": s1['aux_note']}, ignore_index=True)
        print("Writting:", t_dir)
        mean_df.to_csv(save_path)





if __name__ == '__main__':
    mean_datas()
    # resample_to_csv()
    # filter_symbol()
    # import json
    # result = json.dumps("{'N': 74535, '~': 571, 'J': 83, 'V': 6902, 'A': 2546, 'F': 802, '|': 131, 'R': 7257, 'j': 229, '"': 437, \'x\': 193, '\\L': 8074, '\\Q': 15, '\\a': 150, 'E': 106, 'e': 16, 'S': 2, \'[\': 6, '!': 472,\']\': 6}")
    # print(result)

