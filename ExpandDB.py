import matplotlib.pyplot as plt
from matplotlib.axes._subplots import Axes
from matplotlib.font_manager import FontManager
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pylab import mpl
import subprocess
import pandas as pd
import os
import numpy as np
import json
import wfdb.processing as wfp
import math


def get_matplot_zh_font():
    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)

    output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
    fonts = output.decode()
    fontArr = fonts.split('\n')
    zh_fonts = set(f.split(',', 1)[0] for f in fontArr)
    # zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
    available = list(mat_fonts & zh_fonts)
    # remove .xxxx
    available = [i for i in available if not i.startswith('.')]

    print('*' * 10, '可用的字体', '*' * 10)
    for f in available:
        print(f)
    return available

def set_matplot_zh_font():
    available = get_matplot_zh_font()
    if len(available) > 0:
        mpl.rcParams['font.sans-serif'] = available[0]    # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False          # 解决保存图像是负号'-'显示为方块的问题

def detext_xqrs(datas, disp_err=False) -> int:
    xqrs = wfp.XQRS(np.array(datas), fs=250)
    xqrs.detect(verbose=False)
    if len(xqrs.qrs_inds) == 0:
        if disp_err:
            ax: Axes = plt.subplot()
            ax.plot(datas)
            plt.show()
        print("QRS Detect failed. return 75")
        return 75
    index = int(xqrs.qrs_inds[0])
    return index

def expand_ecg(datas, type=1, gain=1.1, bate=0.9) -> list:
    datas_expand = datas.copy()
    if type == 0:  # R-
        r_peak_index = detext_xqrs(datas)
        i_r = r_peak_index + 10
        if i_r > 179:
            i_r = 179
        for i in range(r_peak_index - 10, i_r, 1):
        # for i in range(len(datas)):
            datas_expand[i] *= bate
        return datas_expand
    if type == 1:  # R+
        # r_peak_index = detext_xqrs(datas)
        # for i in range(r_peak_index - 5, r_peak_index + 5, 1):
        for i in range(len(datas)):
            datas_expand[i] *= gain
        return datas_expand
    elif type == 2:  # R scale
        expand_width_percent = 0.15
        expand_width_step = int(math.floor(1.0 / expand_width_percent))
        for i in range(75 + 20, 75 - 20, -expand_width_step):
            sub_datas = datas_expand[i - 1:i + 1]

            # Max
            # s = max(sub_datas)
            # Mean
            s = 0.
            for sub_d in sub_datas:
                s += sub_d
            s = s / len(sub_datas)

            datas_expand.insert(i, s)

        iC = int((20 + 25.0) / expand_width_step / 2.0) - 1
        # datas_expand.extend([0 for j in range(iC)])
        # r_peak_index = detext_xqrs(datas_expand)
        r_peak_index = 75 + iC
        # if r_peak_index < iC:
        #     r_peak_index = iC
        datas_expand = datas_expand[r_peak_index - 75:r_peak_index + 105]
        return datas_expand
    elif type == 3:
        datas_expand = expand_ecg(datas_expand, 1)
        datas_expand = expand_ecg(datas_expand, 2)
        return datas_expand
    elif type == 4:
        datas_expand = expand_ecg(datas_expand, 0)
        datas_expand = expand_ecg(datas_expand, 2)
        return datas_expand
    elif type == 5:  # --
        return expand_ecg(datas_expand, 0, bate=0.82)
    elif type == 6:  # ++
        return expand_ecg(datas_expand, 1, gain=1.18)
    elif type == 7:  # -+All Peak Up/Down
        peaks, _ = wfp.find_peaks(datas_expand)
        peaks = peaks.tolist()
        n_peak = []
        last_peak_value = 0
        for k in range(len(peaks) - 2):
            if abs(peaks[k] - last_peak_value) > 10:
                n_peak.append(peaks[k])
                last_peak_value = peaks[k]
        for p in n_peak:
            p_l = p - 5
            if p_l < 0:
                p_l = 0
            p_r = p + 5
            if p_r > 179:
                p_r = 179
            for i in range(p_l, p_r):
                datas_expand[i] *= gain
                # if datas_expand[i] > 0:
                #     datas_expand[i] *= gain
                # else:
                #     datas_expand[i] *= bate

        return datas_expand
    elif type == 8:  # ++--All Peak Random 不好
        peaks, _ = wfp.find_peaks(datas_expand)
        peaks = peaks.tolist()
        n_peak = []
        last_peak_value = 0
        for k in range(len(peaks) - 2):
            if abs(peaks[k] - last_peak_value) > 10:
                n_peak.append(peaks[k])
                last_peak_value = peaks[k]
        for p in n_peak:
            p_l = p - 5
            if p_l < 0:
                p_l = 0
            p_r = p + 5
            if p_r > 179:
                p_r = 179
            r_i = np.random.randint(2)
            for i in range(p_l, p_r):
                if r_i == 0:
                    datas_expand[i] *= gain
                else:
                    datas_expand[i] *= bate
        return datas_expand
    elif type == 9:  # ++--All Peak Up +
        peaks, _ = wfp.find_peaks(datas_expand)
        peaks = peaks.tolist()
        n_peak = []
        last_peak_value = 0
        for k in range(len(peaks) - 2):
            if abs(peaks[k] - last_peak_value) > 10:
                n_peak.append(peaks[k])
                last_peak_value = peaks[k]
        for p in n_peak:
            p_l = p - 5
            if p_l < 0:
                p_l = 0
            p_r = p + 5
            if p_r > 179:
                p_r = 179
            for i in range(p_l, p_r):
                datas_expand[i] *= 1.15
        return datas_expand

    else:
        return datas


def expand_test():
    cur_dir = os.getcwd()
    ecg_type = "S"

    df = pd.read_csv(cur_dir + "/Train-DB-Checked/" + ecg_type + "/datas-mean-old.csv")

    s1 = df.iloc[0]

    datas = json.loads(s1['raw_data'])



    plt.figure(figsize=(15, 8))

    # Expand-0
    # Expand-1
    ax: Axes = plt.subplot(231)
    ax.plot(expand_ecg(datas, 0), 'r')
    ax.plot(expand_ecg(datas, 1), 'r')
    ax.plot(expand_ecg(datas, 5), 'r')
    ax.plot(expand_ecg(datas, 6), 'r')
    ax.plot(datas, 'k')
    ax.set_title("全部波增强0,1,5,6")
    codes = [Path.MOVETO] + [Path.LINETO]
    codes = codes * 2
    vertices = [(75 - 5, -0.5), (75 - 5, 3),
                (75 + 5, -0.5), (75 + 5, 3)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    path_patch = PathPatch(path, edgecolor='green')
    ax.add_patch(path_patch)

    # Expand-2
    datas_expand2 = expand_ecg(datas, 2)
    r_peak_index = detext_xqrs(datas_expand2)
    datas_expand2 = datas_expand2[r_peak_index-75:r_peak_index+105]
    ax2 = plt.subplot(232)
    ax2.set_title("波形拉长2")
    ax2.plot(datas, 'k')
    ax2.plot(datas_expand2, 'r')

    # Expand 7, 8, 9
    ax7 = plt.subplot(233)
    ax7.set_title("Peaks增强7,9")
    ax7.plot(expand_ecg(datas, 7), 'r')
    # ax7.plot(expand_ecg(datas, 8), 'r')
    ax7.plot(expand_ecg(datas, 9), 'r')
    ax7.plot(datas, 'k')
    # codes = [Path.MOVETO] + [Path.LINETO]
    # codes = codes * 2
    # vertices = [(75 - 5, -0.5), (75 - 5, 3), (75 + 5, -0.5), (75 + 5, 3)]
    # vertices = np.array(vertices, float)
    # path = Path(vertices, codes)
    # path_patch = PathPatch(path, edgecolor='green')
    # ax7.add_patch(path_patch)


    # Expand-3, combine 1 and 2.
    datas_expand3 = expand_ecg(datas, 3)
    r_peak_index = detext_xqrs(datas_expand3)
    datas_expand3 = datas_expand3[r_peak_index - 75:r_peak_index + 105]
    ax3 = plt.subplot(234)
    ax3.set_title("结合3=1,2")
    ax3.plot(datas, 'k')
    ax3.plot(datas_expand3, 'r')

    # Expand-4, combine 0 and 2.
    datas_expand4 = expand_ecg(datas, 4)
    r_peak_index = detext_xqrs(datas_expand4)
    datas_expand4 = datas_expand4[r_peak_index - 75:r_peak_index + 105]
    ax4 = plt.subplot(235)
    ax4.set_title("结合4=0,2")
    ax4.plot(datas, 'k')
    ax4.plot(datas_expand4, 'r')

    plt.show()

def expand_dataset(type=None, expand_type_list=[0, 1, 2, 3, 4, 5, 6, 7, 9]):
    if type is None:
        return
    cur_path = os.getcwd() + "/Train-DB-Checked/"
    file_path = cur_path + type + "/datas-mean.csv"
    target_path = cur_path + type + "/datas-mean-old.csv"
    if os.path.exists(target_path):
        print("File exist, skip rename.")
    else:
        if os.path.exists(file_path):
            os.rename(file_path, target_path)

    df = pd.read_csv(target_path)
    expand_df = pd.DataFrame()

    length = len(df.index)
    print("Length:", length)

    save_path = cur_path + type + "/datas_mean.csv"

    for i in range(length):

        print("Proccess index:", i)

        row_org = df.iloc[i]
        raw_data = json.loads(row_org['raw_data'])

        # data_expands = []

        for j in expand_type_list:
            t_ex_data = expand_ecg(raw_data, j)
            if len(t_ex_data) == 180:
                row_expand = pd.Series({'raw_data': json.dumps(t_ex_data)})
                expand_df = expand_df.append(row_expand, ignore_index=True)
            # data_expands.append()
        # data_expand0 = expand_ecg(raw_data, 0)
        # data_expand1 = expand_ecg(raw_data, 1)
        # data_expand2 = expand_ecg(raw_data, 2)
        # data_expand3 = expand_ecg(raw_data, 3)
        # data_expand4 = expand_ecg(raw_data, 4)
        # data_expand5 = expand_ecg(raw_data, 5)
        # data_expand6 = expand_ecg(raw_data, 6)
        # data_expand7 = expand_ecg(raw_data, 7)
        # data_expand9 = expand_ecg(raw_data, 9)

        # if len(data_expand0) == 180:
        #     row_expand1 = pd.Series({'raw_data': json.dumps(data_expand1)})
        #     expand_df = expand_df.append(row_expand1, ignore_index=True)
        # if len(data_expand1) == 180:
        #     row_expand1 = pd.Series({'raw_data': json.dumps(data_expand1)})
        #     expand_df = expand_df.append(row_expand1, ignore_index=True)
        # if len(data_expand2) == 180:
        #     row_expand2 = pd.Series({'raw_data': json.dumps(data_expand2)})
        #     expand_df = expand_df.append(row_expand2, ignore_index=True)
        # if len(data_expand3) == 180:
        #     row_expand3 = pd.Series({'raw_data': json.dumps(data_expand3)})
        #     expand_df = expand_df.append(row_expand3, ignore_index=True)
        # if len(data_expand4) == 180:
        #     row_expand4 = pd.Series({'raw_data': json.dumps(data_expand4)})
        #     expand_df = expand_df.append(row_expand4, ignore_index=True)

    expand_df = expand_df.append(df, ignore_index=True, sort=False)
    ex_length = len(expand_df.index)
    print("Expend Length:", ex_length)
    expand_df.to_csv(file_path)
    with open(cur_path + type + "/info-%s.txt"%(str(ex_length)), 'wt') as f:
        f.write("num:%s"%(str(ex_length)))

if __name__ == '__main__':
    # set_matplot_zh_font()
    # expand_test()
    # expand_dataset(type='F')
    # expand_dataset(type='Q')
    # expand_dataset(type='S', expand_type_list=[0, 1, 2])
    expand_dataset(type='V', expand_type_list=[0, 1])

