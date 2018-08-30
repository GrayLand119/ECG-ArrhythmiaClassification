import os
import pandas as pd


def count_samples(type=None, write_to_info=False) -> int:
    cur_dir = os.getcwd()
    type_dir = os.listdir(cur_dir + "/Train-DB-Checked")

    if type is None:
        dirs = list(filter(lambda x: os.path.isdir(cur_dir + "/Train-DB-Checked/" + x), type_dir))
    else:
        dirs = ["%s/Train-DB-Checked/%s/datas-resample.csv" % (cur_dir, type)]

    for t_dir in dirs:
        df: pd.DataFrame = pd.read_csv("%s/Train-DB-Checked/%s/datas-resample.csv" % (cur_dir, t_dir))
        count = int(len(df.index))
        if write_to_info:
            with open("%s/Train-DB-Checked/%s/info-%s.txt" % (cur_dir, t_dir, str(count)), 'w') as f:
                f.write("nums:%d" % (int(count)))
        else:
            return count

    return 0

if __name__ == '__main__':
    count_samples(write_to_info=True)
    pass

