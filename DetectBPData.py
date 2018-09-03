import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import wfdb.processing as wfp
import tensorflow as tf
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.axes._subplots import Axes
from matplotlib.backends.backend_pdf import PdfPages


def load_labels() -> dict:
    labels_dic = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
    return labels_dic


def load_model(model_name) -> ():

    saver_file_path = os.getcwd() + "/TrainingModels/" + model_name

    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(os.getcwd() + "/TrainingModels/" + model_name + ".meta")
    saver.restore(sess, saver_file_path)

    graph = tf.get_default_graph()
    input_values = graph.get_operation_by_name("input_values").outputs[0]
    # x_ = tf.reshape(input_values, [1, INPUT_NODE, 1])
    pred_tensor = tf.get_collection("pred_tensor")[0]

    return input_values, pred_tensor
    # logits = graph.get_operation_by_name("trainning/logits:0")
    # fake_values = np.arange(-0.5, 1.3, 0.01)

    # df = pd.read_csv(os.getcwd() + '/Train-DB-Checked/N/datas-mean.csv')

    # times = 10
    # labels = load_labels()
    # while times > 0:
    #     index = np.random.randint(95000, 100000)
    #     raw_data = df.iloc[index]['raw_data']
    #     datas = json.loads(raw_data)
    #     # x_ = tf.reshape(input_values, [1, INPUT_NODE, 1])
    #     # Pred
    #     # sess.run(final_tensor, feed_dict={input_values: [fake_values]})
    #     result = sess.run(pred_tensor, feed_dict={input_values: [datas]}).tolist()[0]
    #     max_index = result.index(max(result))
    #     print(result)
    #     print(max_index)
    #     for k, v in labels.items():
    #         if v == max_index:
    #             print("Type is %s" % (k))
    #             break
                # sess.run(tf.argmax(final_tensor, 1))
                # sess.run(tf.get_default_graph().get_tensor_by_name("training_count:0"))
                # for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
                #     print(op.name, op.values())
    pass

def detext_xqrs(datas, disp_err=False) -> list:
    if type(datas) == 'list':
        xqrs = wfp.XQRS(np.array(datas), fs=250)
    else:
        xqrs = wfp.XQRS(datas, fs=250)
    xqrs.detect(verbose=False)
    if len(xqrs.qrs_inds) == 0:
        print("QRS Detect failed. return 75")
    # index = int(xqrs.qrs_inds[0])
    return xqrs.qrs_inds

def detet_with_file(labels, sess, input_values, pred_tensor, file_path, pdf=None, start_min=0, end_min=20) -> bool:

    content = ""
    max_limit = 2.5
    min_limit = -2.5

    f = open(file_path, 'rb')
    size = os.path.getsize(file_path)
    print("Size:", size)
    # start_min = start_min
    # end_min = 20
    start_length = start_min * 60 * 250
    end_length = end_min * 60 * 250 + start_length

    print("DataLength:%d" % (end_length - start_min))
    for i in range(start_length, end_length):
        if i == start_length:
            try:
                t_seek_i = min(size - 1, start_length * 22)
                f.seek(t_seek_i, 1)
            except Exception as e:
                print(e)
                return False
            f.readline()
            continue
        t_read: bytes = f.readline()
        t_read = t_read.decode()
        # t_read = f.readline()
        if t_read == "":
            if i == start_length + 1:
                return False
            else:
                return True
        content += t_read

    # print(content)
    f.close()

    contentIO = StringIO(content)

    np_arr1: np.array = np.loadtxt(contentIO, dtype=np.float64)
    print("Load finished.")

    xprs_arr = detext_xqrs(np_arr1)

    # print(xprs_arr)
    all_data = np_arr1.tolist()

    if pdf:
        plt.figure(figsize=(16*3, 9*3))
    else:
        plt.figure(figsize=(8, 5))

    ax1 = plt.subplot(332)
    ax1.set_title("Fig-1\nXQRS Detect")
    ax12 = plt.subplot(333)
    ax12.set_title("Fig-2\nMean<Fig-1>")
    ax2: Axes = plt.subplot(312)
    ax2.set_title("Origin Data %d~%d minutes, total %d minutes" % (start_min, end_min, end_min - start_min))
    ax2.axis(ymin=min_limit, ymax=max_limit)
    ax3 = plt.subplot(313)
    ax3.set_title("Output Unnormal Data")

    beat_datas = []
    unnormal_beats = []

    unnormal_index = 0



    result_set = {}
    i_total = 0
    for index in xprs_arr:
        # if index < 75 or index > :
        #     continue
        data = all_data[index - 75: index + 105]
        if len(data) != 180:
            continue
        # noise depart
        chunk_data = all_data[max(index - 360, 0): index + 360]
        if max(chunk_data) > max_limit:
            continue
        if min(chunk_data) < min_limit:
            continue

        i_total += 1

        beat_datas.extend(data)
        ax1.plot(data)
        # mean
        data_np = np.array(data)

        mean = data_np.mean()
        data_mean = data_np - mean

        # beat_means.extend(data_mean)
        ax12.plot(data_mean)

        result = sess.run(pred_tensor, feed_dict={input_values: [data_mean.tolist()]}).tolist()[0]
        max_result = max(result)
        max_index = result.index(max_result)

        padding_w = 180

        for k, v in labels.items():

            if k != "N" and k != 'S' and k != "Q":
                codes = [Path.MOVETO] + [Path.LINETO]
                # disp Error Line in ax 2
                vertices = [(index, -5), (index, 5)]
                vertices = np.array(vertices, float)
                path = Path(vertices, codes)
                path_patch = PathPatch(path, edgecolor='red')
                ax2.add_patch(path_patch)
                # disp Annotation in ax3
                t_i = 75 + (180 + padding_w) * unnormal_index
                vertices = [(t_i, -5), (t_i, 5)]
                vertices = np.array(vertices, float)
                path = Path(vertices, codes)
                path_patch = PathPatch(path, edgecolor='red')
                ax3.add_patch(path_patch)
                ax3.text(t_i - 15, 1., k, {'color': 'red'})

                unnormal_index += 1
                unnormal_b = all_data[index - 75: index + 105]
                t_np = np.array(unnormal_b)
                t_np -= t_np.mean()
                unnormal_b = t_np.tolist()
                unnormal_beats.extend(unnormal_b)
                unnormal_beats.extend(np.zeros(padding_w, dtype=np.float64))
                # cover padding
                codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
                t_i += 105
                vertices = [(t_i, 3), (t_i + padding_w, 3), (t_i + padding_w, -3), (t_i, -3), (0, 0)]
                vertices = np.array(vertices, float)
                path = Path(vertices, codes)
                path_patch = PathPatch(path, edgecolor='None', facecolor="green")
                ax3.add_patch(path_patch)

            if v == max_index:
                result_set[k] = result_set.get(k, 0) + 1
                print("Type is %s, R-Peak index %d, acc is %.3f" % (k, index, max_result))
                # t_r = k + "," + str(result)
                # detect_result[index] = t_r
                break

    all_data_length = len(all_data)
    codes = [Path.MOVETO] + [Path.LINETO] + [Path.MOVETO] + [Path.LINETO]
    vertices = [(0, max_limit), (all_data_length - 1, max_limit), (0, min_limit), (all_data_length - 1, min_limit)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    path_patch = PathPatch(path, edgecolor='green')
    ax2.add_patch(path_patch)
    if pdf:
        ax2.plot(all_data, linewidth=0.1)
        ax3.plot(unnormal_beats, linewidth=0.1)
    else:
        ax2.plot(all_data, linewidth=1)
        ax3.plot(unnormal_beats, linewidth=1)
    # with open()

    info_content = "Result:\n"
    info_content += "Total: %d\n" % (i_total)
    for k, v in result_set.items():
        info_content += "Type %s count %d.\n" % (k, v)
    print(info_content)

    ax0 = plt.subplot(331)
    ax0.set_axis_off()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax0.text(.2, .5, info_content, fontdict={'color': 'k'}, bbox=props)

    # plt.show()
    # pdf_path = file_path.replace(".txt", "_%d_%d.pdf" % (start_min, end_min))
    if pdf:
        pdf.savefig(dpi=720)
        return True
    else:
        plt.show()

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    labels = load_labels()

    print("Loading Model...")
    input_values, pred_tensor = load_model("ModelC.ckpt-12000")
    print("Load Model finished.")

    cur_path = os.getcwd() + "/m2后台数据/proc/"
    file_path = cur_path + "shihu2018-06-05_17_46_32.txt"

    print("Loading file...")


    has_data = True
    # pdf_path = file_path.replace(".txt", ".pdf")
    # pdf = PdfPages(pdf_path)
    page_min = 10
    start_min = 10
    end_min = page_min
    while has_data:
        has_data = detet_with_file(labels, sess, input_values, pred_tensor, file_path, pdf=None, start_min=start_min,
                                   end_min=end_min)
        start_min += page_min
        end_min += page_min
        break

    # pdf.close()

    # arr = np_arr1.tolist()
    #

    #
    # ax1 = plt.subplot(311)
    # ax1.plot(arr[:sep_length])
    #
    # ax2 = plt.subplot(312)
    # ax2.plot(arr[sep_length:sep_length * 2])
    #
    # ax3 = plt.subplot(313)
    # ax3.plot(arr[sep_length * 2:sep_length * 3])
    #
    # print("Disp....")
    # plt.show()

    pass