import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import json
import os
import math

BATCH_SIZE = 500
LEARNNING_RATE = 0.001
TRAINNING_STEPS = 100000

INPUT_NODE = 180
OUTPUT_NODE = 5

# 258 x 5
LAYER1_OUTPUT_SIZE = 258
LAYER1_OUTPUT_DEEP = 5

# N 101785
N_NUM = 100000
TRAINING_N_NUM = 80000
TESTING_N_NUM = 20000
VALIDATION_N_NUM = 20000
# S 4294
S_NUM = 40000
TRAINING_S_NUM = math.floor(S_NUM * 0.8)
TESTING_S_NUM = math.floor(S_NUM * 0.1)
VALIDATION_S_NUM = math.ceil(S_NUM * 0.1)
# V 8544
V_NUM = 40000
TRAINING_V_NUM = math.floor(V_NUM * 0.8)
TESTING_V_NUM = math.floor(V_NUM * 0.1)
VALIDATION_V_NUM = math.ceil(V_NUM * 0.1)
# F 827
F_NUM = 40000
TRAINING_F_NUM = 32000
TESTING_F_NUM = 4000
VALIDATION_F_NUM = 4000
# Q 10386
Q_NUM = 40000
TRAINING_Q_NUM = math.floor(Q_NUM * 0.8)
TESTING_Q_NUM = math.floor(Q_NUM * 0.1)
VALIDATION_Q_NUM = math.ceil(Q_NUM * 0.1)

TRAIN_MODEL_NAME = "ModelC"
TEST_MODEL_NAME = "ModelC"


def load_samples():
    cur_path = os.getcwd() + "/Train-DB-Checked/"

    samples = {}
    class_types = list("NSVFQ")

    for type_s in class_types:
        t_data = pd.read_csv(cur_path + type_s + '/datas-training.csv')
        t_length = len(t_data.index)
        samples["training_" + type_s] = t_data
        samples["training_" + type_s + "_length"] = t_length
        print("%s training set length %d " %(type_s, t_length))

        t_data = pd.read_csv(cur_path + type_s + '/datas-testing.csv')
        t_length = len(t_data.index)
        samples["testing_" + type_s] = t_data
        samples["testing_" + type_s + "_length"] = t_length
        print("%s testing set length %d " % (type_s, t_length))

        t_data = pd.read_csv(cur_path + type_s + '/datas-validation_path.csv')
        t_length = len(t_data.index)
        samples["validation_" + type_s] = t_data
        samples["validation_" + type_s + "_length"] = t_length
        print("%s validation set length %d " % (type_s, t_length))
    # samples_N = pd.read_csv(cur_path + 'N/datas-mean.csv')
    # samples_S = pd.read_csv(cur_path + 'S/datas-mean.csv')
    # samples_V = pd.read_csv(cur_path + 'V/datas-mean.csv')
    # samples_F = pd.read_csv(cur_path + 'F/datas-mean.csv')
    # samples_Q = pd.read_csv(cur_path + 'Q/datas-mean.csv')

    # return {"N": samples_N, "S": samples_S, "V": samples_V, "F": samples_F, "Q": samples_Q}
    return samples

def labels_map():
    pass


def get_batch(samples_dict: dict, batch_size, sample_type: str = 'sample_type'):
    """
    sample_type has 3 types : 'training', 'testing', 'validation'
    :return: (samples, labels)
    """
    # t_length = samples_dict[sample_type + "_N_length"]
    indices_N = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    indices_S = np.random.choice(samples_dict[sample_type + "_S_length"], batch_size, False)
    indices_V = np.random.choice(samples_dict[sample_type + "_V_length"], batch_size, False)
    indices_F = np.random.choice(samples_dict[sample_type + "_F_length"], batch_size, False)
    indices_Q = np.random.choice(samples_dict[sample_type + "_Q_length"], batch_size, False)

    # if sample_type == 'training':
    #     offset_N = 0
    #     offset_S = 0
    #     offset_V = 0
    #     offset_F = 0
    #     offset_Q = 0
    # elif sample_type == 'testing':
    #     indices_N = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_S = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_V = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_F = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_Q = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     offset_N = TRAINING_N_NUM
    #     offset_S = TRAINING_S_NUM
    #     offset_V = TRAINING_V_NUM
    #     offset_F = TRAINING_F_NUM
    #     offset_Q = TRAINING_Q_NUM
    # elif sample_type == 'validation':
    #     indices_N = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_S = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_V = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_F = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     indices_Q = np.random.choice(samples_dict[sample_type + "_N_length"], batch_size, False)
    #     offset_N = TRAINING_N_NUM + TESTING_N_NUM
    #     offset_S = TRAINING_S_NUM + TESTING_S_NUM
    #     offset_V = TRAINING_V_NUM + TESTING_V_NUM
    #     offset_F = TRAINING_F_NUM + TESTING_F_NUM
    #     offset_Q = TRAINING_Q_NUM + TESTING_Q_NUM

    samples_N: pd.DataFrame = samples_dict[sample_type + '_N']
    samples_S: pd.DataFrame = samples_dict[sample_type + '_S']
    samples_V: pd.DataFrame = samples_dict[sample_type + '_V']
    samples_F: pd.DataFrame = samples_dict[sample_type + '_F']
    samples_Q: pd.DataFrame = samples_dict[sample_type + '_Q']

    # datas_N = []
    # datas_F = []
    # datas_V = []

    datas = []
    labels = []

    label = 0
    for index in range(batch_size):
        # label = np.random.randint(OUTPUT_NODE)
        label = np.random.randint(0, 5)
        if label == 0:
            data_json = samples_N['raw_data'].iloc[indices_N[index]]
        elif label == 1:
            data_json = samples_S['raw_data'].iloc[indices_S[index]]
        elif label == 2:
            data_json = samples_V['raw_data'].iloc[indices_V[index]]
        elif label == 3:
            data_json = samples_F['raw_data'].iloc[indices_F[index]]
        elif label == 4:
            data_json = samples_Q['raw_data'].iloc[indices_Q[index]]
        else:
            print('Error!')
        # label += 1
        # if label == 5:
        #     label = 0

        data_list = json.loads(data_json)
        if len(data_list) != 180:
            print("Error!")

        datas.append(data_list)
        label_set = np.zeros(OUTPUT_NODE, dtype=np.float32)
        label_set[label] = 1.0
        labels.append(label_set)

    return datas, labels


def trainning_model(input_values, n_classes, reuse=False):
    with tf.variable_scope("trainning", reuse=reuse):
        # Layer 0,1
        # input 180 -> 5filters -> 180x180x5 -> 3x3size,1stride -> 178x178x5
        conv1 = tf.layers.conv1d(inputs=input_values, filters=5, kernel_size=3, strides=1, padding='valid',
                                 activation=tf.nn.relu, name='conv1')
        # input 178x178x5 -> 89x89x5
        avg_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='valid', name='pool1')

        # Layer 2,3
        # input 89x89x5 -> 10filters -> 89x89x10 -> 4x4,1 -> 86x86x10
        conv2 = tf.layers.conv1d(inputs=avg_pool_1, filters=10, kernel_size=4, strides=1, padding='valid',
                                 activation=tf.nn.relu, name='conv2')
        # input 86x86x10 -> 43x43x10
        avg_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='valid', name='pool2')

        # Layer 4,5
        # input 43x43x10 -> 20filters -> 43x43x20 -> 4x4,1 -> 40x40x20
        conv3 = tf.layers.conv1d(inputs=avg_pool_2, filters=20, kernel_size=4, strides=1, padding='valid',
                                 activation=tf.nn.relu, name='conv3')
        # input 40x40x20 -> 20x20x20
        avg_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='valid', name='pool3')

        # 20x20x20 -> reshape
        pool_shape = avg_pool_3.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2]  # * pool_shape[3]
        flat = tf.reshape(avg_pool_3, [pool_shape[0], nodes])

        # flat = tf.reshape(avg_pool_3, (-1, int(20 * 20)))

        # Full link
        dense1 = tf.layers.dense(inputs=flat, units=30, activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.2), name='dense2')

        logits = tf.layers.dense(inputs=dense2, units=n_classes, activation=None, name='logits')

    return logits


def gen_labels_file():
    labels_dic = {}
    labels_dic['N'] = 0
    labels_dic['S'] = 1
    labels_dic['V'] = 2
    labels_dic['F'] = 3
    labels_dic['Q'] = 4
    labels_json = json.dumps(labels_dic)
    with open("%s/Train-DB/labels.txt" % (os.getcwd()), 'wt') as f:
        f.write(labels_json)


def load_labels() -> dict:
    labels_dic = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

    return labels_dic

    # file_path = "%s/Train-DB/labels.txt" % (os.getcwd())
    # if os.path.exists(file_path):
    #     with open(file_path, 'rt') as f:
    #         datas = json.loads(f.read())
    #         return datas
    # else:
    #     labels_dic = {}
    #     labels_dic['N'] = 0
    #     labels_dic['S'] = 1
    #     labels_dic['V'] = 2
    #     labels_dic['F'] = 3
    #     labels_dic['Q'] = 4
    #     return labels_dic


g_labels_map_dic = load_labels()


def trainning(is_continue=False):
    input_values = tf.placeholder(tf.float32, [None, INPUT_NODE], name='input_values')
    output_values = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='output_values')

    x_ = tf.reshape(input_values, [BATCH_SIZE, INPUT_NODE, 1])
    x_p = tf.reshape(input_values, [1, INPUT_NODE, 1])

    logits = trainning_model(x_, OUTPUT_NODE)
    logits_p = trainning_model(x_p, OUTPUT_NODE, reuse=True)

    final_tensor = tf.nn.softmax(logits)
    pred_tensor = tf.nn.softmax(logits_p)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_values))

    # 这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNNING_RATE).minimize(cost)

    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(output_values, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    samples_dict = load_samples()

    # training_count = tf.get_variable("training_count", shape=[1], dtype=tf.int32,
    #                                  initializer=tf.constant_initializer(0, tf.int32))

    ck_path = os.getcwd() + "/TrainingModels/"
    can_restore = False
    store_path = os.getcwd() + "/TrainingModels/" + TRAIN_MODEL_NAME + ".ckpt"
    saver = tf.train.Saver()
    if is_continue and tf.train.checkpoint_exists(ck_path + "checkpoint"):
        ckpt = tf.train.get_checkpoint_state(ck_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver_file_path: str = ckpt.model_checkpoint_path

            if saver_file_path.count(TRAIN_MODEL_NAME) > 0 and os.path.exists(saver_file_path + ".meta"):
                can_restore = True
                saver = tf.train.Saver(filename=saver_file_path)
    # saver = tf.train.Saver()

    tf.add_to_collection("pred_tensor", pred_tensor)
    tf.add_to_collection("evaluation_step", evaluation_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_i = 0

        if can_restore:
            print("Find Model... Continue Training.")
            saver.restore(sess, saver_file_path)
            start_i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

            testing_samples, testing_labels = get_batch(samples_dict, BATCH_SIZE, sample_type='testing')
            testing_acc = sess.run(evaluation_step,
                                   feed_dict={input_values: testing_samples, output_values: testing_labels})

            print("Testing acc = %.2f%%" % (testing_acc * 100))
            # for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
            #     print(op.name, op.values())

        # print("Step...", sess.graph_def().get)
        print("Start Training...Testing each 500 steps, Saving each 1000 steps")
        for i in range(start_i + 1, TRAINNING_STEPS):
            training_samples, training_labels = get_batch(samples_dict, BATCH_SIZE, sample_type='training')
            sess.run(train_step, feed_dict={input_values: training_samples, output_values: training_labels})

            if i % 100 == 0 or i == start_i + 1:
                loss = cost.eval(feed_dict={input_values: training_samples, output_values: training_labels})
                print("Iteration %d/%d:loss %f" % (i, TRAINNING_STEPS, loss))

            if i % 500 == 0 or i == start_i + 1:
                # Testing
                testing_samples, testing_labels = get_batch(samples_dict, BATCH_SIZE, sample_type='testing')
                # testing_acc = sess.run(evaluation_step,
                #                        feed_dict={input_values: testing_samples, output_values: testing_labels})
                y_pred = logits.eval(feed_dict={input_values: testing_samples,
                                                output_values: testing_labels})  # ,keep_prob: 1.0})


                y_pred = np.argmax(y_pred, axis=1)
                y_true = np.argmax(testing_labels, axis=1)

                Conf_Mat = confusion_matrix(y_true, y_pred)  # 利用专用函数得到混淆矩阵
                Acc = np.mean(y_pred == y_true)
                Acc_N = Conf_Mat[0][0] / np.sum(Conf_Mat[0])
                Acc_S = Conf_Mat[1][1] / np.sum(Conf_Mat[1])
                Acc_V = Conf_Mat[2][2] / np.sum(Conf_Mat[2])
                Acc_F = Conf_Mat[3][3] / np.sum(Conf_Mat[3])
                Acc_Q = Conf_Mat[4][4] / np.sum(Conf_Mat[4])

                print('\nAccuracy=%.2f%%' % (Acc * 100))
                print('Accuracy_N=%.2f%%' % (Acc_N * 100))
                print('Accuracy_S=%.2f%%' % (Acc_S * 100))
                print('Accuracy_V=%.2f%%' % (Acc_V * 100))
                print('Accuracy_F=%.2f%%' % (Acc_F * 100))
                print('Accuracy_Q=%.2f%%' % (Acc_Q * 100))
                print('\nConfusion Matrix:\n')
                print(Conf_Mat)

                Se_N = Conf_Mat[0][0] / np.sum(Conf_Mat[:, 0])
                Se_S = Conf_Mat[1][1] / np.sum(Conf_Mat[:, 1])
                Se_V = Conf_Mat[2][2] / np.sum(Conf_Mat[:, 2])
                Se_F = Conf_Mat[3][3] / np.sum(Conf_Mat[:, 3])
                Se_Q = Conf_Mat[4][4] / np.sum(Conf_Mat[:, 4])
                print('Sensity_N=%.2f%%' % (Se_N * 100))
                print('Sensity_S=%.2f%%' % (Se_S * 100))
                print('Sensity_V=%.2f%%' % (Se_V * 100))
                print('Sensity_F=%.2f%%' % (Se_F * 100))
                print('Sensity_Q=%.2f%%' % (Se_Q * 100))
                print("======================================")

                # print("Testing acc = %.2f%%" % (testing_acc * 100))

            if (i % 1000 == 0 or i == TRAINNING_STEPS - 1) and i > 0:
                print("Save model to path:", store_path)
                saver.save(sess, store_path, global_step=i)

        # Validation
        validation_samples, validation_labels = get_batch(samples_dict, BATCH_SIZE, sample_type='validation')
        validation_acc = sess.run(evaluation_step,
                                  feed_dict={input_values: validation_samples, output_values: validation_labels})

        print("Validation acc = %.2f%%" % (validation_acc * 100))


def detect_ecg():
    pass
    # from tensorflow.python.client import device_lib as _device_lib
    # local_device_protos = _device_lib.list_local_devices()
    # return [x.name for x in local_device_protos if x.device_type == 'GPU']


def analysis_model():
    # input_values = tf.placeholder(tf.float32, [None, INPUT_NODE], name='input_values')
    # output_values = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='output_values')

    # x_ = tf.reshape(input_values, [1, INPUT_NODE, 1])

    # logits = trainning_model(x_, OUTPUT_NODE)
    # final_tensor = tf.nn.softmax(logits)

    saver_file_path = os.getcwd() + "/TrainingModels/" + TEST_MODEL_NAME + ".ckpt"

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            "/Users/languilin/Desktop/WorkSpacePrivate/GitHubGrayLand/ECG-Arrhythmia/TrainingModels/" + TEST_MODEL_NAME + ".ckpt.meta")
        saver.restore(sess, saver_file_path)

        graph = tf.get_default_graph()
        input_values = graph.get_operation_by_name("input_values").outputs[0]
        # x_ = tf.reshape(input_values, [1, INPUT_NODE, 1])
        pred_tensor = tf.get_collection("pred_tensor")[0]
        # logits = graph.get_operation_by_name("trainning/logits:0")
        # fake_values = np.arange(-0.5, 1.3, 0.01)

        df = pd.read_csv(os.getcwd() + '/Train-DB-Checked/N/datas-mean.csv')

        times = 10
        labels = load_labels()
        while times > 0:
            index = np.random.randint(95000, 100000)
            raw_data = df.iloc[index]['raw_data']
            datas = json.loads(raw_data)
            # x_ = tf.reshape(input_values, [1, INPUT_NODE, 1])
            # Pred
            # sess.run(final_tensor, feed_dict={input_values: [fake_values]})
            result = sess.run(pred_tensor, feed_dict={input_values: [datas]}).tolist()[0]
            max_index = result.index(max(result))
            print(result)
            print(max_index)
            for k, v in labels.items():
                if v == max_index:
                    print("Type is %s" % (k))
                    break
                    # sess.run(tf.argmax(final_tensor, 1))
                    # sess.run(tf.get_default_graph().get_tensor_by_name("training_count:0"))
                    # for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
                    #     print(op.name, op.values())
    pass

def validation(times=10, model_name=""):

    model_path = os.getcwd() + "/TrainingModels/" + model_name

    # if not tf.train.checkpoint_exists(model_path + "checkpoint"):
    #     print("Can't find checkpoint file in path:", model_path)
    #     return
    #
    # ckpt = tf.train.get_checkpoint_state(model_path)
    # if not ckpt or not ckpt.model_checkpoint_path:
    #     print("Can't find model_checkpoint_path")
    #     return

    samples_dict = load_samples()

    with tf.Session() as sess:
        # Validation

        saver = tf.train.import_meta_graph(model_path + ".meta")
        saver.restore(sess, model_path)

        graph = tf.get_default_graph()

        input_values = graph.get_tensor_by_name("input_values:0")
        output_values = graph.get_tensor_by_name("output_values:0")

        # final_tensor = graph.get_tensor_by_name("final_tensor:0")
        # final_tensor = graph.get_operation_by_name("training/final_tensor")

        # with tf.name_scope("evaluation"):
        #     correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(output_values, 1))
        #     evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        evaluation_step = tf.get_collection("evaluation_step")[0]
        for i in range(times):
            validation_samples, validation_labels = get_batch(samples_dict, BATCH_SIZE, sample_type='validation')
            validation_acc = sess.run(evaluation_step,
                                      feed_dict={input_values: validation_samples, output_values: validation_labels})

            print("Validation acc = %.2f%%" % (validation_acc * 100))
    pass

if __name__ == '__main__':

    while 1:
        # input_str = input("Please Input:\n"
        #                   "1. Training\n"
        #                   "2. validation\n"
        #                   "q/Q: Quit\n")

        input_str = "1"
        if input_str == '1':
            trainning(is_continue=True)
        elif input_str == '2':
            validation(model_name="ModelC.ckpt-6000")
        else:
            break

        break
    # analysis_model()
    pass
