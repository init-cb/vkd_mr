import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# # 所以数据到底是怎么个格式？
# # 所以，这个数据的生成，肯定是在分割为train、val、和test之前
# def get_multimodal_data(TRAIN_file, VAL_file, TEST_file, IMG, TXT, LABELS, maxlen):
#     # 从处理好的数据文件中的读取dataframe，train、val、test分为三个文件
#     # IMG\TXT\LABELS 分别是数据文件中的三列
#     train_df = pd.read_pickle(TRAIN_file)
#     val_df = pd.read_pickle(VAL_file)
#     test_df = pd.read_pickle(TEST_file)
#
#     print("Preparing train data")
#     x1_train = train_df[TXT].astype(str).values
#     x2_train = train_df[IMG].values
#     y_train = train_df[LABELS].values
#
#     print("Preparing val data")
#     x1_val = val_df[TXT].astype(str).values
#     x2_val = val_df[IMG].values
#     y_val = val_df[LABELS].values
#
#     print("Preparing test data")
#     x1_test = test_df[TXT].astype(str).values
#     x2_test = test_df[IMG].values
#     y_test = test_df[LABELS].values
#
#     return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test


def getTargetWeights(y):
    y = y.values
    weights = np.zeros(y.shape[1])
    for c in range(y.shape[1]):
        # print(np.sum(y, axis=c))
        weights[c] = np.sum(y[:c])
    weights = weights / y.shape[0]
    for c in range(y.shape[1]):
        weights[c] = 1 - weights[c]
        weights[c] = weights[c] ** 2
    return weights


def split_train_val_test(filename):
    df = pd.read_csv(filename, encoding="gbk")
    print(df.head())

    # 读数据
    path_img = copy.deepcopy(df['PATH_img'].values)
    explain_text = copy.deepcopy(df['EXPLAIN'].values)
    labels = copy.deepcopy(df['LABEL'].values)

    y_label = list()

    classes_word = ['易损', '稳定']
    classes_word_i = ['Vulnerable', 'Stable']
    for idx in tqdm(range(len(labels))):
        for cls in classes_word:
            if labels[idx] == cls:
                new_y_i = np.zeros(len(classes_word))
                new_y_i[classes_word.index(cls)] = 1
                y_label.append(new_y_i)
            else:
                continue

    new_df = pd.DataFrame(columns=['IMG', 'TEXT', 'LABEL'])
    x1 = path_img
    x2 = explain_text
    y = pd.DataFrame(y_label, columns=np.array(classes_word_i))
    print(y)

    # new_df.to_pickle(filename[:-4] + 'new_df.pkl')

    # = new_df['IMG']
    # = new_df['TEXT']
    # = new_df['LABEL']
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, stratify=y)
    X1_val, X1_test, X2_val, X2_test, y_val, y_test = train_test_split(X1_test, X2_test, y_test, test_size=0.5)

    return X1_train, X1_test, X1_val, X2_train, X2_test, X2_val, y_train, y_val, y_test
