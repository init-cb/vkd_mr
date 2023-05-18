import torch
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from VKD_model import VKD
from Utils.trainer import pytorch_model_run
from Utils.dataset_loading import get_data_loaders
from Utils.load_mri_data import split_train_val_test, getTargetWeights
from data_preprocessing.BERTtokenizer import McbertEmbedding
import json

with open('config.json', 'r', encoding='utf-8') as json_f:
    config = json.load(json_f)
    TARGET_CSV_PATH = config['target_CSV_PATH']


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(model)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# def get_data_paths(data_choose):
#     if data_choose == 1:
#         TXT = "Img"
#         IMG = "Img"
#         NF = "No Finding"
#         PO = "postoperatively"
#         TRAIN = "../train_mri_plaque.csv"
#         TEST = "../test_mri_plaque.csv"
#         VAL = "../val_mri_plaque.csv"
#
#         return TXT, IMG, NF, PO, TRAIN, TEST, VAL
#     else:
#         return '', '', '', '', '', '', '',


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用{}".format(device))

    latent_dim = 512
    bert_embed_size = 768
    # n_feature_maps = 1024
    n_feature_maps = 8
    feature_map_size = 49
    maxlen = 256
    batch_size = 8
    n_epochs = 50
    SEED = 10
    debug = 0
    embed_size = 200
    data_choose = 1
    target_CSV_PATH = TARGET_CSV_PATH

    bert = McbertEmbedding()
    # x1:IMG; x2:text; y:label
    print("读取数据中...")
    x1_train, x1_val, x1_test, x2_train, x2_val, x2_test, y1_train, y1_val, y1_test = split_train_val_test(
        target_CSV_PATH)
    # get_multimodal_data(
    # TRAIN, VAL,
    # TEST, IMG,
    # TXT, maxlen)
    class_weights = getTargetWeights(y1_train)
    model_name = 'vkd_MRI'
    train_loader, val_loader, test_loader = get_data_loaders(x1_train, x2_train, y1_train, x1_val, x2_val, y1_val,
                                                             x1_test, x2_test, y1_test, batch_size)

    print("读取数据完成")
    # 模型要改
    vkd_model = VKD(
        latent_dim,
        bert_embed_size,
        n_feature_maps,
        feature_map_size,
        maxlen,
        class_weights,
        turn_off_recognition_grad=False,
    )

    count_parameters(vkd_model)

    print('读取模型并训练...')
    # vkd_model.load_state_dict(torch.load(model_name), strict=False)
    vkd_model = pytorch_model_run(train_loader, val_loader, vkd_model, model_name, n_epochs=n_epochs,
                                  batch_size=batch_size)
