import torch
import torchvision
import numpy as np
import cv2
import nibabel as nib
from torch.utils.data import Dataset

from Utils.load_data import get_multimodal_data, prepare_embeddings, getTokenEmbed, getTargetWeights
from data_preprocessing.BERTtokenizer import McbertEmbedding


def get_data_loaders(x1_train, x2_train, y1_train, x1_val, x2_val, y1_val, x1_test, x2_test, y1_test, batch_size):
    train_dataset = MultimodalDataset(x1_train, x2_train, y1_train)
    train_loader = torch.utils.data.dataloader.DataLoader(dataset=train_dataset,
                                                          batch_size=batch_size, drop_last=True, num_workers=0,
                                                          pin_memory=True,
                                                          sampler=train_dataset.sampler)
    val_dataset = MultimodalDataset(x1_val, x2_val, y1_val)
    val_loader = torch.utils.data.dataloader.DataLoader(dataset=val_dataset,
                                                        batch_size=batch_size, drop_last=True, num_workers=0,
                                                        pin_memory=True)  # , sampler = val_dataset.sampler)
    test_dataset = MultimodalDataset(x1_test, x2_test, y1_test)
    test_loader = torch.utils.data.dataloader.DataLoader(dataset=test_dataset,
                                                         batch_size=batch_size, drop_last=True, num_workers=0,
                                                         pin_memory=True)

    return train_loader, val_loader, test_loader


class MultimodalDataset(Dataset):
    def __init__(self, IMG, TEXT, y, proxy=0):
        self.T = TEXT
        self.X = IMG
        self.y = y
        self.proxy = proxy
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
        self.class_weights = self.get_class_weights()
        self.sample_weights = torch.DoubleTensor(self.get_sample_weights())
        self.bert = McbertEmbedding()
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(self.sample_weights, len(self.sample_weights))

    def get_sample_weights(self):
        y = self.y.values
        sample_weights = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            sample_weights[i] = np.sum(y[i, :] * self.class_weights) / np.sum(y, axis=1)[i]
        sample_weights[0] *= 1
        return sample_weights

    # 计算每个类别的权重，根据类别在数据集中的出现频率来确定。
    # 根据代码中的注释，y是一个二维的numpy数组，每一行对应一个样本，每一列对应一个类别。
    # y的值是0或1，表示样本是否属于该类别。例如，如果有三个类别，A、B和C，那么y的一行可能是[1, 0, 1]，表示该样本属于A和C类别，不属于B类别。
    def get_class_weights(self):
        y = self.y.values
        weights = np.zeros(y.shape[1])
        for c in range(y.shape[1]):
            weights[c] = np.sum(y[:c])
        weights = weights / y.shape[0]
        for c in range(y.shape[1]):
            if weights[c] != 0.0:
                weights[c] = 1 / weights[c]
        return weights

    def __len__(self):
        return self.T.shape[0]

    # 通过dataloader读入数据后，读入的数据是：
    # X：img的地址
    # T：文本的原文
    # 读入图像地址后，读取图像并转为tensor
    # 读入文本后，使用中文bert转为词嵌入，再转为tensor
    def __getitem__(self, idx):
        # img = torchvision.transforms.functional.to_tensor(cv2.imread(self.X[idx]))
        # text = torchvision.transforms.functional.to_tensor(cv2.imread(self.T[idx]))
        img_from_path = nib.load(self.X[idx])
        # img = torchvision.transforms.functional.to_tensor(img_from_path)
        img = torch.from_numpy(img_from_path.get_fdata())

        text_emb = self.bert.word_vector(self.T[idx])

        # print('')
        # print("img_tensor的 shape：{}".format(img.shape))
        # print("text_tensor 的 shape：{}".format(text_emb.shape))
        # print(self.y)
        # return text[0, :, :], img, self.y[idx]
        return text_emb, img, self.y.iloc[idx].values
