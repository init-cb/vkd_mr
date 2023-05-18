from Utils.attention_modules import (Attention, TransformerEncoderLayer, BertPooler, PositionalEncoding)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as visionmodels
import numpy as np
from Model import resnet
import json

with open('config.json', 'r', encoding='utf-8') as json_f:
    config = json.load(json_f)
    PRETRAIN_PATH = config['pretrain_resnet_Path']


# 生成预训练模型
def generate_model(model_type='resnet', model_depth=50,
                   input_W=180, input_H=200, input_D=100, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path='',
                   nb_class=1):
    assert model_type in [
        'resnet'
    ]

    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 256
    elif model_depth == 18:
        model = resnet.resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 34:
        model = resnet.resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 50:
        model = resnet.resnet50(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 101:
        model = resnet.resnet101(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 152:
        model = resnet.resnet152(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 200:
        model = resnet.resnet200(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048

    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=nb_class, bias=True))

    if not no_cuda:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # k 是每一层的名称，v是权重数值
    net_dict.update(pretrain_dict)  # 字典 dict2 的键/值对更新到 dict 里。
    model.load_state_dict(net_dict)  # model.load_state_dict()函数把加载的权重复制到模型的权重中去

    print("-------- pre-train model load successfully --------")

    return model


class VKD(nn.Module):
    def __init__(self,
                 latent_dim,
                 bert_embed_size,
                 n_feature_maps,
                 feature_map_size,
                 max_num_words,
                 class_weights=None,
                 dropout_rate=0.5,
                 num_transformers=1,
                 turn_off_recognition_grad=True,
                 **kwargs):
        super(VKD, self).__init__()

        # 初始化参数
        # acivation functions
        self.leakyRelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.MSEloss = nn.MSELoss()
        self.BCEloss = nn.BCEWithLogitsLoss()

        self.latent_dim = latent_dim
        self.token_dim = bert_embed_size
        self.n_feature_maps = n_feature_maps
        self.feature_map_size = feature_map_size
        self.max_num_words = max_num_words
        if class_weights.any() != None:
            self.BCElossde = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weights))
        else:
            self.BCElossde = nn.BCEWithLogitsLoss()

        # self.densenet121 = visionmodels.densenet121(pretrained=True)

        self.resNet = generate_model(model_type='resnet', model_depth=50,
                                     input_W=180, input_H=200, input_D=100, resnet_shortcut='B',
                                     no_cuda=False, gpu_id=[0],
                                     pretrain_path=PRETRAIN_PATH,
                                     nb_class=1)
        # if torch.cuda.is_available():
        #     self.resNet = torch.nn.DataParallel(self.resNet).cuda()

        # # 视觉模型
        # modules = list(self.densenet121.children())[:-1]  # delete the last fc layer.
        # self.densenet121 = nn.Sequential(*modules)

        print(self.resNet)
        # 去掉Dataparallel层
        self.resNet = self.resNet.module
        print(self.resNet)
        # modules = list(self.resNet)
        # self.resNet = nn.Sequential(*modules)

        self.resNetLinear = nn.Sequential(
            nn.Linear(self.feature_map_size, 48),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))

        # 文本
        self.textpool = BertPooler(self.token_dim)
        self.pe_t = PositionalEncoding(self.token_dim, max_len=self.max_num_words)
        self.transformer_t = TransformerEncoderLayer(d_model=self.token_dim)
        # AdaptiveMaxPool是PyTorch中提供的自适应池化层。
        # 其主要特殊的地方在于：
        # 无论输入Input的size是多少，输出的size总为指定的size。
        self.pool = nn.AdaptiveMaxPool1d(1)

        # VAE
        self.fc_mu = nn.Sequential(
            nn.Linear(self.max_num_words, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))
        self.fc_var = nn.Sequential(
            nn.Linear(self.max_num_words, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))
        self.fcp_mu = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))

        self.fcp_var = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate))

        modules = []

        dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
        in_channels = self.latent_dim

        for d_dim in dense_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU())
            )
            in_channels = d_dim
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, 2)
            ))
        # 在Python中，*符号被用于解包一个列表或元组。在这个例子中，*modules被用于解包一个包含多个神经网络层的列表。
        # 这个解包操作允许我们将多个层组合成一个神经网络模型。
        self.generation = nn.Sequential(*modules)
        self.latent_dim += self.n_feature_maps
        modules_p = []

        dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
        in_channels = self.latent_dim
        for d_dim in dense_dims:
            modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU())
            )
            in_channels = d_dim
        modules_p.append(
            nn.Sequential(
                nn.Linear(in_channels, 2)
            ))
        self.generation_p = nn.Sequential(*modules_p)
        recognition_network_layers = [self.fc_mu, self.fc_var, self.generation_p]
        if turn_off_recognition_grad:
            for layer in recognition_network_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, img_input, text_input):
        print('')
        print("VKD_FORWARD_img_input 的 shape：{}".format(img_input.shape))
        print("VKD_FORWARD_text_input 的 shape：{}".format(text_input.shape))
        # img features
        img_input = torch.unsqueeze(img_input, 1)
        img_features = self.resNet(img_input)
        # img_features = self.resNetLinear(img_features.view(-1, 1024, 49))

        # text features
        text_features = self.transformer_t(self.pe_t(text_input))

        print("img_features 的 shape：{}".format(img_features))
        print("img_features 的 shape：{}".format(img_features.shape))
        out_p = torch.squeeze(self.pool(img_features))
        out_r = torch.squeeze(self.pool(text_features))

        print("out_p.shape:{}".format(out_p.shape))
        print("out_p.shape:{}".format(out_p))

        mu_p = self.fcp_mu(out_p)
        logvar_p = self.fcp_var(out_p)
        z_p = self.reparameterize(mu_p, logvar_p)

        mu_r = self.fc_mu(out_r)
        logvar_r = self.fc_var(out_r)
        z_r = self.reparameterize(mu_r, logvar_r)

        print("z_p.shape = {}".format(z_p.shape))
        print("out_p.shape = {}".format(out_p.shape))
        y_pred = self.generation_p(torch.cat([z_p, out_p], 0))
        mu = mu_p
        logvar = logvar_p
        y_predr = self.generation(z_r)
        mur = mu_r
        logvarr = logvar_r

        return y_pred, mu, logvar, y_predr, mur, logvarr

    def testing(self, img_input):

        # img features
        img_features = self.resNet(img_input)
        img_features = self.resNetLinear(img_features.view(-1, 1024, 49))

        out_p = torch.squeeze(self.pool(img_features))

        mu_p = self.fcp_mu(out_p)
        logvar_p = self.fcp_var(out_p)
        z_p = self.reparameterize(mu_p, logvar_p)

        return self.generation_p(torch.cat([z_p, out_p], 1)), mu_p, logvar_p

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)

        std = torch.exp(0.5 * logvar)
        for i in range(100):
            eps = torch.randn_like(std)
            z += eps * std + mu
        return z / 100

    def reparameterize_single(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def kl_loss_single(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    def kl_loss_multi(self, mu_p, logvar_p, mu_r, logvar_r):
        p = torch.distributions.Normal(mu_p, logvar_p)
        r = torch.distributions.Normal(mu_r, logvar_r)
        return torch.distributions.kl_divergence(p, r).mean()

    def loss_function(self,
                      *args,
                      **kwargs):

        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # 输入：y_pred, y_batch, mu, logvar, y_predr, mur, logvarr,kl_ann_factor[2]
        epsilon = 1e-8
        recon_weight = 1
        recons = args[0]  # y_pred
        label = args[1]  # y_batch
        mu = args[2] + epsilon
        logvar = args[3] + epsilon
        recons_r = args[4]  # y_predr
        mu_r = args[5] + epsilon
        logvar_r = args[6] + epsilon
        annealing_factor = args[7]

        recons_loss = self.BCEloss(recons, label)
        recons_loss_r = self.BCEloss(recons_r, label)

        kld_loss = self.kl_loss_multi(mu, logvar, mu_r, logvar_r)
        kld_weight = 1e-3

        loss = recons_loss * recon_weight + kld_loss * kld_weight * annealing_factor + recons_loss_r * recon_weight
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'Recons_r': recons_loss_r, }
