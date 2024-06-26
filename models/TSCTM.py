import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TopicDistQuant import TopicDistQuant
from models.TSC import TSC
from utils.metrics_logger import MetricsLogger


ACTIVATION_MAP = {
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "relu": F.relu,
    "relu6": F.relu6,
    "leakyrelu": F.leaky_relu,
    "elu": F.elu,
    "softplus": F.softplus,
}

INIT_MAP = {
    "xavier": nn.init.xavier_uniform_,
    "kaiming": nn.init.kaiming_uniform_
}

NORMALISATION_MAP = {
    "batch_norm": nn.BatchNorm1d,
    "layer_norm": nn.LayerNorm,
}


class TSCTM(nn.Module):
    def __init__(self, config, ml: MetricsLogger):
        super().__init__()

        self.config = config
        self.ml = ml

        hidden_dim = config.en1_units
        self.fc11 = nn.Linear(config.vocab_size, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, config.num_topic)

        self.mean_bn = NORMALISATION_MAP[self.config.normalisation](config.num_topic)
        self.mean_bn.weight.requires_grad = False

        self.decoder_bn = NORMALISATION_MAP[self.config.normalisation](config.vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.fcd1 = nn.Linear(config.num_topic, config.vocab_size, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                INIT_MAP[self.config.init](m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.activation = ACTIVATION_MAP[self.config.activation]

        self.topic_dist_quant = TopicDistQuant(config.num_topic, config.num_topic, commitment_cost=self.config.commitment_cost)
        self.contrast_loss = TSC(config.use_aug, temperature=config.temperature, weight_contrast=config.weight_contrast)

    def get_beta(self):
        return self.fcd1.weight.T

    def encode(self, inputs):
        e1 = self.activation(self.fc11(inputs))
        e1 = self.activation(self.fc12(e1))
        return self.mean_bn(self.fc21(e1))

    def decode(self, theta):
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def get_theta(self, inputs):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)
        return softmax_theta

    def forward(self, inputs):
        if self.config.use_aug:
            return self.forward_aug(inputs)
        else:
            return self.forward_noaug(inputs)

    def forward_noaug(self, inputs):
        bow = inputs['bow']

        theta = self.encode(bow)
        softmax_theta = F.softmax(theta, dim=1)

        quant_rst = self.topic_dist_quant(softmax_theta)

        recon = self.decode(quant_rst['quantized'])

        self.ml.log('quant.loss', quant_rst['loss'])
        loss = self.loss_function(recon, bow) + quant_rst['loss']

        features = torch.cat([F.normalize(theta, dim=1).unsqueeze(1)], dim=1)
        contrastive_loss = self.contrast_loss(features, quant_idx=quant_rst['encoding_indices'])
        loss += contrastive_loss

        self.ml.log('contrastive.loss', contrastive_loss)
        self.ml.log('loss', loss)

        return softmax_theta, loss, contrastive_loss

    def forward_aug(self, inputs):
        theta = self.encode(inputs[0])
        softmax_theta = F.softmax(theta, dim=1)

        quant_rst = self.topic_dist_quant(softmax_theta)

        recon = self.decode(quant_rst['quantized'])
        loss = self.loss_function(recon, inputs[0]) + quant_rst['loss']

        contrast_feature_list = list()
        for x in inputs[1:]:
            theta0 = self.encode(x)
            contrast_feature_list.append(theta0)

            softmax_theta0 = F.softmax(theta0, dim=1)
            quant_rst0 = self.topic_dist_quant(softmax_theta0)
            loss += quant_rst0['loss']
            loss += self.loss_function(self.decode(quant_rst0['quantized']), x)

        loss /= len(inputs)

        contrast_feature = torch.cat([F.normalize(f, dim=1).unsqueeze(1) for f in contrast_feature_list], dim=1)
        features = torch.cat([F.normalize(theta, dim=1).unsqueeze(1), contrast_feature], dim=1)

        contrastive_loss = self.contrast_loss(features, quant_idx=quant_rst['encoding_indices'], weight_same_quant=self.config.weight_same_quant)

        loss += contrastive_loss

        return softmax_theta, loss, contrastive_loss

    def loss_function(self, recon_x, x):
        loss = -(x * (recon_x).log()).sum(axis=1)
        loss = loss.mean()
        return loss
