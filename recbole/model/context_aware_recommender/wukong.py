# -*- coding: utf-8 -*-
# @Time   : 2024/7/2
# @Author : RecBole Team
# @Email  : 
# @File   : WuKong.py

r"""
WuKong
################################################
Reference:
    Zhang et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation" in arXiv 2024.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender_my import ContextRecommender
from recbole.model.layers import MLPLayers, BaseFactorizationMachine


class WuKong(ContextRecommender):
    r"""WuKong is a context-based recommendation model that implements factorization machines-based model
    with linear compression blocks and factorization machine blocks for large-scale recommendation.
    """

    def __init__(self, config, dataset):
        super(WuKong, self).__init__(config, dataset)
        
        # load parameters info
        self.num_layers = config["num_layers"]
        self.compression_dim = config["compression_dim"]
        self.mlp_hidden_sizes = config["mlp_hidden_sizes"]
        self.fmb_units = config["fmb_units"]
        self.fmb_dim = config["fmb_dim"]
        self.project_dim = config["project_dim"]
        # self.project_dim = config["embedding_size"]
        # self.fmb_dim = config["embedding_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.interaction_layers = nn.ModuleList([
            WuKongLayer(
                num_features=self.num_feature_field, 
                embedding_dim=self.embedding_size, 
                project_dim=self.project_dim, 
                fmb_units=self.fmb_units, 
                fmb_dim=self.fmb_dim, 
                compressed_dim=self.compression_dim,
                dropout_rate=self.dropout_prob
            ) for _ in range(self.num_layers)
        ])
        
        # print([self.num_feature_field * self.embedding_size] + self.mlp_hidden_sizes + [1])
        self.final_mlp = MLPLayers(
            layers=[self.num_feature_field * self.embedding_size] + self.mlp_hidden_sizes + [1],
            dropout=self.dropout_prob,
            activation='relu',
            bn=False,
            last_activation=None
        )
        
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        for name, submodule in self.named_modules():
            self._init_weights(name, submodule)

    def _init_weights(self, name, module):
        if name not in ['id2afeats', 'id2tfeats']:
            if isinstance(module, nn.Embedding):
                xavier_normal_(module.weight.data)
            elif isinstance(module, nn.Linear):
                xavier_normal_(module.weight.data)
                if module.bias is not None:
                    constant_(module.bias.data, 0)

    def forward(self, interaction):
        wukong_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        
        # Pass through WuKong layers
        feature_emb = wukong_all_embeddings
        for layer in self.interaction_layers:
            feature_emb = layer(feature_emb)
        
        # Final MLP prediction
        # print(feature_emb.shape)
        # 把 feature_emb 展平为 [batch_size, num_field * embed_dim]
        # print(feature_emb)
        feature_emb = feature_emb.view(feature_emb.size(0), -1) 
        # 这里预想的输入形状应该是 [batch_size, num_field * embed_dim]
        y_pred = self.final_mlp(feature_emb)
        # print(y_pred)
        return y_pred.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        # print(self.loss(output, label.float()))
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))


class FactorizationMachineBlock(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, project_dim=8):
        super(FactorizationMachineBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.num_features = num_features
        self.projection_matrix = nn.Parameter(torch.randn(self.num_features, self.project_dim))
    
    def forward(self, x):
        batch_size = x.size(0)
        x_fm = x.view(batch_size, self.num_features, self.embedding_dim)
        projected = torch.matmul(x_fm.transpose(1, 2), self.projection_matrix)
        fm_matrix = torch.matmul(x_fm, projected)
        return fm_matrix.view(batch_size, -1)


class FMB(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, fmb_units=[32,32], fmb_dim=40, project_dim=8):
        super(FMB, self).__init__()
        self.fm_block = FactorizationMachineBlock(num_features, embedding_dim, project_dim)
        self.layer_norm = nn.LayerNorm(num_features * project_dim)
        model_layers = [nn.Linear(num_features * project_dim, fmb_units[0]), nn.ReLU()]
        for i in range(1, len(fmb_units)):
            model_layers.append(nn.Linear(fmb_units[i-1], fmb_units[i]))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Linear(fmb_units[-1], fmb_dim))
        self.mlp = nn.Sequential(*model_layers)
    
    def forward(self, x):
        y = self.fm_block(x)
        y = self.layer_norm(y)
        y = self.mlp(y)
        y = F.relu(y)
        # print(y)
        return y


class LinearCompressionBlock(nn.Module):
    """ Linear Compression Block (LCB) """
    def __init__(self, num_features=14, embedding_dim=16, compressed_dim=8, dropout_rate=0.2):
        super(LinearCompressionBlock, self).__init__()
        self.linear = nn.Linear(num_features * embedding_dim, compressed_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        return self.dropout(self.linear(x.view(x.size(0), -1)))


class WuKongLayer(nn.Module):
    def __init__(self, num_features=14, embedding_dim=16, project_dim=4, fmb_units=[40,40,40], fmb_dim=40, compressed_dim=40, dropout_rate=0.2):
        super(WuKongLayer, self).__init__()
        self.fmb = FMB(num_features, embedding_dim, fmb_units, fmb_dim, project_dim)
        # self.fmb = BaseFactorizationMachine(reduce_sum=False)
        self.lcb = LinearCompressionBlock(num_features, embedding_dim, compressed_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(num_features * embedding_dim)
        self.transform = nn.Linear(fmb_dim + compressed_dim, num_features * embedding_dim)
    
    def forward(self, x):
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        concat_out = torch.cat([fmb_out, lcb_out], dim=1)
        concat_out = self.transform(concat_out)
        add_norm_out = self.layer_norm(concat_out + x.view(x.size(0), -1))
        # add_norm_out = concat_out + x.view(x.size(0), -1)
        return add_norm_out.view(x.size(0), x.size(1), x.size(2))
