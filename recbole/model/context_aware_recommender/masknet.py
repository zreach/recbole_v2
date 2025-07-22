# -*- coding: utf-8 -*-
# @Time   : 2024/7/1
# @Author : RecBole Team
# @Email  : 
# @File   : masknet.py

r"""
MaskNet
################################################
Reference:
    Zhenqin Wu et al. "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask" in arXiv 2021.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender_my import ContextRecommender
from recbole.model.layers import MLPLayers


class MaskNet(ContextRecommender):
    r"""MaskNet is a context-based recommendation model. It introduces feature-wise multiplication to CTR ranking 
    models by instance-guided mask. The model learns to selectively emphasize or de-emphasize different feature 
    interactions for different instances.
    """

    def __init__(self, config, dataset):
        super(MaskNet, self).__init__(config, dataset)
        
        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.model_type = config["model_type"]
        self.parallel_num_blocks = config["parallel_num_blocks"]
        self.parallel_block_dim = config["parallel_block_dim"]
        self.reduction_ratio = config["reduction_ratio"]

        # define layers and loss
        # print(self.dropout_prob)
        if self.model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(
                input_dim=self.embedding_size * self.num_feature_field,
                output_dim=1,
                hidden_units=self.mlp_hidden_size,
                reduction_ratio=self.reduction_ratio,
                dropout_rates=self.dropout_prob
            )
        elif self.model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(
                input_dim=self.embedding_size * self.num_feature_field,
                output_dim=1,
                num_blocks=self.parallel_num_blocks,
                block_dim=self.parallel_block_dim,
                hidden_units=self.mlp_hidden_size,
                reduction_ratio=self.reduction_ratio,
                dropout_rates=self.dropout_prob
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
        masknet_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        
        # Flatten embeddings for mask network
        feature_emb = masknet_all_embeddings.view(masknet_all_embeddings.shape[0], -1)
        
        # Pass through mask network
        y_pred = self.mask_net(feature_emb, feature_emb)
        
        return y_pred.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        print(self.loss(output, label.float()).item())
        return self.loss(output, label.float())

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))


class SerialMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, hidden_units=[], reduction_ratio=1, dropout_rates=0):
        super(SerialMaskNet, self).__init__()
        print(dropout_rates)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList()
        
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(MaskBlock(
                input_dim=input_dim,
                hidden_dim=self.hidden_units[idx],
                output_dim=self.hidden_units[idx + 1],
                reduction_ratio=reduction_ratio,
                dropout_rate=dropout_rates[idx]
            ))
        
        if output_dim is not None:
            self.fc = nn.Linear(self.hidden_units[-1], output_dim)
        else:
            self.fc = None

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for mask_block in self.mask_blocks:
            v_out = mask_block(V_emb, v_out)
        
        if self.fc is not None:
            v_out = self.fc(v_out)
        
        return v_out


class ParallelMaskNet(nn.Module):
    def __init__(self, input_dim, output_dim=None, num_blocks=1, block_dim=64, 
                 hidden_units=[], reduction_ratio=1, dropout_rates=0):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = num_blocks
        
        self.mask_blocks = nn.ModuleList([
            MaskBlock(
                input_dim=input_dim,
                hidden_dim=input_dim,
                output_dim=block_dim,
                reduction_ratio=reduction_ratio,
                dropout_rate=dropout_rates
            ) for _ in range(num_blocks)
        ])

        self.dnn = MLPLayers(
            layers=[block_dim * num_blocks] + hidden_units + [output_dim] if output_dim else hidden_units,
            dropout=dropout_rates,
            activation='relu'
        )

    def forward(self, V_emb, V_hidden):
        block_out = []
        for mask_block in self.mask_blocks:
            block_out.append(mask_block(V_emb, V_hidden))
        
        concat_out = torch.cat(block_out, dim=-1)
        v_out = self.dnn(concat_out)
        
        return v_out


class MaskBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, reduction_ratio=1, dropout_rate=0):
        super(MaskBlock, self).__init__()
        
        # Mask layer for generating instance-guided masks
        self.mask_layer = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim)
        )
        
        # Hidden layer with layer normalization and dropout
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(nn.ReLU())
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        # Generate mask based on input embeddings
        V_mask = self.mask_layer(V_emb)
        
        # Apply mask through element-wise multiplication
        masked_hidden = V_mask * V_hidden
        
        # Pass through hidden layer
        v_out = self.hidden_layer(masked_hidden)
        
        return v_out
