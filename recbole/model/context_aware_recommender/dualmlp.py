# -*- coding: utf-8 -*-
# @Time   : 2025/6/7
# @Author : Your Name
# @Email  : your.email@example.com
# @File   : dualmlp.py

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.layers import MLPLayers
from recbole.model.abstract_recommender_my import ContextRecommender

class DualMLP(ContextRecommender):
    """DualMLP: User and Item embeddings are processed by separate MLPs, then concatenated and passed through a final MLP."""

    def __init__(self, config, dataset):
        super(DualMLP, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size_1 = config["mlp1_hidden_size"]
        self.mlp_hidden_size_2 = config["mlp2_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        print(self.mlp_hidden_size_1)
        print(self.mlp_hidden_size_2)
        self.in_feature_num = self.num_feature_field * self.embedding_size

        mlp_size_list_1 = [self.in_feature_num] + self.mlp_hidden_size_1 + [1]
        mlp_size_list_2 = [self.in_feature_num] + self.mlp_hidden_size_2 + [1]

        self.mlp1= MLPLayers(mlp_size_list_1, dropout=self.dropout_prob, bn=False)
        self.mlp2 = MLPLayers(mlp_size_list_2, dropout=self.dropout_prob, bn=False)
        # self.predict_layer = nn.Linear(self.mlp_hidden_size_1[-1] + self.mlp_hidden_size_2[-1], 1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization``
        for name, submodule in self.named_modules():
            self._init_weights(name, submodule)

    def _init_weights(self, name, module):
        if name != 'id2feature':
            if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                xavier_normal_(module.weight.data)

    def forward(self, interaction):
        embeddings = self.concat_embed_input_fields(
            interaction
        )
        batch_size = embeddings.shape[0]
        # fm_output = self.first_order_linear(interaction)

        embeddings = embeddings.view(batch_size, -1) # (batch_size, in_feature_num)
        
        output = self.mlp1(embeddings) + self.mlp2(embeddings)  # (batch_size, 1)
        return output.squeeze(-1)
        

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label) + self.reg_emb_loss()

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))