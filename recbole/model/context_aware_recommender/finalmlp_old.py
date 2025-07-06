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

class FinalMLP(ContextRecommender):

    def __init__(self, config, dataset):
        super(FinalMLP, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size_1 = config["mlp_hidden_size"]
        self.mlp_hidden_size_2 = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        print(self.mlp_hidden_size_1)
        self.in_feature_num = self.num_feature_field * self.embedding_size

        mlp_size_list_1 = [self.in_feature_num] + self.mlp_hidden_size_1 + [1]
        mlp_size_list_2 = [self.in_feature_num] + self.mlp_hidden_size_2 + [1]

        self.mlp1= MLPLayers(mlp_size_list_1, dropout=self.dropout_prob, bn=True)
        self.mlp2 = MLPLayers(mlp_size_list_2, dropout=self.dropout_prob, bn=True)
        # self.predict_layer = nn.Linear(self.mlp_hidden_size_1[-1] + self.mlp_hidden_size_2[-1], 1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
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
        embeddings = embeddings.view(batch_size, -1) # (batch_size, in_feature_num)
        
        output = self.mlp1(embeddings) + self.mlp2(embeddings)
        return output.squeeze(-1)
        

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
    
class FeatureSelection(nn.Module):
    def __init__(self, config, dataset):
        super(FeatureSelection, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.feature_dim = config["feature_dim"]
        self.fs_hidden_units = config.get("fs_hidden_units", [])
        self.fs1_context = config.get("fs1_context", [])
        self.fs2_context = config.get("fs2_context", [])
        self.dropout_prob = config.get("dropout_prob", 0.0)

        if len(self.fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, self.embedding_dim))
        else:
            self.fs1_ctx_emb = None

        if len(self.fs2_context) == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs2_context)
        self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)
        self.fs2_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs2_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)

    def forward(self, X, flat_emb):
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2
        return feature1, feature2