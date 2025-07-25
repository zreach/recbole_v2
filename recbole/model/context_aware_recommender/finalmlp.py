# -*- coding: utf-8 -*-
# @Time   : 2024/7/1
# @Author : RecBole Team
# @Email  : 
# @File   : finalmlp.py

r"""
FinalMLP
################################################
Reference:
    Mao Ye et al. "FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction" in AAAI 2023.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers


class FinalMLP(ContextRecommender):
    r"""FinalMLP is a context-based recommendation model. It is an enhanced two-stream MLP model 
    that uses feature selection mechanisms and interaction aggregation for CTR prediction.
    """

    def __init__(self, config, dataset):
        super(FinalMLP, self).__init__(config, dataset)

        # load parameters info
        self.mlp1_hidden_size = config["mlp1_hidden_size"]
        self.mlp1_dropout = config["mlp1_dropout"]
        self.mlp2_hidden_size = config["mlp2_hidden_size"]
        self.mlp2_dropout = config["mlp2_dropout"]
        self.use_fs = config["use_fs"]
        self.fs_hidden_units = config["fs_hidden_units"]
        self.fs1_context = config["fs1_context"]
        self.fs2_context = config["fs2_context"]
        self.num_heads = config["num_heads"]
        #TODO
        # 改fs_context
        # define layers
        feature_dim = self.embedding_size * self.num_feature_field
        
        # MLP layers
        self.mlp1 = MLPLayers(
            layers=[feature_dim] + self.mlp1_hidden_size,
            dropout=self.mlp1_dropout,
            activation='relu'
        )
        
        self.mlp2 = MLPLayers(
            layers=[feature_dim] + self.mlp2_hidden_size,
            dropout=self.mlp2_dropout,
            activation='relu'
        )

        # Feature selection module
        if self.use_fs:
            self.fs_module = FeatureSelection(
                feature_dim=feature_dim,
                embedding_dim=self.embedding_size,
                fs_hidden_units=self.fs_hidden_units,
                fs1_context_size=len(self.fs1_context),
                fs2_context_size=len(self.fs2_context)
            )

        # Interaction aggregation module
        self.fusion_module = InteractionAggregation(
            x_dim=self.mlp1_hidden_size[-1],
            y_dim=self.mlp2_hidden_size[-1],
            output_dim=1,
            num_heads=self.num_heads
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
        finalmlp_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        
        # Flatten embeddings
        flat_emb = finalmlp_all_embeddings.view(finalmlp_all_embeddings.shape[0], -1)
        
        # Feature selection
        if self.use_fs:
            feat1, feat2 = self.fs_module(flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        
        # Pass through MLPs
        mlp1_output = self.mlp1(feat1)
        mlp2_output = self.mlp2(feat2)
        
        # Interaction aggregation
        y_pred = self.fusion_module(mlp1_output, mlp2_output)
        
        return y_pred.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label.float())

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))


class FeatureSelection(nn.Module):
    def __init__(self, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context_size=0, fs2_context_size=0):
        super(FeatureSelection, self).__init__()
        
        # Context bias parameters when no context features are specified
        if fs1_context_size == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_bias = None
            
        if fs2_context_size == 0:
            self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs2_ctx_bias = None

        # Gate networks for feature selection
        fs1_input_dim = embedding_dim if fs1_context_size == 0 else embedding_dim * fs1_context_size
        fs2_input_dim = embedding_dim if fs2_context_size == 0 else embedding_dim * fs2_context_size
        
        self.fs1_gate = MLPLayers(
            layers=[fs1_input_dim] + fs_hidden_units + [feature_dim],
            dropout=0,
            activation='relu',
            output_activation='sigmoid'
        )
        
        self.fs2_gate = MLPLayers(
            layers=[fs2_input_dim] + fs_hidden_units + [feature_dim],
            dropout=0,
            activation='relu',
            output_activation='sigmoid'
        )

    def forward(self, flat_emb):
        batch_size = flat_emb.size(0)
        
        # Generate gates for feature selection
        if self.fs1_ctx_bias is not None:
            fs1_input = self.fs1_ctx_bias.repeat(batch_size, 1)
        else:
            # In practice, this would use context features from the input
            # For simplicity, we use the bias parameter
            fs1_input = torch.zeros(batch_size, flat_emb.size(1) // flat_emb.size(0), device=flat_emb.device)
            
        if self.fs2_ctx_bias is not None:
            fs2_input = self.fs2_ctx_bias.repeat(batch_size, 1)
        else:
            # In practice, this would use context features from the input
            # For simplicity, we use the bias parameter
            fs2_input = torch.zeros(batch_size, flat_emb.size(1) // flat_emb.size(0), device=flat_emb.device)

        gt1 = self.fs1_gate(fs1_input) * 2  # Scale gate values
        gt2 = self.fs2_gate(fs2_input) * 2
        
        # Apply gates to features
        feature1 = flat_emb * gt1
        feature2 = flat_emb * gt2
        
        return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        
        # Linear transformations
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        
        # Interaction parameters
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim))
        
        # Initialize parameters
        xavier_normal_(self.w_xy)

    def forward(self, x, y):
        # Linear combination
        output = self.w_x(x) + self.w_y(y)
        
        # Multi-head interaction
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        
        # Compute interaction through tensor operations
        xy = torch.matmul(
            torch.matmul(
                head_x.unsqueeze(2), 
                self.w_xy.view(self.num_heads, self.head_x_dim, -1)
            ).view(-1, self.num_heads, self.output_dim, self.head_y_dim),
            head_y.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum over heads and add to output
        output += xy.sum(dim=1)
        
        return output
