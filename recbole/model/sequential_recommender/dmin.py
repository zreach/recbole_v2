# -*- coding: utf-8 -*-
# @Time   : 2025/9/29
# @Author : GitHub Copilot
# @Email  : copilot@github.com

r"""
DMIN
##############################################
Reference:
    Multi-Interest Network with Dynamic routing for Recommendation at Tmall

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from recbole.utils import ModelType, InputType, FeatureType
from recbole.model.layers import (
    FMEmbedding,
    MLPLayers,
    ContextSeqEmbLayer,
    SequenceAttLayer,
)
from recbole.model.abstract_recommender_my import SequentialRecommender


class DMIN(SequentialRecommender):
    """DMIN (Deep Multi-Interest Network) captures diverse user interests through multi-interest extraction
    and uses dynamic routing to aggregate multiple interests for final prediction.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DMIN, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.device = config["device"]
        self.num_interest = config.get("num_interest", 4)  # number of interests
        self.routing_iter = config.get("routing_iter", 3)  # dynamic routing iterations
        self.dropout_prob = config["dropout_prob"]
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.pooling_mode = config["pooling_mode"]

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        num_item_feature = sum(
            (
                1
                if dataset.field2type[field]
                not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT]
                or field in config["numerical_features"]
                else 0
            )
            for field in self.item_feat.interaction.keys()
        )
        num_user_feature = sum(
            (
                1
                if dataset.field2type[field]
                not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT]
                or field in config["numerical_features"]
                else 0
            )
            for field in self.user_feat.interaction.keys()
        )
        print(num_item_feature)
        print(self.embedding_size)
        self.item_feat_dim = num_item_feature * self.embedding_size
        self.user_feat_dim = num_user_feature * self.embedding_size

        # Multi-interest extraction layers
        self.multi_interest_extractor = MultiInterestExtractor(
            self.item_feat_dim, self.num_interest, self.routing_iter
        )
        
        # Interest aggregation layer
        self.interest_aggregator = InterestAggregator(
            self.item_feat_dim, self.num_interest
        )

        # Embedding layer
        self.embedding_layer = ContextSeqEmbLayer(
            dataset, self.embedding_size, self.pooling_mode, self.device
        )

        # DNN layers
        self.dnn_mlp_list = [
            self.item_feat_dim + self.user_feat_dim
        ] + self.mlp_hidden_size
        
        self.dnn_mlp_layers = MLPLayers(
            self.dnn_mlp_list, activation="ReLU", dropout=self.dropout_prob, bn=True
        )
        self.dnn_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)
        self.other_parameter_name = ["embedding_layer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item_seq, item_seq_len, next_items):
        max_length = item_seq.shape[1]
        
        # Concatenate item sequence with target item
        item_seq_next_item = torch.cat(
            (item_seq, next_items.unsqueeze(1)), dim=-1
        )
        
        # Get embeddings
        sparse_embedding, dense_embedding = self.embedding_layer(
            user, item_seq_next_item
        )
        
        # Process embeddings
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=-2)
            table_shape = feature_table[type].shape
            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(
                table_shape[:-2] + (feat_num * embedding_size,)
            )

        user_feat_list = feature_table["user"]
        item_feat_list, target_item_feat_emb = feature_table["item"].split(
            [max_length, 1], dim=1
        )
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # Extract multiple interests
        interests = self.multi_interest_extractor(item_feat_list, item_seq_len)
        
        # Aggregate interests based on target item
        aggregated_interest = self.interest_aggregator(
            interests, target_item_feat_emb
        )

        # Concatenate features for final prediction
        dmin_in = torch.cat([aggregated_interest, user_feat_list], dim=-1)
        
        # DNN prediction
        dmin_out = self.dnn_mlp_layers(dmin_in)
        preds = self.dnn_predict_layer(dmin_out)
        
        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        
        output = self.forward(user, item_seq, item_seq_len, next_items)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        
        scores = self.forward(user, item_seq, item_seq_len, next_items)
        return self.sigmoid(scores)


class MultiInterestExtractor(nn.Module):
    """Multi-Interest Extractor using capsule network with dynamic routing"""
    
    def __init__(self, input_dim, num_interest, routing_iter):
        super(MultiInterestExtractor, self).__init__()
        self.input_dim = input_dim
        self.num_interest = num_interest
        self.routing_iter = routing_iter
        
        # Capsule parameters
        self.W = nn.Parameter(torch.randn(input_dim, num_interest, input_dim))
        self.bilinear = nn.Bilinear(input_dim, input_dim, 1)
        
    def forward(self, item_seq, seq_len):
        batch_size, max_len, feat_dim = item_seq.shape
        
        # Create mask for valid sequence positions
        mask = torch.arange(max_len, device=item_seq.device).expand(
            batch_size, max_len
        ) < seq_len.unsqueeze(1)
        
        # Transform input through capsule weights
        # item_seq: [B, T, D] -> [B, T, K, D]
        u_hat = torch.einsum('btd,dkh->btkh', item_seq, self.W)
        
        # Dynamic routing
        b = torch.zeros(batch_size, max_len, self.num_interest, device=item_seq.device)
        
        for _ in range(self.routing_iter):
            # Routing weights with mask
            c = F.softmax(b, dim=-1)  # [B, T, K]
            c = c * mask.unsqueeze(-1).float()  # Apply sequence mask
            
            # Weighted sum to get interest capsules
            s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)  # [B, K, D]
            
            # Squash function
            s_norm = torch.norm(s, dim=-1, keepdim=True)
            v = (s_norm ** 2) / (1 + s_norm ** 2) * s / (s_norm + 1e-8)  # [B, K, D]
            
            # Update routing logits
            if _ < self.routing_iter - 1:
                agreement = torch.sum(u_hat * v.unsqueeze(1), dim=-1)  # [B, T, K]
                b = b + agreement
        
        return v  # [B, K, D] - K interests for each user


class InterestAggregator(nn.Module):
    """Aggregate multiple interests based on target item"""
    
    def __init__(self, feat_dim, num_interest):
        super(InterestAggregator, self).__init__()
        self.feat_dim = feat_dim
        self.num_interest = num_interest
        
        # Attention mechanism for interest aggregation
        self.attention_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )
        
    def forward(self, interests, target_item):
        """
        Args:
            interests: [B, K, D] - multiple interests
            target_item: [B, D] - target item embedding
        """
        batch_size, num_interest, feat_dim = interests.shape
        
        # Expand target item for each interest
        target_expanded = target_item.unsqueeze(1).expand(-1, num_interest, -1)  # [B, K, D]
        
        # Concatenate interests with target item
        concat_input = torch.cat([interests, target_expanded], dim=-1)  # [B, K, 2D]
        
        # Calculate attention weights
        attention_scores = self.attention_net(concat_input).squeeze(-1)  # [B, K]
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B, K, 1]
        
        # Weighted aggregation of interests
        aggregated_interest = torch.sum(
            attention_weights * interests, dim=1
        )  # [B, D]
        
        return aggregated_interest