# -*- coding: utf-8 -*-
# @Time   : 2025/10/6
# @Author : GitHub Copilot
# @Email  : copilot@github.com

r"""
Transformer_CB
##############################################
Reference:
    Ashish Vaswani et al. "Attention is All You Need" in NIPS 2017

基于DIN_CB架构的Transformer序列预测模型
"""

import torch
import torch.nn as nn
import pickle
import os
import math
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender_my import SequentialRecommender
from recbole.model.layers import MLPLayers, ContextSeqEmbLayer
from recbole.model.aggregator import Aggregator
from recbole.utils import InputType, FeatureType


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, max_len=100):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_key_padding_mask: [batch_size, seq_len] True表示padding位置
        """
        # 位置编码
        src = src * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        src = self.pos_encoding(src)
        src = src.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        src = self.dropout(src)
        
        # Transformer编码
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        return output


class Transformer4Rec(SequentialRecommender):
    """基于Transformer的序列推荐模型，支持音频特征和传统特征融合"""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Transformer4Rec, self).__init__(config, dataset)

        # 基础配置 - 添加默认值
        self.LABEL_FIELD = config.get("LABEL_FIELD", "rating")
        self.embedding_size = config.get("embedding_size", 64)
        self.mlp_hidden_size = config.get("mlp_hidden_size", [256, 128])
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.pooling_mode = config.get("pooling_mode", "mean")
        self.dropout_prob = config.get("dropout_prob", 0.1)

        # Transformer特定配置 - 添加默认值
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 2)
        self.dim_feedforward = config.get("dim_feedforward", 256)
        self.aggregation_method = config.get("aggregation_method", "last")  # 'last', 'mean', 'max', 'attention'

        # 音频和item特征配置 - 添加默认值
        self.use_item_id = not config.get('no_itemid', False)
        self.use_audio = config.get('use_audio', False)
        self.audio_fusion_method = config.get('audio_fusion_method', 'replace')

        # 验证配置合理性
        if not self.use_item_id and not self.use_audio:
            raise ValueError("At least one of use_item_id or use_audio must be True")

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()
        self.proj_method = config.get('proj_method', 'mlp')

        # 初始化音频aggregator - 添加默认值
        if self.use_audio:
            a_feature_path = config.get('a_feature_path', './audio_features.pkl')
            if not os.path.exists(a_feature_path):
                raise FileNotFoundError(f"Audio feature file not found: {a_feature_path}")
            
            with open(a_feature_path, 'rb') as fp:
                music_features_array = pickle.load(fp)
            
            self.a_aggregator = Aggregator(
                embedding_size=self.embedding_size,
                proj_method=config.get('proj_method', 'mlp'),
                token2id=self.token2id,
                feature_dict=music_features_array,
                config=config,
                layer=config.get('afeat_layer', -1),
                mlp_dropout=config.get('wav_dropout', 0.2),
                mlp_size_list=config.get('wav_mlp_sizes', [512]),
                n_clusters=config.get('n_clusters', 16),
                n_stage=config.get('n_stage', 2),
                n_users=self.n_users,
                n_items=self.n_items,
                token_field_names=getattr(self, 'token_field_names', None),
                token_field_offsets=getattr(self, 'token_field_offsets', None),
                token_embedding_table=getattr(self, 'token_embedding_table', None),
                USER_ID=self.USER_ID
            )
            
            self.audio_embedding_dim = self.embedding_size * self.a_aggregator.num_feature_filed

        # 计算item特征维度
        if self.use_item_id:
            # 添加数值特征的默认处理
            numerical_features = config.get("numerical_features", [])
            num_item_feature = sum(
                (
                    1
                    if dataset.field2type[field]
                    not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT]
                    or field in numerical_features
                    else 0
                )
                for field in self.item_feat.interaction.keys()
            )
            base_item_dim = num_item_feature * self.embedding_size
        else:
            num_item_feature = 0
            base_item_dim = 0

        # 计算最终的item特征维度
        if not self.use_item_id and self.use_audio:
            final_item_dim = self.audio_embedding_dim
        elif self.use_item_id and not self.use_audio:
            final_item_dim = base_item_dim
        elif self.use_item_id and self.use_audio:
            if self.audio_fusion_method == 'concat':
                final_item_dim = base_item_dim + self.audio_embedding_dim
            elif self.audio_fusion_method == 'add':
                final_item_dim = base_item_dim
            elif self.audio_fusion_method == 'mlp':
                final_item_dim = base_item_dim
                fusion_input_dim = base_item_dim + self.audio_embedding_dim
                self.audio_fusion_mlp = nn.Sequential(
                    nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(fusion_input_dim // 2, base_item_dim)
                )
            elif self.audio_fusion_method == 'replace':
                final_item_dim = self.audio_embedding_dim
            else:
                raise ValueError(f"Unknown audio_fusion_method: {self.audio_fusion_method}")
        
        self.final_item_dim = final_item_dim

        # 确保特征维度与embedding_size兼容
        if self.final_item_dim != self.embedding_size:
            self.feature_projection = nn.Linear(self.final_item_dim, self.embedding_size)
        else:
            self.feature_projection = None

        # 初始化Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            d_model=self.embedding_size,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_prob,
            max_len=getattr(self, 'max_seq_length', 50)  # 添加默认最大序列长度
        )

        # 序列聚合方法
        if self.aggregation_method == "attention":
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=self.embedding_size,
                num_heads=self.nhead,
                dropout=self.dropout_prob,
                batch_first=True
            )
            # 可学习的查询向量
            self.query_vector = nn.Parameter(torch.randn(1, 1, self.embedding_size))

        # 预测层
        self.dnn_input_dim = 3 * self.embedding_size  # user_emb + target_emb + interaction
        self.dnn_list = [self.dnn_input_dim] + self.mlp_hidden_size
        
        self.dnn_mlp_layers = MLPLayers(
            self.dnn_list, activation="ReLU", dropout=self.dropout_prob, bn=True
        )
        
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)

        # 只有在使用item_id时才初始化embedding_layer
        if self.use_item_id:
            self.embedding_layer = ContextSeqEmbLayer(
                dataset, self.embedding_size, self.pooling_mode, self.device
            )

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # 参数初始化
        self.apply(self._init_weights)
        self.other_parameter_name = []
        if self.use_item_id:
            self.other_parameter_name.append("embedding_layer")
        if self.use_audio:
            self.other_parameter_name.append("a_aggregator")

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_audio_embeddings(self, interaction):
        """获取音频embedding"""
        if not self.use_audio:
            return None
        
        audio_embeddings = self.a_aggregator(interaction)
        batch_size = audio_embeddings.size(0)
        audio_embeddings_flat = audio_embeddings.view(batch_size, -1)
        
        return audio_embeddings_flat

    def get_item_representations(self, user, item_seq):
        """获取item序列的表示"""
        max_length = item_seq.shape[1]
        batch_size = item_seq.shape[0]

        # 传统特征
        if self.use_item_id:
            sparse_embedding, dense_embedding = self.embedding_layer(user, item_seq)
            feature_table = {}
            for type in self.types:
                feature_table[type] = []
                if sparse_embedding[type] is not None:
                    feature_table[type].append(sparse_embedding[type])
                if dense_embedding[type] is not None:
                    feature_table[type].append(dense_embedding[type])
                if len(feature_table[type]) > 0:
                    feature_table[type] = torch.cat(feature_table[type], dim=-2)
                    table_shape = feature_table[type].shape
                    feat_num, embedding_size = table_shape[-2], table_shape[-1]
                    feature_table[type] = feature_table[type].view(
                        table_shape[:-2] + (feat_num * embedding_size,)
                    )
                else:
                    feature_table[type] = torch.zeros(
                        batch_size, max_length, 0, device=self.device
                    )
            traditional_item_features = feature_table["item"]
        else:
            traditional_item_features = None

        # 音频特征
        if self.use_audio:
            if self.proj_method == 'attention_new':
                audio_item_features = self.a_aggregator.get_sequence_embedding(item_seq)
            else:
                audio_item_features = []
                for t in range(max_length):
                    items_t = item_seq[:, t]
                    valid_mask = items_t != 0
                    if valid_mask.any():
                        valid_items = items_t[valid_mask]
                        audio_interaction = {'tracks_id': valid_items}
                        if hasattr(self.a_aggregator, 'user_id_field_idx') and self.a_aggregator.user_id_field_idx is not None:
                            audio_interaction['user_id'] = user[valid_mask]
                        audio_emb = self.get_audio_embeddings(audio_interaction)
                        audio_feat_t = torch.zeros(batch_size, self.audio_embedding_dim, device=self.device)
                        audio_feat_t[valid_mask] = audio_emb
                    else:
                        audio_feat_t = torch.zeros(batch_size, self.audio_embedding_dim, device=self.device)
                    audio_item_features.append(audio_feat_t.unsqueeze(1))
                audio_item_features = torch.cat(audio_item_features, dim=1)
        else:
            audio_item_features = None

        # 特征融合
        if not self.use_item_id and self.use_audio:
            final_item_features = audio_item_features
        elif self.use_item_id and not self.use_audio:
            final_item_features = traditional_item_features
        elif self.use_item_id and self.use_audio:
            if self.audio_fusion_method == 'concat':
                final_item_features = torch.cat([traditional_item_features, audio_item_features], dim=-1)
            elif self.audio_fusion_method == 'add':
                if traditional_item_features.size(-1) == audio_item_features.size(-1):
                    final_item_features = traditional_item_features + audio_item_features
                else:
                    if not hasattr(self, 'audio_proj'):
                        self.audio_proj = nn.Linear(
                            audio_item_features.size(-1), traditional_item_features.size(-1)
                        ).to(self.device)
                    audio_proj = self.audio_proj(audio_item_features)
                    final_item_features = traditional_item_features + audio_proj
            elif self.audio_fusion_method == 'mlp':
                combined = torch.cat([traditional_item_features, audio_item_features], dim=-1)
                batch_size, seq_len, combined_dim = combined.shape
                combined_flat = combined.view(-1, combined_dim)
                fused_flat = self.audio_fusion_mlp(combined_flat)
                final_item_features = fused_flat.view(batch_size, seq_len, -1)
            elif self.audio_fusion_method == 'replace':
                final_item_features = audio_item_features

        # 投影到embedding_size维度
        if self.feature_projection is not None:
            final_item_features = self.feature_projection(final_item_features)

        return final_item_features

    def aggregate_sequence(self, sequence_output, item_seq_len):
        """聚合序列表示为用户表示"""
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        if self.aggregation_method == "last":
            # 使用最后一个有效位置
            user_emb = []
            for i in range(batch_size):
                last_idx = min(item_seq_len[i].item() - 1, seq_len - 1)
                last_idx = max(last_idx, 0)
                user_emb.append(sequence_output[i, last_idx, :])
            user_emb = torch.stack(user_emb)
            
        elif self.aggregation_method == "mean":
            # 平均池化（考虑padding）
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(self.device)
            mask = mask < item_seq_len.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            user_emb = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1)
            
        elif self.aggregation_method == "max":
            # 最大池化
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(self.device)
            mask = mask < item_seq_len.unsqueeze(1)
            sequence_output_masked = sequence_output.clone()
            sequence_output_masked[~mask] = float('-inf')
            user_emb = sequence_output_masked.max(dim=1)[0]
            
        elif self.aggregation_method == "attention":
            # 注意力池化
            query = self.query_vector.expand(batch_size, -1, -1)
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(self.device)
            mask = mask >= item_seq_len.unsqueeze(1)  # True for padding positions
            
            user_emb, _ = self.attention_pooling(
                query, sequence_output, sequence_output, 
                key_padding_mask=mask
            )
            user_emb = user_emb.squeeze(1)
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
        return user_emb

    def forward(self, user, item_seq, item_seq_len, next_items):
        batch_size = item_seq.shape[0]
        
        # 获取序列item表示
        item_seq_features = self.get_item_representations(user, item_seq)
        
        # 创建padding mask
        seq_len = item_seq.shape[1]
        padding_mask = torch.arange(seq_len).expand(batch_size, seq_len).to(self.device)
        padding_mask = padding_mask >= item_seq_len.unsqueeze(1)  # True for padding positions
        
        # Transformer编码
        sequence_output = self.transformer_encoder(item_seq_features, padding_mask)
        
        # 聚合序列表示
        user_emb = self.aggregate_sequence(sequence_output, item_seq_len)
        
        # 获取目标item表示
        next_items_expanded = next_items.unsqueeze(1)
        target_item_features = self.get_item_representations(user, next_items_expanded)
        target_item_emb = target_item_features.squeeze(1)
        
        # DNN预测
        din_in = torch.cat([user_emb, target_item_emb, user_emb * target_item_emb], dim=-1)
        din_out = self.dnn_mlp_layers(din_in)
        preds = self.dnn_predict_layers(din_out)
        
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
        scores = self.sigmoid(self.forward(user, item_seq, item_seq_len, next_items))
        return scores