# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2020/10/21
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
DIN
##############################################
Reference:
    Guorui Zhou et al. "Deep Interest Network for Click-Through Rate Prediction" in ACM SIGKDD 2018

Reference code:
    - https://github.com/zhougr1993/DeepInterestNetwork/tree/master/din
    - https://github.com/shenweichen/DeepCTR-Torch/tree/master/deepctr_torch/models

"""

import torch
import torch.nn as nn
import pickle
import os
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender_my import SequentialRecommender
from recbole.model.layers import MLPLayers, SequenceAttLayer, ContextSeqEmbLayer
from recbole.model.aggregator import Aggregator
from recbole.utils import InputType, FeatureType


class DIN_CB(SequentialRecommender):
    """Deep Interest Network utilizes the attention mechanism to get the weight of each user's behavior according
    to the target items, and finally gets the user representation.

    Note:
        In the official source code, unlike the paper, user features and context features are not input into DNN.
        We just migrated and changed the official source code.
        But You can get user features embedding from user_feat_list.
        Besides, in order to compare with other models, we use AUC instead of GAUC to evaluate the model.

    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DIN_CB, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.device = config["device"]
        self.pooling_mode = config["pooling_mode"]
        self.dropout_prob = config["dropout_prob"]

        # 新增：是否使用item_id和音频特征相关配置
        self.use_item_id = not config.get('no_itemid', True)
        self.use_audio = config.get('use_audio', False)
        self.audio_fusion_method = config.get('audio_fusion_method', 'replace')  # 'concat', 'add', 'mlp', 'replace'

        # 验证配置合理性
        if not self.use_item_id and not self.use_audio:
            raise ValueError("At least one of use_item_id or use_audio must be True")

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        # 新增：初始化音频aggregator
        if self.use_audio:
            # 加载音频特征
            a_feature_path = config['a_feature_path']
            if not os.path.exists(a_feature_path):
                raise FileNotFoundError(f"Audio feature file not found: {a_feature_path}")
            
            with open(a_feature_path, 'rb') as fp:
                music_features_array = pickle.load(fp)
            
            # 创建aggregator
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
                # 传入token相关信息（如果需要user_id）
                token_field_names=getattr(self, 'token_field_names', None),
                token_field_offsets=getattr(self, 'token_field_offsets', None),
                token_embedding_table=getattr(self, 'token_embedding_table', None),
                USER_ID=self.USER_ID
            )
            
            # 音频特征的输出维度
            self.audio_embedding_dim = self.embedding_size * self.a_aggregator.num_feature_filed

        # 计算item特征维度
        if self.use_item_id:
            # 计算传统item特征数量
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
            base_item_dim = num_item_feature * self.embedding_size
        else:
            # 不使用item_id时，基础维度为0
            num_item_feature = 0
            base_item_dim = 0

        # 计算最终的item特征维度
        if not self.use_item_id and self.use_audio:
            # 只使用音频特征
            final_item_dim = self.audio_embedding_dim
        elif self.use_item_id and not self.use_audio:
            # 只使用传统item特征
            final_item_dim = base_item_dim
        elif self.use_item_id and self.use_audio:
            # 两种特征都使用
            if self.audio_fusion_method == 'concat':
                final_item_dim = base_item_dim + self.audio_embedding_dim
            elif self.audio_fusion_method == 'add':
                final_item_dim = base_item_dim
            elif self.audio_fusion_method == 'mlp':
                final_item_dim = base_item_dim
                # 创建音频融合MLP
                fusion_input_dim = base_item_dim + self.audio_embedding_dim
                self.audio_fusion_mlp = nn.Sequential(
                    nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(fusion_input_dim // 2, base_item_dim)
                )
            elif self.audio_fusion_method == 'replace':
                # 音频特征替代传统特征
                final_item_dim = self.audio_embedding_dim
            else:
                raise ValueError(f"Unknown audio_fusion_method: {self.audio_fusion_method}")
        
        # 保存最终维度用于验证
        self.final_item_dim = final_item_dim

        # 计算DNN和attention的输入维度
        self.dnn_input_dim = 3 * final_item_dim
        self.att_input_dim = 4 * final_item_dim
        
        self.dnn_list = [self.dnn_input_dim] + self.mlp_hidden_size
        self.att_list = [self.att_input_dim] + self.mlp_hidden_size

        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )
        
        self.attention = SequenceAttLayer(
            mask_mat,
            self.att_list,
            activation="Sigmoid",
            softmax_stag=False,
            return_seq_weight=False,
        )
        
        self.dnn_mlp_layers = MLPLayers(
            self.dnn_list, activation="Dice", dropout=self.dropout_prob, bn=True
        )

        # 只有在使用item_id时才初始化embedding_layer
        if self.use_item_id:
            self.embedding_layer = ContextSeqEmbLayer(
                dataset, self.embedding_size, self.pooling_mode, self.device
            )
        
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
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
        
        # 使用aggregator获取音频embedding
        audio_embeddings = self.a_aggregator(interaction)  # [batch_size, num_audio_features, embedding_size]
        
        # 展平音频embedding
        batch_size = audio_embeddings.size(0)
        audio_embeddings_flat = audio_embeddings.view(batch_size, -1)  # [batch_size, num_audio_features * embedding_size]
        
        return audio_embeddings_flat

    def get_item_representations(self, user, item_seq_next_item):
        """优化版：分步处理序列，降低显存峰值"""
        max_length = item_seq_next_item.shape[1]
        batch_size = item_seq_next_item.shape[0]

        # 分离序列和目标item
        # item_seq = item_seq_next_item[:, :-1]  # [batch_size, seq_len]
        # target_items = item_seq_next_item[:, -1]  # [batch_size]
        
        # final_item_features = []

        # 传统特征
        if self.use_item_id:
            sparse_embedding, dense_embedding = self.embedding_layer(user, item_seq_next_item)
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

        # 音频特征（分步处理，节省显存）
        if self.use_audio:
            audio_item_features = []
            for t in range(max_length):
                items_t = item_seq_next_item[:, t]  # [batch_size]
                valid_mask = items_t != 0
                if valid_mask.any():
                    valid_items = items_t[valid_mask]
                    audio_interaction = {'tracks_id': valid_items}
                    if hasattr(self.a_aggregator, 'user_id_field_idx') and self.a_aggregator.user_id_field_idx is not None:
                        audio_interaction['user_id'] = user[valid_mask]
                    audio_emb = self.get_audio_embeddings(audio_interaction)  # [valid_num, audio_dim]
                    # 填充到batch_size
                    audio_feat_t = torch.zeros(batch_size, self.audio_embedding_dim, device=self.device)
                    audio_feat_t[valid_mask] = audio_emb
                else:
                    audio_feat_t = torch.zeros(batch_size, self.audio_embedding_dim, device=self.device)
                audio_item_features.append(audio_feat_t.unsqueeze(1))  # [batch_size, 1, audio_dim]
            audio_item_features = torch.cat(audio_item_features, dim=1)  # [batch_size, max_length, audio_dim]
        else:
            audio_item_features = None

        # 融合
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
            else:
                raise ValueError(f"Unknown audio_fusion_method: {self.audio_fusion_method}")
        return final_item_features

    def forward(self, user, item_seq, item_seq_len, next_items):
        max_length = item_seq.shape[1]
        batch_size = item_seq.shape[0]
        
        # 拼接历史序列和目标item
        item_seq_next_item = torch.cat((item_seq, next_items.unsqueeze(1)), dim=-1)
        
        # 获取所有item的表示
        all_item_features = self.get_item_representations(user, item_seq_next_item)
        
        # 分离序列特征和目标item特征
        item_feat_list, target_item_feat_emb = all_item_features.split([max_length, 1], dim=1)
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # 验证维度
        if target_item_feat_emb.size(-1) != self.final_item_dim:
            print(f"Warning: Dimension mismatch! Expected: {self.final_item_dim}, "
                  f"Got: {target_item_feat_emb.size(-1)}")

        # attention机制
        user_emb = self.attention(target_item_feat_emb, item_feat_list, item_seq_len)
        user_emb = user_emb.squeeze(1)

        # DNN预测
        din_in = torch.cat([user_emb, target_item_feat_emb, user_emb * target_item_feat_emb], dim=-1)
        
        # 验证DNN输入维度
        if din_in.size(-1) != self.dnn_input_dim:
            print(f"Warning: DNN input dimension mismatch! Expected: {self.dnn_input_dim}, "
                  f"Got: {din_in.size(-1)}")
        
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