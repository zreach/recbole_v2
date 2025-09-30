import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.cluster import KMeans
from recbole.model.layers import FMEmbedding, FMFirstOrderLinear, FLEmbedding, MLPLayers
from tqdm import tqdm 

class Aggregator(nn.Module):
    def __init__(self, embedding_size, token2id, feature_dict, config, proj_method="linear", layer=-1, n_clusters=2, n_stage=2, mlp_dropout=0.2, mlp_size_list=None, n_users=None, n_items=None, token_field_names=None, token_field_offsets=None, 
             token_embedding_table=None, USER_ID=None):
        super().__init__()
        self.layer = layer
        self.proj_method = proj_method
        self.embedding_size = embedding_size
        self.token2id = token2id
        self.feature_dict = feature_dict
        self.has_time = False
        feature_shape = list(feature_dict.values())[0].shape
        self.config = config
        self.n_users = n_users
        self.n_items = n_items

        self.token_field_names = token_field_names
        self.token_field_offsets = token_field_offsets
        self.token_embedding_table = token_embedding_table
        self.USER_ID = USER_ID
        
        if proj_method == 'pre_gate' and self.token_field_names is not None and self.USER_ID is not None:
            self.user_id_field_idx = None
            print(self.token_field_names)
            for i, field_name in enumerate(self.token_field_names):
                if field_name == self.USER_ID:
                    self.user_id_field_idx = i
                    break
            
            if self.user_id_field_idx is None:
                raise ValueError(f"USER_ID field {self.USER_ID} not found in token fields")
        # self.n_stage

        if len(feature_shape) == 1:
            H = feature_shape
            for k, v in feature_dict.items():
                feature_dict[k] = v.reshape(1, 1, -1)
            L = 1
            T = 1
        elif len(feature_shape) == 2:
            L, H = feature_shape
            for k, v in feature_dict.items():
                feature_dict[k] = v.reshape(L, 1, H)
            
            T = 1
        elif len(feature_shape) == 3:
            L, T, H = feature_shape
            self.has_time = True
        else:
            raise ValueError(f"Feature dimension not supported: {len(feature_shape)}")
        self.feature_size = H

        if layer == 'weighted_sum' or proj_method in ['transformer', 'moe', 'all', 'attention', 'attention_origin', 'attention_self', 'item_weight', 'rnn', 'attention_global', 'gate', 'pre_gate', 'pre_moe']:
            self.weights = nn.Parameter(torch.ones((L, 1)), requires_grad=True)

            # 保存所有层的信息，不进行聚合
            # feature_token = {}
            feature_token = np.zeros((len(self.token2id['tracks_id']), L, H))
            for k, v in self.token2id['tracks_id'].items():
                if k == '[PAD]':
                    feature = np.zeros((L, H))
                else:
                    feature = feature_dict[k]  # shape: [L, T, H] 或 [L, H]
                    
                    if len(feature.shape) == 3:  # [L, T, H]
                        # 先对时间维度取平均: [L, T, H] -> [L, H]
                        feature = np.mean(feature, axis=1)  # [L, H]
                feature_token[v] = feature
                # feature_token[v] = torch.tensor(feature, dtype=torch.float32)  # 保持 [L, H] 格式
            
            # 创建embedding保存所有层信息: [num_tracks, L, H]
            # all_features = torch.stack(list(feature_token.values()))  # [num_tracks, L, H]
            # self.id2feats = nn.Embedding.from_pretrained(feature_token.view(-1, L * H))  # 展平存储
            self.id2feats = nn.Embedding.from_pretrained(torch.from_numpy(feature_token).view(-1, L * H).float())
            self.id2feats.requires_grad_(False)
            
            
            self.feature_dim = H  # 保存特征维度
        
        else:
            # TODO 还有对时间的平均
            feature_token = torch.zeros((len(self.token2id['tracks_id']), H))
            for k, v in self.token2id['tracks_id'].items():
                if k == '[PAD]':
                    feature = np.zeros((H))
                else:
                    feature = feature_dict[k]
                    feature = np.mean(feature, axis=1, keepdims=False) # 对时间维度做平均
                    if layer == "mean":
                        feature = np.mean(feature, axis=0, keepdims=False)
                    else:
                        feature = feature[layer] # 取某层
                feature_token[v] = torch.Tensor(feature)
            # all_features = torch.stack(list(feature_token.values()))  # [num_tracks, H]
            self.id2feats = nn.Embedding.from_pretrained(feature_token)  
            self.id2feats.requires_grad_(False)
            
            self.feature_dim = H  # 保存特征维度

        # feature_token = torch.zeros((len(self.token2id['tracks_id']), H))
        self.L = L

        if proj_method in ['mlp', 'linear', 'transformer', 'moe', 'rnn']:
            self.num_feature_filed = 1
        elif proj_method in ['cluster']:
            self.num_feature_filed = self.L
        elif proj_method in ['rq-kmeans']:
            self.num_feature_filed = n_stage * self.L
        # 线性聚合参数
        if proj_method == 'linear':
            self.net = nn.Linear(H, embedding_size, bias=True)
        #     self.net = nn.Sequential(
        #     nn.Linear(H, embedding_size, bias=True),
        #     nn.Dropout(mlp_dropout)
        # )
        # MLP聚合参数
        elif proj_method == 'mlp':
            size_list = [
                H
            ] + mlp_size_list + [self.embedding_size]
            print(size_list)
            print(mlp_dropout)
            self.net = MLPLayers(size_list, mlp_dropout, last_activation=False)
        
        elif proj_method == 'pre_gate':
            # Pre-Gate方法：使用user_id学习L维权重，先加权特征再通过MLP
            
            # 直接使用现有的token_embedding_table中的user_id embedding
            # 不需要额外创建user_gate_embedding
            
            # 找到user_id在token_field_names中的位置
            self.user_id_field_idx = None
            for i, field_name in enumerate(self.token_field_names):
                if field_name == self.USER_ID:
                    self.user_id_field_idx = i
                    break
            
            if self.user_id_field_idx is None:
                raise ValueError(f"USER_ID field {self.USER_ID} not found in token fields")
            
            # 新增：全局可学习权重
            self.use_global_weights = config.get('use_global_weights', True)
            if self.use_global_weights:
                # 初始化全局L维权重参数
                print('Using global learnable weights for pre_gate')
                self.global_weights = nn.Parameter(torch.zeros(L))  # [L]
                # 可选：全局权重的初始化方式
                global_init_method = config.get('global_init_method', 'zeros')  # 'zeros', 'uniform', 'normal', 'ones'
                if global_init_method == 'uniform':
                    nn.init.uniform_(self.global_weights, -0.1, 0.1)
                elif global_init_method == 'normal':
                    nn.init.normal_(self.global_weights, 0.0, 0.1)
                elif global_init_method == 'ones':
                    nn.init.ones_(self.global_weights)
                elif global_init_method == 'zeros':
                    nn.init.zeros_(self.global_weights)
                else:
                    raise ValueError(f"Unknown global_init_method: {global_init_method}")
            
            # Gate网络：从user embedding映射到L维权重
            gate_hidden_size = config.get('gate_hidden_size', self.embedding_size // 2)
            
            if gate_hidden_size > 0:
                self.gate_network = nn.Sequential(
                    nn.Linear(self.embedding_size, gate_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(gate_hidden_size, L)
                )
            else:
                # 如果隐藏层大小为0，直接线性映射
                self.gate_network = nn.Linear(self.embedding_size, L)
            
            # 权重组合方式
            self.weight_combination = config.get('weight_combination', 'add')  # 'add', 'weighted_add', 'concat'
            
            if self.weight_combination == 'weighted_add':
                # 可学习的组合系数
                self.combination_alpha = nn.Parameter(torch.tensor(0.5))  # 用于控制全局权重和用户权重的比例
                # alpha * global_weights + (1-alpha) * user_weights
            elif self.weight_combination == 'concat':
                # 如果是拼接模式，需要额外的网络来处理拼接后的权重
                self.weight_fusion = nn.Sequential(
                    nn.Linear(2 * L, L),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(L, L)
                )
            
            # 权重归一化方式
            self.gate_norm_method = config.get('gate_norm_method', 'softmax')  # 'softmax', 'sigmoid', 'none'
            
            # Gate温度参数（用于softmax）
            self.gate_temperature = config.get('gate_temperature', 1.0)
            
            # Gate dropout
            self.gate_dropout = config.get('gate_dropout', 0.0)
            
            # 特征聚合方式
            self.feature_aggregation = config.get('feature_aggregation', 'weighted_sum')  # 'weighted_sum', 'weighted_concat'
            
            if self.feature_aggregation == 'weighted_sum':
                # 加权求和后的特征维度仍为H
                final_mlp_input_dim = H
            elif self.feature_aggregation == 'weighted_concat':
                # 将加权后的L个特征拼接
                final_mlp_input_dim = L * H
            else:
                raise ValueError(f"Unknown feature_aggregation: {self.feature_aggregation}")
            
            # 最终MLP：处理加权后的特征
            if mlp_size_list is None:
                self.final_mlp = nn.Sequential(
                    nn.Linear(final_mlp_input_dim, final_mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(final_mlp_input_dim // 2, embedding_size)
                )
            else:
                size_list = [final_mlp_input_dim] + mlp_size_list + [embedding_size]
                self.final_mlp = MLPLayers(size_list, mlp_dropout, last_activation=False)
            
            # 输出单个特征
            self.num_feature_filed = 1
        elif proj_method == 'pre_moe':
            # Pre-MoE方法：使用user_id学习L维权重，选择top1特征进入MLP
            
            # 直接使用现有的token_embedding_table中的user_id embedding
            # 找到user_id在token_field_names中的位置
            self.user_id_field_idx = None
            for i, field_name in enumerate(self.token_field_names):
                if field_name == self.USER_ID:
                    self.user_id_field_idx = i
                    break
            
            if self.user_id_field_idx is None:
                raise ValueError(f"USER_ID field {self.USER_ID} not found in token fields")
            
            # 新增：全局可学习权重（可选）
            self.use_global_weights = config.get('use_global_weights', True)
            if self.use_global_weights:
                # 初始化全局L维权重参数
                self.global_weights = nn.Parameter(torch.zeros(L))  # [L]
                # 全局权重的初始化方式
                global_init_method = config.get('global_init_method', 'zeros')  # 'zeros', 'uniform', 'normal', 'ones'
                if global_init_method == 'uniform':
                    nn.init.uniform_(self.global_weights, -0.1, 0.1)
                elif global_init_method == 'normal':
                    nn.init.normal_(self.global_weights, 0.0, 0.1)
                elif global_init_method == 'ones':
                    nn.init.ones_(self.global_weights)
                elif global_init_method == 'zeros':
                    nn.init.zeros_(self.global_weights)
                else:
                    raise ValueError(f"Unknown global_init_method: {global_init_method}")
            
            # Gate网络：从user embedding映射到L维权重
            gate_hidden_size = config.get('gate_hidden_size', self.embedding_size // 2)
            
            if gate_hidden_size > 0:
                self.gate_network = nn.Sequential(
                    nn.Linear(self.embedding_size, gate_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(gate_hidden_size, L)
                )
            else:
                # 如果隐藏层大小为0，直接线性映射
                self.gate_network = nn.Linear(self.embedding_size, L)
            
            # 权重组合方式（如果使用全局权重）
            self.weight_combination = config.get('weight_combination', 'add')  # 'add', 'weighted_add', 'concat'
            
            if self.weight_combination == 'weighted_add':
                # 可学习的组合系数
                self.combination_alpha = nn.Parameter(torch.tensor(0.5))
            elif self.weight_combination == 'concat':
                # 如果是拼接模式，需要额外的网络来处理拼接后的权重
                self.weight_fusion = nn.Sequential(
                    nn.Linear(2 * L, L),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(L, L)
                )
            
            # Gate温度参数（用于softmax）
            self.gate_temperature = config.get('gate_temperature', 1.0)
            
            # Gate dropout
            self.gate_dropout = config.get('gate_dropout', 0.0)
            
            # Top-k选择参数（对于pre_moe，固定为1）
            self.topk = 1  # 只选择top1
            
            # 是否使用Gumbel-Softmax进行可微分的离散选择
            self.use_gumbel = config.get('use_gumbel', False)
            self.gumbel_temperature = config.get('gumbel_temperature', 1.0)
            self.gumbel_hard = config.get('gumbel_hard', True)
            
            # 最终MLP：处理选中的单个特征
            if mlp_size_list is None:
                self.final_mlp = nn.Sequential(
                    nn.Linear(H, H // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(H // 2, embedding_size)
                )
            else:
                size_list = [H] + mlp_size_list + [embedding_size]
                self.final_mlp = MLPLayers(size_list, mlp_dropout, last_activation=False)
            
            # 输出单个特征
            self.num_feature_filed = 1
        elif proj_method == 'gate':
            # Gate方法：先得到L个embedding，再用item_feature计算gate分数聚合
            
            # Gate模式设置
            self.gate_mode = config.get('gate_mode', 'separate')  # 'separate' 或 'shared'
            
            if self.gate_mode == 'separate':
                # 为每一层创建独立的MLP
                self.layer_mlps = nn.ModuleList()
                for i in range(L):
                    if mlp_size_list is None:
                        layer_mlp = nn.Sequential(
                            nn.Linear(H, H // 2),
                            nn.ReLU(),
                            nn.Dropout(mlp_dropout),
                            nn.Linear(H // 2, embedding_size)
                        )
                    else:
                        size_list = [H] + mlp_size_list + [embedding_size]
                        layer_mlp = MLPLayers(size_list, mlp_dropout, last_activation=False)
                    self.layer_mlps.append(layer_mlp)
                    
            elif self.gate_mode == 'shared':
                # 所有层共享同一个MLP
                if mlp_size_list is None:
                    self.shared_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    self.shared_mlp = MLPLayers(size_list, mlp_dropout, last_activation=False)
                    
            else:
                raise ValueError(f"Unknown gate_mode: {self.gate_mode}")
            
            # Gate网络：使用item特征计算L个embedding的权重
            self.gate_input_type = config.get('gate_input_type', 'mean_feature')  # 'mean_feature', 'item_id', 'both'
            
            if self.gate_input_type == 'mean_feature':
                # 使用所有层特征的平均值作为gate输入
                gate_input_dim = H
            elif self.gate_input_type == 'item_id':
                # 使用item ID embedding作为gate输入
                self.item_gate_embedding_size = config.get('item_gate_embedding_size', embedding_size)
                self.item_gate_embedding = nn.Embedding(self.n_items, self.item_gate_embedding_size)
                gate_input_dim = self.item_gate_embedding_size
            elif self.gate_input_type == 'user_id':
                # 使用user ID embedding作为gate输入
                self.user_gate_embedding_size = config.get('user_gate_embedding_size', embedding_size)
                self.user_gate_embedding = nn.Embedding(self.n_users, self.user_gate_embedding_size)
                gate_input_dim = self.user_gate_embedding_size
            elif self.gate_input_type == 'both':
                # 使用特征平均值和item ID embedding的拼接
                self.item_gate_embedding_size = config.get('item_gate_embedding_size', embedding_size)
                self.item_gate_embedding = nn.Embedding(self.n_items, self.item_gate_embedding_size)
                gate_input_dim = H + self.item_gate_embedding_size
            elif self.gate_input_type == 'user_feature':
                # 使用特征平均值和user ID embedding的拼接
                self.user_gate_embedding_size = config.get('user_gate_embedding_size', embedding_size)
                self.user_gate_embedding = nn.Embedding(self.n_users, self.user_gate_embedding_size)
                gate_input_dim = H + self.user_gate_embedding_size
            elif self.gate_input_type == 'both_id':
                # 使用user ID embedding和item ID embedding的拼接
                self.user_gate_embedding_size = config.get('user_gate_embedding_size', embedding_size)
                self.user_gate_embedding = nn.Embedding(self.n_users, self.user_gate_embedding_size)
                self.item_gate_embedding_size = config.get('item_gate_embedding_size', embedding_size)
                self.item_gate_embedding = nn.Embedding(self.n_items, self.item_gate_embedding_size)
                gate_input_dim = self.user_gate_embedding_size + self.item_gate_embedding_size
            elif self.gate_input_type == 'all':
                # 使用特征平均值、user ID embedding和item ID embedding的拼接
                self.user_gate_embedding_size = config.get('user_gate_embedding_size', embedding_size)
                self.user_gate_embedding = nn.Embedding(self.n_users, self.user_gate_embedding_size)
                self.item_gate_embedding_size = config.get('item_gate_embedding_size', embedding_size)
                self.item_gate_embedding = nn.Embedding(self.n_items, self.item_gate_embedding_size)
                gate_input_dim = H + self.user_gate_embedding_size + self.item_gate_embedding_size
            else:
                raise ValueError(f"Unknown gate_input_type: {self.gate_input_type}")
            
            # Gate网络
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, gate_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(gate_input_dim // 2, L),
                nn.Softmax(dim=-1)
            )
            
            # 可选的门控参数
            self.gate_temperature = config.get('gate_temperature', 1.0)  # 温度参数
            self.gate_dropout = config.get('gate_dropout', 0.0)  # gate dropout
            
            # 输出模式：只支持聚合为1个embedding
            self.num_feature_filed = 1
            
            # 可选：最终融合层
            self.use_final_fusion = config.get('use_final_fusion', False)
            if self.use_final_fusion:
                self.final_fusion = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(embedding_size, embedding_size)
                )
            else:
                self.final_fusion = nn.Identity()
                
        elif proj_method == 'transformer':
            from torch.nn import MultiheadAttention, LayerNorm

            self.transformer_layers = config['transformer_layers'] if 'transformer_layers' in config else 2
            self.transformer_heads = config['transformer_heads'] if 'transformer_heads' in config else 8
            self.transformer_dropout = config['transformer_dropout'] if 'transformer_dropout' in config else 0.1

            assert H % self.transformer_heads == 0, f"Hidden size {H} must be divisible by number of heads {self.transformer_heads}"

            self.transformer_encoder_layers = nn.ModuleList()
            for _ in range(self.transformer_layers):
                # 多头注意力层
                attention_layer = MultiheadAttention(
                    embed_dim=H,
                    num_heads=self.transformer_heads,
                    dropout=self.transformer_dropout,
                    batch_first=True
                )
                # 层归一化
                norm1 = LayerNorm(H)
                norm2 = LayerNorm(H)
                # 前馈网络
                ffn = nn.Sequential(
                    nn.Linear(H, H * 4),
                    nn.ReLU(),
                    nn.Dropout(self.transformer_dropout),
                    nn.Linear(H * 4, H),
                    nn.Dropout(self.transformer_dropout)
                )
                
                self.transformer_encoder_layers.append(nn.ModuleDict({
                    'attention': attention_layer,
                    'norm1': norm1,
                    'norm2': norm2,
                    'ffn': ffn
                }))

            # 位置编码（可选）
            self.use_pos_encoding = config['use_pos_encoding'] if 'use_pos_encoding' in config else False
            if self.use_pos_encoding:
                self.pos_encoding = nn.Parameter(torch.randn(1, L, H) * 0.02)
            
            # 最终投影层
            self.output_projection = nn.Linear(H, embedding_size)
            
            # 聚合方法：'cls', 'mean', 'max', 'last'
            self.pooling_method = config['pooling_method'] if 'pooling_method' in config else 'cls'
            if self.pooling_method == 'cls':
                # 添加CLS token
                self.cls_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        elif proj_method == 'all':
            self.all_output_mode = config.get('all_output_mode', 'mean')  # 'mean' 或 'concat'
            self.all_mlp_mode = config.get('all_mlp_mode', 'separate')  # 'separate' 或 'shared'
            
            if self.all_output_mode == 'concat':
                self.num_feature_filed = L
            else:
                self.num_feature_filed = 1
            
            if self.all_mlp_mode == 'separate':
                # 为每一层创建独立的MLP
                self.layer_mlps = nn.ModuleList()
                for i in range(L):
                    if mlp_size_list is None:
                        layer_mlp = nn.Sequential(
                            nn.Linear(H, H // 2),
                            nn.ReLU(),
                            nn.Dropout(mlp_dropout),
                            nn.Linear(H // 2, embedding_size)
                        )
                    else:
                        size_list = [H] + mlp_size_list + [embedding_size]
                        layer_mlp = MLPLayers(size_list, mlp_dropout, last_activation=False)
                    self.layer_mlps.append(layer_mlp)
                    
            elif self.all_mlp_mode == 'shared':
                # 所有层共享同一个MLP
                if mlp_size_list is None:
                    self.shared_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    self.shared_mlp = MLPLayers(size_list, mlp_dropout, last_activation=False)
                    
            else:
                raise ValueError(f"Unknown all_mlp_mode: {self.all_mlp_mode}")
            if self.all_output_mode == 'mlp':
                # 输入维度是L个embedding拼接：L * embedding_size
                fusion_input_dim = L * embedding_size
                
                # 融合MLP的结构
                fusion_mlp_sizes = config.get('all_fusion_mlp_sizes', [fusion_input_dim // 2])
                
                if len(fusion_mlp_sizes) == 0:
                    # 如果没有指定隐藏层，直接线性变换
                    self.fusion_mlp = nn.Linear(fusion_input_dim, embedding_size)
                else:
                    # 创建多层MLP
                    fusion_size_list = [fusion_input_dim] + fusion_mlp_sizes + [embedding_size]
                    self.fusion_mlp = MLPLayers(fusion_size_list, mlp_dropout, last_activation=False)
                
                # 可选：在融合前对每个embedding进行layer normalization
                self.use_layer_norm = config.get('all_use_layer_norm', False)
                if self.use_layer_norm:
                    self.layer_norm = nn.LayerNorm(embedding_size)
                
                # 可选：在拼接前对每个embedding添加位置编码
                self.use_position_encoding = config.get('all_use_position_encoding', False)
                if self.use_position_encoding:
                    self.position_embeddings = nn.Parameter(torch.randn(L, embedding_size) * 0.02)
        elif proj_method == 'item_weight':
            # 直接为每个item学习一个L维度的权重向量
            # 为每一层创建独立的MLP进行特征投影到embedding_size
            self.layer_projection_mlps = nn.ModuleList()
            for i in range(L):
                if mlp_size_list is None:
                    layer_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    layer_mlp = MLPLayers(size_list, mlp_dropout)
                self.layer_projection_mlps.append(layer_mlp)
            
            # 为每个item学习一个L维度的权重向量
            self.item_layer_weights = nn.Embedding(self.n_items, L)
            
            # 初始化权重为均匀分布
            nn.init.uniform_(self.item_layer_weights.weight, 0.0, 1.0)
            
            # 权重归一化方式
            self.weight_norm_method = config.get('item_weight_norm_method', 'softmax')  # 'softmax', 'sigmoid', 'none'
            
            # 权重温度参数（用于softmax）
            self.weight_temperature = config.get('item_weight_temperature', 1.0)
            
            # 输出模式
            self.item_weight_output_mode = config.get('item_weight_output_mode', 'weighted_sum')  # 'weighted_sum' 或 'concat'
            
            if self.item_weight_output_mode == 'weighted_sum':
                self.num_feature_filed = 1
                # 可选：最终融合层
                self.final_fusion = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(embedding_size, embedding_size)
                ) if config.get('use_final_fusion', False) else nn.Identity()
            else:  # concat mode
                self.num_feature_filed = L
                # 可选：输出投影层
                self.output_projection = nn.Linear(embedding_size, embedding_size) if config.get('use_output_projection', False) else nn.Identity()
        elif proj_method == 'rnn':
            # RNN方法：将L层特征按序列输入RNN
            
            # RNN参数配置
            self.rnn_type = config.get('rnn_type', 'LSTM')  # 'LSTM', 'GRU', 'RNN'
            self.rnn_hidden_size = config.get('rnn_hidden_size', H)  # RNN隐藏层大小
            self.rnn_num_layers = config.get('rnn_num_layers', 1)  # RNN层数
            self.rnn_dropout = config.get('rnn_dropout', 0.0)  # RNN dropout
            self.rnn_bidirectional = config.get('rnn_bidirectional', False)  # 是否双向
            
            # 输入投影层：将每层特征投影到RNN输入维度
            self.input_projection = nn.Linear(H, self.rnn_hidden_size)
            
            # 创建RNN
            if self.rnn_type == 'LSTM':
                self.rnn = nn.LSTM(
                    input_size=self.rnn_hidden_size,
                    hidden_size=self.rnn_hidden_size,
                    num_layers=self.rnn_num_layers,
                    dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0.0,
                    bidirectional=self.rnn_bidirectional,
                    batch_first=True
                )
            elif self.rnn_type == 'GRU':
                self.rnn = nn.GRU(
                    input_size=self.rnn_hidden_size,
                    hidden_size=self.rnn_hidden_size,
                    num_layers=self.rnn_num_layers,
                    dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0.0,
                    bidirectional=self.rnn_bidirectional,
                    batch_first=True
                )
            elif self.rnn_type == 'RNN':
                self.rnn = nn.RNN(
                    input_size=self.rnn_hidden_size,
                    hidden_size=self.rnn_hidden_size,
                    num_layers=self.rnn_num_layers,
                    dropout=self.rnn_dropout if self.rnn_num_layers > 1 else 0.0,
                    bidirectional=self.rnn_bidirectional,
                    batch_first=True
                )
            else:
                raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
            
            # 计算RNN输出维度
            rnn_output_size = self.rnn_hidden_size
            if self.rnn_bidirectional:
                rnn_output_size *= 2
            
            # 输出处理方式
            self.rnn_pooling_method = config.get('rnn_pooling_method', 'last')  # 'last', 'mean', 'max', 'first'
            
            # 输出投影层：从RNN输出投影到embedding_size
            if mlp_size_list is None:
                # 默认投影结构
                self.output_projection = nn.Sequential(
                    nn.Linear(rnn_output_size, rnn_output_size // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(rnn_output_size // 2, embedding_size)
                )
            else:
                # 自定义投影结构
                size_list = [rnn_output_size] + mlp_size_list + [embedding_size]
                self.output_projection = MLPLayers(size_list, mlp_dropout)
            
            # 特征字段数量
            self.num_feature_filed = 1
        elif proj_method == 'attention':
            # 用户注意力机制
            self.user_attention_embedding = nn.Embedding(self.n_users, embedding_size)  # 用户直接用embedding_size维度
            
            # 为每一层创建独立的MLP进行特征投影到embedding_size
            self.layer_projection_mlps = nn.ModuleList()
            for i in range(L):
                if mlp_size_list is None:
                    layer_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    layer_mlp = MLPLayers(size_list, mlp_dropout)
                self.layer_projection_mlps.append(layer_mlp)
            
            # 注意力机制参数
            self.attention_dropout = config.get('attention_dropout', 0.1)
            
            # 注意力计算：用户query和层feature都是embedding_size维度，直接计算
            # 可选：多头注意力
            self.num_attention_heads = config.get('num_attention_heads', 2)
            if self.num_attention_heads > 1:
                assert embedding_size % self.num_attention_heads == 0
                self.head_dim = embedding_size // self.num_attention_heads
                
                # 多头注意力的投影层
                self.multi_head_query_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_key_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_value_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_output_projection = nn.Linear(embedding_size, embedding_size)
            else:
                # 单头注意力的投影层（可选，也可以直接用原始特征）
                self.query_projection = nn.Linear(embedding_size, embedding_size)
                self.key_projection = nn.Linear(embedding_size, embedding_size)
                self.value_projection = nn.Linear(embedding_size, embedding_size)
            
            # 注意力输出处理
            self.attention_output_mode = config.get('attention_output_mode', 'weighted_sum')  # 'weighted_sum' 或 'concat'
            
            if self.attention_output_mode == 'weighted_sum':
                self.num_feature_filed = 1
                # 最终融合层（可选）
                self.final_fusion = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(embedding_size, embedding_size)
                ) if config.get('use_final_fusion', False) else nn.Identity()
            else:  # concat mode
                self.num_feature_filed = L
                # 可选：添加一个输出投影层
                self.output_projection = nn.Linear(embedding_size, embedding_size) if config.get('use_output_projection', False) else nn.Identity()
        elif proj_method == 'attention_self':
            # 自注意力机制：不使用user_id，直接计算特征间的互注意力
            
            # 为每一层创建独立的MLP进行特征投影到embedding_size
            self.layer_projection_mlps = nn.ModuleList()
            for i in range(L):
                if mlp_size_list is None:
                    layer_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    layer_mlp = MLPLayers(size_list, mlp_dropout)
                self.layer_projection_mlps.append(layer_mlp)
            
            # 注意力机制参数
            self.attention_dropout = config.get('attention_dropout', 0.1)
            
            # 自注意力计算：只需要query、key、value投影层
            # 可选：多头注意力
            self.num_attention_heads = config.get('num_attention_heads', 2)
            if self.num_attention_heads > 1:
                assert embedding_size % self.num_attention_heads == 0
                self.head_dim = embedding_size // self.num_attention_heads
                
                # 多头自注意力的投影层
                self.multi_head_query_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_key_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_value_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_output_projection = nn.Linear(embedding_size, embedding_size)
            else:
                # 单头自注意力的投影层
                self.query_projection = nn.Linear(embedding_size, embedding_size)
                self.key_projection = nn.Linear(embedding_size, embedding_size)
                self.value_projection = nn.Linear(embedding_size, embedding_size)
            
            # 位置编码（可选）
            self.use_pos_encoding = config.get('use_pos_encoding', False)
            if self.use_pos_encoding:
                self.pos_encoding = nn.Parameter(torch.randn(1, L, embedding_size) * 0.02)
            
            # 注意力输出处理
            self.attention_output_mode = config.get('attention_output_mode', 'weighted_sum')  # 'weighted_sum' 或 'concat'
            
            if self.attention_output_mode == 'weighted_sum':
                self.num_feature_filed = 1
                # 最终融合层（可选）
                self.final_fusion = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(embedding_size, embedding_size)
                ) if config.get('use_final_fusion', True) else nn.Identity()
            else:  # concat mode
                self.num_feature_filed = L
                # 可选：输出投影层
                self.output_projection = nn.Linear(embedding_size, embedding_size) if config.get('use_output_projection', False) else nn.Identity()
        elif proj_method == 'attention_origin':
            # 用户注意力机制
            self.user_attention_embedding = nn.Embedding(self.n_users, embedding_size)  # 用户直接用embedding_size维度
            
            # 为每一层创建独立的MLP进行特征投影到embedding_size
            self.layer_projection_mlps = nn.ModuleList()
            for i in range(L):
                if mlp_size_list is None:
                    layer_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    layer_mlp = MLPLayers(size_list, mlp_dropout)
                self.layer_projection_mlps.append(layer_mlp)
            
            # 注意力机制参数
            self.attention_dropout = config.get('attention_dropout', 0.1)
            
            # 注意力计算：用户query和层feature都是embedding_size维度，直接计算
            # 可选：多头注意力
            self.num_attention_heads = config.get('num_attention_heads', 2)
            if self.num_attention_heads > 1:
                assert embedding_size % self.num_attention_heads == 0
                self.head_dim = embedding_size // self.num_attention_heads
                
                # 多头注意力的投影层
                self.multi_head_query_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_key_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_value_projection = nn.Linear(embedding_size, embedding_size)
                self.multi_head_output_projection = nn.Linear(embedding_size, embedding_size)
            else:
                # 单头注意力的投影层（可选，也可以直接用原始特征）
                self.query_projection = nn.Linear(embedding_size, embedding_size)
                self.key_projection = nn.Linear(embedding_size, embedding_size)
                self.value_projection = nn.Linear(embedding_size, embedding_size)
            
            # 注意力输出处理
            self.attention_output_mode = config.get('attention_output_mode', 'weighted_sum')  # 'weighted_sum' 或 'concat'
            
            if self.attention_output_mode == 'weighted_sum':
                self.num_feature_filed = 1
                # 最终融合层（可选）
                self.final_fusion = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(embedding_size, embedding_size)
                ) if config.get('use_final_fusion', True) else nn.Identity()
            else:  # concat mode
                self.num_feature_filed = L
                # 可选：添加一个输出投影层
                self.output_projection = nn.Linear(embedding_size, embedding_size) if config.get('use_output_projection', False) else nn.Identity()
        elif proj_method == 'attention_global':
            # 全局查询向量注意力机制
            
            # K个可学习的全局query向量，维度为H
            self.num_global_queries = config.get('num_global_queries', 4)  # K个全局query
            self.global_queries = nn.Parameter(torch.randn(self.num_global_queries, H) * 0.02)
            
            # 注意力机制参数
            self.attention_dropout = config.get('attention_dropout', 0.1)
            
            # 多头注意力配置
            self.num_attention_heads = config.get('num_attention_heads', 4)
            assert H % self.num_attention_heads == 0
            self.head_dim = H // self.num_attention_heads
            
            # 全局query的自注意力层（在H维度上）
            self.self_attention_query_proj = nn.Linear(H, H)
            self.self_attention_key_proj = nn.Linear(H, H)
            self.self_attention_value_proj = nn.Linear(H, H)
            self.self_attention_output_proj = nn.Linear(H, H)
            
            # 全局query与特征的互注意力层（都在H维度上）
            self.cross_attention_query_proj = nn.Linear(H, H)  # 用于全局query
            self.cross_attention_key_proj = nn.Linear(H, H)    # 用于特征
            self.cross_attention_value_proj = nn.Linear(H, H)  # 用于特征
            self.cross_attention_output_proj = nn.Linear(H, H)
            
            # Layer Normalization
            self.self_attention_norm = nn.LayerNorm(H)
            self.cross_attention_norm = nn.LayerNorm(H)
            
            # 为每个query创建独立的MLP：从H维度投影到embedding_size
            self.query_mlps = nn.ModuleList()
            for i in range(self.num_global_queries):
                if mlp_size_list is None:
                    query_mlp = nn.Sequential(
                        nn.Linear(H, H // 2),
                        nn.ReLU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(H // 2, embedding_size)
                    )
                else:
                    size_list = [H] + mlp_size_list + [embedding_size]
                    query_mlp = MLPLayers(size_list, mlp_dropout)
                self.query_mlps.append(query_mlp)
            
            # 输出K个特征
            self.num_feature_filed = self.num_global_queries
        
        elif proj_method == 'moe':
            self.moe_topk = config.get('moe_topk', 1)
            self.gate_type = config['moe_gate_type'] if 'moe_gate_type' in config else 'user'
            self.moe_mode = config.get('moe_mode', 'separate')
            self.experts = nn.ModuleList()
            for i in range(L):
                expert = nn.Sequential(
                    nn.Linear(H, H // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(H // 2, embedding_size)
                )
                self.experts.append(expert)
            

            # 门控网络：根据输入特征选择专家
            if self.gate_type == 'both':
                self.user_routing_embedding_size = config.get('user_routing_embedding_size', self.embedding_size)
                self.user_routing_embedding = nn.Embedding(self.n_users, self.user_routing_embedding_size)
                gate_input_dim = H + self.user_routing_embedding_size
            elif self.gate_type == 'both_id':
                self.user_routing_embedding_size = config.get('user_routing_embedding_size', self.embedding_size)
                self.user_routing_embedding = nn.Embedding(self.n_users, self.user_routing_embedding_size)
                self.item_routing_embedding_size = config.get('item_routing_embedding_size', self.embedding_size)
                self.item_id_embedding = nn.Embedding(self.n_items, self.item_routing_embedding_size)
                gate_input_dim = self.user_routing_embedding_size + self.item_routing_embedding_size
            elif self.gate_type == 'user':
                self.user_routing_embedding_size = config.get('user_routing_embedding_size', self.embedding_size)
                self.user_routing_embedding = nn.Embedding(self.n_users, self.user_routing_embedding_size)
                gate_input_dim = self.user_routing_embedding_size
            elif self.gate_type == 'item':
                gate_input_dim = H
            elif self.gate_type == 'item_id':
                self.item_routing_embedding_size = config.get('item_routing_embedding_size', self.embedding_size)
                self.item_id_embedding = nn.Embedding(self.n_items, self.item_routing_embedding_size)
                gate_input_dim = self.item_routing_embedding_size
            elif self.gate_type == 'dot_product':
                # gate_input_dim = H
                self.user_routing_embedding = nn.Embedding(self.n_users, L)
                self.item_projection_mlp = nn.Sequential(
                    nn.Linear(H, H // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(H // 2, L)
                )
                
                # 可选：添加一个温度参数用于缩放点积分数
                self.dot_temperature = config.get('dot_temperature', 1.0)
            elif self.gate_type == 'dot_product_mlp':
                # 新增：点积 + MLP 组合方法
                self.user_routing_embedding = nn.Embedding(self.n_users, L)
                self.item_projection_mlp = nn.Sequential(
                    nn.Linear(H, H // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(H // 2, L)
                )
                
                # concat后的MLP：将user_emb和item_proj拼接后通过MLP得到另一个L维向量
                self.concat_mlp = nn.Sequential(
                    nn.Linear(L + L, L),  # 输入是2L维（user_emb + item_proj）
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(L, L)
                )
                
                # 温度参数
                self.dot_temperature = config.get('dot_temperature', 1.0)
            if self.gate_type not in ['dot_product', 'dot_product_mlp']:
                self.gate_network = nn.Sequential(
                    nn.Linear(gate_input_dim, gate_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(gate_input_dim // 2, L),
                    nn.Softmax(dim=-1)
                )
            # self.gate_network = nn.Linear(gate_input_dim, L)

            # 可选：temperature参数用于控制选择的锐度
            self.temperature = config['moe_temperature'] if 'moe_temperature' in config else 1.0
            
            # 可选：是否使用hard selection (top-1) 或 soft selection (weighted)
            self.use_hard_selection = config['moe_hard_selection'] if 'moe_hard_selection' in config else True
        # 聚类聚合参数
        
        elif proj_method == 'cluster':
            self.embedding_tables = nn.ModuleList()
            keys = list(feature_dict.keys())
            self.n_clusters = n_clusters

            cluster_save_root = config['cluster_save_root'] if 'cluster_ave_root' in config else './cluster_results'
            os.makedirs(cluster_save_root, exist_ok=True)
            cluster_save_path = os.path.join(cluster_save_root, f"{config['dataset']}_n{n_clusters}_kmeans.pkl")
            # self.L = L
            if os.path.exists(cluster_save_path):
                print(f"Loading existing cluster results from {cluster_save_path}")
                with open(cluster_save_path, 'rb') as f:
                    cluster_data = pickle.load(f)
                    track_to_cluster_map = cluster_data['track_to_cluster_map']
                    track_ids_ordered = cluster_data['track_ids_ordered']

            else:
                print(f"Cluster results not found. Creating new clusters...")
                
                track_ids_ordered = []
                features_ordered = []

                for track_id, idx in sorted(token2id['tracks_id'].items(), key=lambda x: x[1]):
                    if track_id == '[PAD]':
                        feature = np.zeros((L, T, H))
                    else:
                        feature = feature_dict[track_id]
                    track_ids_ordered.append(track_id)
                    features_ordered.append(feature)

                track_to_cluster_map = []
                print("Clustering features ...")
                
                features_array = np.mean(np.array(features_ordered), axis=2, keepdims=False)
                for l in tqdm(range(self.L)):
                    vectors = features_array[:, l, :]
                    
                    import gc
                    gc.collect()
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    kmeans.fit(vectors)
                    labels = kmeans.labels_
                    track_to_cluster_map.append(labels)
                    self.embedding_tables.append(nn.Embedding(n_clusters, embedding_size))
                
                # 保存聚类结果
                cluster_data = {
                    'track_to_cluster_map': track_to_cluster_map,
                    'track_ids_ordered': track_ids_ordered
                }
                print(f"Saving cluster results to {cluster_save_path}")
                with open(cluster_save_path, 'wb') as f:
                    pickle.dump(cluster_data, f)
            # 无论是加载还是新建，都需要创建embedding tables
            if os.path.exists(cluster_save_path) and len(self.embedding_tables) == 0:
                for l in range(self.L):
                    self.embedding_tables.append(nn.Embedding(n_clusters, embedding_size))
            
            self.register_buffer('track_to_cluster_map', 
                                torch.tensor(track_to_cluster_map).T)  # [num_tracks, L]
            self.track_ids_ordered = track_ids_ordered
        elif proj_method == 'rq-kmeans':
            self.embedding_tables = nn.ModuleList()
            
            for l in range(L):  # 对每一层
                layer_embeddings = nn.ModuleList()
                for stage in range(n_stage):  # 对每个stage
                    layer_embeddings.append(nn.Embedding(n_clusters, embedding_size))
                self.embedding_tables.append(layer_embeddings)

            from recbole.model.RQ_optimized import RQKMeans
            keys = list(feature_dict.keys())
            self.n_clusters = n_clusters

            track_ids_ordered = []
            features_ordered = []
    
            for track_id, idx in sorted(token2id['tracks_id'].items(), key=lambda x: x[1]):
                # if track_id in feature_dict:
                if track_id == '[PAD]':
                    feature = np.zeros((L, T, H))
                else:
                    feature = feature_dict[track_id]
                track_ids_ordered.append(track_id)
                features_ordered.append(feature)

            track_to_cluster_map = []

            print("Clustering features ...")
            features_array = np.mean(np.array(features_ordered), axis=2, keepdims=False) # [N, L, H]
            for layer in tqdm(range(L)):  # 对每一层进行编码
                layer_features = features_array[:, layer, :]  # [N, H]
                
                # 为当前层创建RQ编码器
                rq_layer = RQKMeans(n_stages=n_stage, n_clusters=n_clusters)
                rq_layer.fit(layer_features)
                
                # 对当前层所有track进行编码
                layer_encoded_results = []
                for i in tqdm(range(len(layer_features))):
                    code = rq_layer.encode(layer_features[i:i+1])  # 返回 [n_stage] 的编码
                    layer_encoded_results.append(code)  # 去掉batch维度
                
                track_to_cluster_map.append(layer_encoded_results)  # [L, N, n_stage]
            
            print(torch.tensor(track_to_cluster_map).shape)
            encoded_tensor = torch.tensor(track_to_cluster_map).permute(1, 0, 2)  # [N, L, n_stage]
            self.register_buffer('track_to_cluster_map', encoded_tensor)

            # 统计编码碰撞数
            print("Analyzing code collisions...")
            
            # 将每个track的完整编码转换为字符串用于统计
            code_combinations = []
            for i in range(len(track_ids_ordered)):
                # 获取track i的所有层所有stage的编码
                track_codes = []
                for layer in range(L):
                    for stage in range(n_stage):
                        track_codes.append(str(track_to_cluster_map[layer][i][stage]))
                
                # 将所有编码组合成一个字符串
                code_str = '_'.join(track_codes)
                code_combinations.append(code_str)
            
            # 统计碰撞
            from collections import Counter
            code_counter = Counter(code_combinations)

            # 分析结果
            unique_codes = len(code_counter)
            total_tracks = len(track_ids_ordered)
            collision_count = 0
            collision_tracks = 0
            
            for code, count in code_counter.items():
                if count > 1:
                    collision_count += 1
                    collision_tracks += count
            print(f"=== RQ-KMeans Encoding Statistics ===")
            print(f"Total tracks: {total_tracks}")
            print(f"Unique code combinations: {unique_codes}")
            print(f"Collision rate: {collision_tracks/total_tracks:.4f} ({collision_tracks}/{total_tracks})")
            print(f"Number of colliding codes: {collision_count}")
            print(f"Average tracks per colliding code: {collision_tracks/collision_count:.2f}" if collision_count > 0 else "No collisions")

            self.n_stage = n_stage
        else:
            raise ValueError(f"Unknown projection method: {proj_method}")

    def forward(self, interaction):
        if self.proj_method in ['linear', 'mlp']:
            track_ids = interaction['tracks_id']
            
            if hasattr(self, 'layer') and self.layer == 'weighted_sum':
                # 获取所有层特征并reshape回 [batch_size, L, H]
                all_features = self.id2feats(track_ids)  # [batch_size, L*H]
                batch_size = all_features.size(0)
                all_features = all_features.view(batch_size, self.L, self.feature_dim)  # [batch_size, L, H]
                
                # 在forward时进行加权聚合
                # weights: [L, 1], all_features: [batch_size, L, H]
                weights = torch.softmax(self.weights, dim=0)  # 使用softmax确保权重和为1
                wav_features = torch.sum(weights.unsqueeze(0) * all_features, dim=1)  # [batch_size, H]
            else:
                wav_features = self.id2feats(track_ids)
                
            embed_features = self.net(wav_features)
            return embed_features.unsqueeze(1)
        elif self.proj_method == 'cluster':
            return self.get_cluster_embeddings(interaction)
        elif self.proj_method == 'rq-kmeans':
            return self.get_rq_cluster_embeddings(interaction)
        elif self.proj_method == 'transformer':
            return self.get_transformer_embeddings(interaction)
        elif self.proj_method == 'moe':
            return self.get_moe_embeddings(interaction)
        elif self.proj_method == 'all':
            return self.get_all_embeddings(interaction)
        elif self.proj_method == 'attention':
            return self.get_attention_embeddings(interaction)
        elif self.proj_method == 'attention_origin':
            return self.get_attention_origin_embeddings(interaction)
        elif self.proj_method == 'attention_self':
            return self.get_self_attention_embeddings(interaction)
        elif self.proj_method == 'item_weight':
            return self.get_item_weight_embeddings(interaction)
        elif self.proj_method == 'rnn':
            return self.get_rnn_embeddings(interaction)
        elif self.proj_method == 'attention_global':
            return self.get_global_attention_embeddings(interaction)
        elif self.proj_method == 'gate':
            return self.get_gate_embeddings(interaction)
        elif self.proj_method == 'pre_gate':
            return self.get_pre_gate_embeddings(interaction)
        elif self.proj_method == 'pre_moe':
            return self.get_pre_moe_embeddings(interaction)
        else:
            raise ValueError(f"Unknown aggregation method: {self.proj_method}")
    
    def _compute_gate_scores_for_gate_method(self, features, track_ids, user_ids):
        """
        为Gate方法计算门控分数
        Args:
            features: [batch_size, L, H] - 所有层特征
            track_ids: [batch_size] - item ID
            user_ids: [batch_size] - user ID
        Returns:
            gate_scores: [batch_size, L] - 门控分数
        """
        batch_size = features.size(0)
        
        # 根据gate_input_type计算门控输入
        if self.gate_input_type == 'mean_feature':
            # 使用所有层特征的平均值
            gate_input = torch.mean(features, dim=1)  # [batch_size, H]
            
        elif self.gate_input_type == 'item_id':
            # 使用item ID embedding
            gate_input = self.item_gate_embedding(track_ids)  # [batch_size, item_gate_embedding_size]
            
        elif self.gate_input_type == 'user_id':
            # 使用user ID embedding
            gate_input = self.user_gate_embedding(user_ids)  # [batch_size, user_gate_embedding_size]
            
        elif self.gate_input_type == 'both':
            # 使用特征平均值和item ID embedding的拼接
            mean_features = torch.mean(features, dim=1)  # [batch_size, H]
            item_embedding = self.item_gate_embedding(track_ids)  # [batch_size, item_gate_embedding_size]
            gate_input = torch.cat([mean_features, item_embedding], dim=-1)  # [batch_size, H + item_gate_embedding_size]
            
        elif self.gate_input_type == 'both_id':
            # 使用user ID embedding和item ID embedding的拼接
            user_embedding = self.user_gate_embedding(user_ids)  # [batch_size, user_gate_embedding_size]
            item_embedding = self.item_gate_embedding(track_ids)  # [batch_size, item_gate_embedding_size]
            gate_input = torch.cat([user_embedding, item_embedding], dim=-1)  # [batch_size, user_gate_embedding_size + item_gate_embedding_size]
        elif self.gate_input_type == 'user_feature':
            # 新增：使用特征平均值和user ID embedding的拼接
            mean_features = torch.mean(features, dim=1)  # [batch_size, H]
            user_embedding = self.user_gate_embedding(user_ids)  # [batch_size, user_gate_embedding_size]
            gate_input = torch.cat([mean_features, user_embedding], dim=-1)  # [batch_size, H + user_gate_embedding_size]
            
        elif self.gate_input_type == 'all':
            # 使用特征平均值、user ID embedding和item ID embedding的拼接
            mean_features = torch.mean(features, dim=1)  # [batch_size, H]
            user_embedding = self.user_gate_embedding(user_ids)  # [batch_size, user_gate_embedding_size]
            item_embedding = self.item_gate_embedding(track_ids)  # [batch_size, item_gate_embedding_size]
            gate_input = torch.cat([mean_features, user_embedding, item_embedding], dim=-1)  # [batch_size, H + user_gate_embedding_size + item_gate_embedding_size]
            
        else:
            raise ValueError(f"Unknown gate_input_type: {self.gate_input_type}")
        
        # 通过gate网络计算分数
        gate_scores = self.gate_network(gate_input)  # [batch_size, L]
        
        return gate_scores
    
    def get_gate_embeddings(self, interaction):
        """
        使用Gate方法：先得到L个embedding，再用item特征计算gate分数聚合
        Args:
            interaction: 包含tracks_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size]
        """
        track_ids = interaction['tracks_id']
        user_ids = interaction['user_id']  # 新增：获取user_id
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 第一步：根据gate_mode生成L个embedding
        if self.gate_mode == 'separate':
            # 使用独立的MLP处理每层
            layer_embeddings = []
            for layer_idx in range(L):
                layer_features = features[:, layer_idx, :]  # [batch_size, H]
                layer_embedding = self.layer_mlps[layer_idx](layer_features)  # [batch_size, embedding_size]
                layer_embeddings.append(layer_embedding)
            
            all_embeddings = torch.stack(layer_embeddings, dim=1)  # [batch_size, L, embedding_size]
            
        elif self.gate_mode == 'shared':
            # 使用共享的MLP处理所有层
            features_flat = features.view(batch_size * L, H)  # [batch_size * L, H]
            embeddings_flat = self.shared_mlp(features_flat)  # [batch_size * L, embedding_size]
            all_embeddings = embeddings_flat.view(batch_size, L, self.embedding_size)  # [batch_size, L, embedding_size]
        
        # 第二步：计算gate分数
        gate_scores = self._compute_gate_scores_for_gate_method(features, track_ids, user_ids)  # [batch_size, L]
        
        # 应用temperature
        if self.gate_temperature != 1.0:
            gate_scores = gate_scores / self.gate_temperature
            gate_scores = torch.softmax(gate_scores, dim=-1)
        
        # 应用dropout（训练时）
        if self.training and self.gate_dropout > 0:
            gate_scores = torch.dropout(gate_scores, self.gate_dropout, train=True)
        
        # 第三步：使用gate分数加权聚合L个embedding
        gate_scores = gate_scores.unsqueeze(-1)  # [batch_size, L, 1]
        weighted_embedding = torch.sum(gate_scores * all_embeddings, dim=1)  # [batch_size, embedding_size]
        
        # 最终融合层（可选）
        final_embedding = self.final_fusion(weighted_embedding)  # [batch_size, embedding_size]
        
        return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
    
    # def _compute_gate_scores_for_gate_method(self, features, track_ids):
    #     """
    #     为Gate方法计算门控分数
    #     Args:
    #         features: [batch_size, L, H] - 所有层特征
    #         track_ids: [batch_size] - item ID
    #     Returns:
    #         gate_scores: [batch_size, L] - 门控分数
    #     """
    #     batch_size = features.size(0)
        
    #     # 根据gate_input_type计算门控输入
    #     if self.gate_input_type == 'mean_feature':
    #         # 使用所有层特征的平均值
    #         gate_input = torch.mean(features, dim=1)  # [batch_size, H]
            
    #     elif self.gate_input_type == 'item_id':
    #         # 使用item ID embedding
    #         gate_input = self.item_gate_embedding(track_ids)  # [batch_size, item_gate_embedding_size]
            
    #     elif self.gate_input_type == 'both':
    #         # 使用特征平均值和item ID embedding的拼接
    #         mean_features = torch.mean(features, dim=1)  # [batch_size, H]
    #         item_embedding = self.item_gate_embedding(track_ids)  # [batch_size, item_gate_embedding_size]
    #         gate_input = torch.cat([mean_features, item_embedding], dim=-1)  # [batch_size, H + item_gate_embedding_size]
            
    #     else:
    #         raise ValueError(f"Unknown gate_input_type: {self.gate_input_type}")
        
    #     # 通过gate网络计算分数
    #     gate_scores = self.gate_network(gate_input)  # [batch_size, L]
        
    #     return gate_scores
    
    def get_rnn_embeddings(self, interaction):
        """
        使用RNN处理L层特征序列
        Args:
            interaction: 包含tracks_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size]
        """
        track_ids = interaction['tracks_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 输入投影：将每层特征投影到RNN输入维度
        projected_features = self.input_projection(features)  # [batch_size, L, rnn_hidden_size]
        
        # 通过RNN处理序列
        if self.rnn_type in ['LSTM']:
            # LSTM返回 (output, (hidden, cell))
            rnn_output, (final_hidden, final_cell) = self.rnn(projected_features)  # output: [batch_size, L, rnn_hidden_size * directions]
        elif self.rnn_type in ['GRU', 'RNN']:
            # GRU和RNN返回 (output, hidden)
            rnn_output, final_hidden = self.rnn(projected_features)  # output: [batch_size, L, rnn_hidden_size * directions]
        
        # 根据池化方法选择输出
        if self.rnn_pooling_method == 'last':
            # 使用最后一个时间步的输出
            if self.rnn_bidirectional:
                # 对于双向RNN，取正向和反向的最后输出并拼接
                # final_hidden shape: [num_layers * 2, batch_size, rnn_hidden_size]
                forward_hidden = final_hidden[-2]  # 正向最后一层
                backward_hidden = final_hidden[-1]  # 反向最后一层
                pooled_output = torch.cat([forward_hidden, backward_hidden], dim=-1)  # [batch_size, rnn_hidden_size * 2]
            else:
                # 单向RNN，直接取最后一层的隐藏状态
                pooled_output = final_hidden[-1]  # [batch_size, rnn_hidden_size]
                
        elif self.rnn_pooling_method == 'first':
            # 使用第一个时间步的输出
            pooled_output = rnn_output[:, 0, :]  # [batch_size, rnn_hidden_size * directions]
            
        elif self.rnn_pooling_method == 'mean':
            # 对所有时间步取平均
            pooled_output = torch.mean(rnn_output, dim=1)  # [batch_size, rnn_hidden_size * directions]
            
        elif self.rnn_pooling_method == 'max':
            # 对所有时间步取最大值
            pooled_output, _ = torch.max(rnn_output, dim=1)  # [batch_size, rnn_hidden_size * directions]
            
        elif self.rnn_pooling_method == 'attention':
            # 使用注意力机制加权平均
            if not hasattr(self, 'attention_weights'):
                # 如果没有定义注意力权重，使用简单的线性层
                attention_dim = rnn_output.size(-1)
                self.attention_weights = nn.Linear(attention_dim, 1)
            
            # 计算注意力分数
            attention_scores = self.attention_weights(rnn_output)  # [batch_size, L, 1]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, L, 1]
            
            # 加权求和
            pooled_output = torch.sum(attention_weights * rnn_output, dim=1)  # [batch_size, rnn_hidden_size * directions]
            
        else:
            raise ValueError(f"Unsupported pooling method: {self.rnn_pooling_method}")
        
        # 投影到embedding维度
        final_embedding = self.output_projection(pooled_output)  # [batch_size, embedding_size]
        
        return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
    # ...existing code...
    def get_pre_gate_embeddings(self, interaction):
        """
        使用Pre-Gate方法：用user_id学习L维权重，结合全局权重，先加权特征再通过MLP
        Args:
            interaction: 包含tracks_id和user_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size]
        """
        track_ids = interaction['tracks_id']
        user_ids = interaction['user_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 第一步：获取user embedding
        if hasattr(self, 'user_id_field_idx') and self.user_id_field_idx is not None:
            # 使用现有的token_embedding_table获取用户embedding
            user_token_field = user_ids.unsqueeze(1)  # [batch_size, 1]
            user_offset = self.token_field_offsets[self.user_id_field_idx]
            user_embedding_input = user_token_field + user_offset  # [batch_size, 1]
            user_embedding = self.token_embedding_table.embedding(user_embedding_input).squeeze(1)  # [batch_size, embedding_size]
        else:
            # 备用方法：通过完整的token fields获取
            token_fields = []
            for field_name in self.token_field_names:
                if field_name == self.USER_ID:
                    token_fields.append(interaction[field_name].unsqueeze(1))
                else:
                    if field_name in interaction:
                        token_fields.append(interaction[field_name].unsqueeze(1))
                    else:
                        dummy_field = torch.zeros_like(user_ids).unsqueeze(1)
                        token_fields.append(dummy_field)
            
            if len(token_fields) > 0:
                token_fields_tensor = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
                token_embeddings = self.token_embedding_table(token_fields_tensor)  # [batch_size, num_token_field, embed_dim]
                
                user_field_idx = None
                for i, field_name in enumerate(self.token_field_names):
                    if field_name == self.USER_ID:
                        user_field_idx = i
                        break
                
                if user_field_idx is not None:
                    user_embedding = token_embeddings[:, user_field_idx, :]  # [batch_size, embedding_size]
                else:
                    raise ValueError(f"USER_ID field {self.USER_ID} not found in token fields")
            else:
                raise ValueError("No token fields available to get user embedding")
        
        # 第二步：通过gate网络学习用户特定的L维权重
        user_gate_weights = self.gate_network(user_embedding)  # [batch_size, L]
        
        # 第三步：结合全局权重和用户权重
        if self.use_global_weights:
            # 将全局权重扩展到batch维度
            global_weights_expanded = self.global_weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, L]
            
            if self.weight_combination == 'add':
                # 简单相加
                combined_weights = user_gate_weights + global_weights_expanded  # [batch_size, L]
            elif self.weight_combination == 'weighted_add':
                # 加权相加，使用可学习的alpha参数
                alpha = torch.sigmoid(self.combination_alpha)  # 将alpha限制在[0,1]范围
                combined_weights = alpha * global_weights_expanded + (1 - alpha) * user_gate_weights  # [batch_size, L]
            elif self.weight_combination == 'concat':
                # 拼接后通过网络融合
                concatenated_weights = torch.cat([global_weights_expanded, user_gate_weights], dim=-1)  # [batch_size, 2*L]
                combined_weights = self.weight_fusion(concatenated_weights)  # [batch_size, L]
            else:
                raise ValueError(f"Unknown weight_combination: {self.weight_combination}")
        else:
            # 不使用全局权重，直接使用用户权重
            combined_weights = user_gate_weights  # [batch_size, L]
        
        # 第四步：权重归一化
        if self.gate_norm_method == 'softmax':
            # 使用softmax确保权重和为1
            if self.gate_temperature != 1.0:
                combined_weights = combined_weights / self.gate_temperature
            gate_weights = torch.softmax(combined_weights, dim=-1)  # [batch_size, L]
        elif self.gate_norm_method == 'sigmoid':
            # 使用sigmoid将权重限制在[0,1]范围
            gate_weights = torch.sigmoid(combined_weights)  # [batch_size, L]
        elif self.gate_norm_method == 'l1_norm':
            # L1归一化，权重和为1
            gate_weights = torch.abs(combined_weights)  # 确保非负
            gate_weights = gate_weights / (torch.sum(gate_weights, dim=-1, keepdim=True) + 1e-8)  # [batch_size, L]
        elif self.gate_norm_method == 'l2_norm':
            # L2归一化
            gate_weights = combined_weights / (torch.norm(combined_weights, p=2, dim=-1, keepdim=True) + 1e-8)  # [batch_size, L]
        elif self.gate_norm_method == 'none':
            # 不进行归一化，直接使用原始权重
            gate_weights = combined_weights
        else:
            raise ValueError(f"Unknown gate normalization method: {self.gate_norm_method}")
        
        # 应用gate dropout（训练时）
        if self.training and self.gate_dropout > 0:
            gate_weights = torch.dropout(gate_weights, self.gate_dropout, train=True)
        
        # 第五步：使用权重对特征进行加权
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, L, 1]
        weighted_features = features * gate_weights  # [batch_size, L, H]
        
        # 第六步：根据聚合方式处理加权特征
        if self.feature_aggregation == 'weighted_sum':
            # 加权求和：将L个加权特征求和
            aggregated_features = torch.sum(weighted_features, dim=1)  # [batch_size, H]
        elif self.feature_aggregation == 'weighted_concat':
            # 加权拼接：将L个加权特征拼接
            aggregated_features = weighted_features.view(batch_size, -1)  # [batch_size, L * H]
        else:
            raise ValueError(f"Unknown feature_aggregation: {self.feature_aggregation}")
        
        # 第七步：通过最终MLP得到embedding
        final_embedding = self.final_mlp(aggregated_features)  # [batch_size, embedding_size]
        
        return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
    
    def get_global_attention_embeddings(self, interaction):
        """
        使用K个全局query向量（维度为H），先自注意力，再与特征进行互注意力，最后每个query通过独立MLP得到最终embedding
        Args:
            interaction: 包含tracks_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, K, embedding_size]
        """
        track_ids = interaction['tracks_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 扩展全局query到batch维度（query维度为H）
        global_queries = self.global_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, K, H]
        
        # 第一步：全局query的自注意力（在H维度上）
        refined_queries = self._multi_head_self_attention_h(global_queries)  # [batch_size, K, H]
        
        # 第二步：全局query与特征的互注意力（都在H维度上）
        attended_queries = self._multi_head_cross_attention_h(refined_queries, features)  # [batch_size, K, H]
        
        # 第三步：每个query通过独立的MLP得到最终embedding（从H维度投影到embedding_size）
        final_embeddings = []
        for k in range(self.num_global_queries):
            query_k = attended_queries[:, k, :]  # [batch_size, H]
            final_embedding_k = self.query_mlps[k](query_k)  # [batch_size, embedding_size]
            final_embeddings.append(final_embedding_k)
        
        final_embeddings = torch.stack(final_embeddings, dim=1)  # [batch_size, K, embedding_size]
        
        return final_embeddings  # [batch_size, K, embedding_size]
    
    def _multi_head_self_attention_h(self, queries):
        """
        多头自注意力机制（在H维度上）
        Args:
            queries: [batch_size, K, H]
        Returns:
            refined_queries: [batch_size, K, H]
        """
        batch_size, seq_len, hidden_dim = queries.shape
        
        # 线性投影
        Q = self.self_attention_query_proj(queries)  # [batch_size, K, H]
        K = self.self_attention_key_proj(queries)    # [batch_size, K, H]
        V = self.self_attention_value_proj(queries)  # [batch_size, K, H]
        
        # 重塑为多头格式
        # [batch_size, K, num_heads, head_dim] -> [batch_size, num_heads, K, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, K, K]
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, K, K]
        
        # 应用dropout
        if self.training and self.attention_dropout > 0:
            attention_weights = torch.dropout(attention_weights, self.attention_dropout, train=True)
        
        # 加权聚合
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, num_heads, K, head_dim]
        
        # 重塑回原始格式
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim)  # [batch_size, K, H]
        
        # 输出投影
        output = self.self_attention_output_proj(attended_values)  # [batch_size, K, H]
        
        # 残差连接和层归一化
        output = self.self_attention_norm(output + queries)  # [batch_size, K, H]
        
        return output
    
    def _multi_head_cross_attention_h(self, queries, features):
        """
        多头互注意力机制：queries作为Q，features作为K和V（都在H维度上）
        Args:
            queries: [batch_size, K, H] - 全局query
            features: [batch_size, L, H] - 特征
        Returns:
            attended_queries: [batch_size, K, H]
        """
        batch_size, num_queries, hidden_dim = queries.shape
        num_features = features.shape[1]
        
        # 线性投影
        Q = self.cross_attention_query_proj(queries)   # [batch_size, K, H]
        K = self.cross_attention_key_proj(features)    # [batch_size, L, H]
        V = self.cross_attention_value_proj(features)  # [batch_size, L, H]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, num_queries, self.num_attention_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, K, head_dim]
        K = K.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, L, head_dim]
        V = V.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, L, head_dim]
        
        # 计算互注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, K, L]
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, K, L]
        
        # 应用dropout
        if self.training and self.attention_dropout > 0:
            attention_weights = torch.dropout(attention_weights, self.attention_dropout, train=True)
        
        # 加权聚合特征
        attended_values = torch.matmul(attention_weights, V)  # [batch_size, num_heads, K, head_dim]
        
        # 重塑回原始格式
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, num_queries, hidden_dim)  # [batch_size, K, H]
        
        # 输出投影
        output = self.cross_attention_output_proj(attended_values)  # [batch_size, K, H]
        
        # 残差连接和层归一化
        output = self.cross_attention_norm(output + queries)  # [batch_size, K, H]
        
        return output
    def get_self_attention_embeddings(self, interaction):
        """
        使用自注意力机制计算特征间的互注意力
        不需要user_id，直接基于特征内容计算注意力
        Args:
            interaction: 包含tracks_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size] 或 [batch_size, L, embedding_size]
        """
        track_ids = interaction['tracks_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 对每层特征投影到embedding_size维度
        projected_features = []
        for layer_idx in range(L):
            layer_features = features[:, layer_idx, :]  # [batch_size, H]
            projected_feature = self.layer_projection_mlps[layer_idx](layer_features)  # [batch_size, embedding_size]
            projected_features.append(projected_feature)
        
        projected_features = torch.stack(projected_features, dim=1)  # [batch_size, L, embedding_size]
        
        # 添加位置编码（可选）
        if self.use_pos_encoding:
            projected_features = projected_features + self.pos_encoding  # [batch_size, L, embedding_size]
        
        if self.num_attention_heads == 1:
            # 单头自注意力
            queries = self.query_projection(projected_features)  # [batch_size, L, embedding_size]
            keys = self.key_projection(projected_features)  # [batch_size, L, embedding_size]
            values = self.value_projection(projected_features)  # [batch_size, L, embedding_size]
            
            # 计算注意力分数：每个特征作为query与所有特征（包括自己）计算注意力
            attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # [batch_size, L, L]
            attention_scores = attention_scores / (self.embedding_size ** 0.5)  # 缩放
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, L, L]
            
            # 应用dropout
            if self.training and self.attention_dropout > 0:
                attention_weights = torch.dropout(attention_weights, self.attention_dropout, train=True)
            
            # 加权聚合：每个特征位置都得到一个加权后的表示
            attended_features = torch.bmm(attention_weights, values)  # [batch_size, L, embedding_size]
            
        else:
            # 多头自注意力
            queries = self.multi_head_query_projection(projected_features)  # [batch_size, L, embedding_size]
            keys = self.multi_head_key_projection(projected_features)  # [batch_size, L, embedding_size]
            values = self.multi_head_value_projection(projected_features)  # [batch_size, L, embedding_size]
            
            # 重塑为多头格式
            # [batch_size, L, num_heads, head_dim] -> [batch_size, num_heads, L, head_dim]
            queries = queries.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            keys = keys.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            values = values.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            
            # 计算多头注意力
            attention_heads = []
            for head in range(self.num_attention_heads):
                # 每个头的查询、键、值
                head_queries = queries[:, head, :, :]  # [batch_size, L, head_dim]
                head_keys = keys[:, head, :, :]  # [batch_size, L, head_dim]
                head_values = values[:, head, :, :]  # [batch_size, L, head_dim]
                
                # 计算注意力分数
                head_scores = torch.bmm(head_queries, head_keys.transpose(1, 2))  # [batch_size, L, L]
                head_scores = head_scores / (self.head_dim ** 0.5)
                head_weights = torch.softmax(head_scores, dim=-1)  # [batch_size, L, L]
                
                # 应用dropout
                if self.training and self.attention_dropout > 0:
                    head_weights = torch.dropout(head_weights, self.attention_dropout, train=True)
                
                # 加权聚合
                head_attended = torch.bmm(head_weights, head_values)  # [batch_size, L, head_dim]
                attention_heads.append(head_attended)
            
            # 拼接所有头的输出
            attended_features = torch.cat(attention_heads, dim=-1)  # [batch_size, L, embedding_size]
            
            # 最终投影
            attended_features = self.multi_head_output_projection(attended_features)  # [batch_size, L, embedding_size]
        
        # 根据输出模式返回结果
        if self.attention_output_mode == 'weighted_sum':
            # 对所有层的attended特征取平均
            pooled_features = torch.mean(attended_features, dim=1)  # [batch_size, embedding_size]
            
            # 通过最终融合层
            final_embedding = self.final_fusion(pooled_features)  # [batch_size, embedding_size]
            return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
            
        else:  # concat mode
            # 返回所有层的attended特征
            # 通过输出投影层
            attended_features = self.output_projection(attended_features)  # [batch_size, L, embedding_size]
            
            return attended_features  # [batch_size, L, embedding_size]

    def get_transformer_embeddings(self, interaction):
        track_ids = interaction['tracks_id']
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        # 加CLS token
        if self.pooling_method == 'cls':
            cls_token = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, H]
            features = torch.cat([cls_token, features], dim=1)  # [batch_size, L+1, H]
        # 加位置编码
        if self.use_pos_encoding:
            pos_encoding = self.pos_encoding
            if features.size(1) > pos_encoding.size(1):
                # 如果有CLS，补一行
                pos_encoding = torch.cat([torch.zeros(1,1,H,device=features.device), pos_encoding], dim=1)
            features = features + pos_encoding

        x = features
        for layer in self.transformer_encoder_layers:
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        # 聚合
        if self.pooling_method == 'cls':
            pooled = x[:, 0]  # [batch_size, H]
        elif self.pooling_method == 'mean':
            pooled = x.mean(dim=1)  # [batch_size, H]
        elif self.pooling_method == 'max':
            pooled, _ = x.max(dim=1)  # [batch_size, H]
        elif self.pooling_method == 'last':
            pooled = x[:, -1]  # [batch_size, H]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        embed_features = self.output_projection(pooled)  # [batch_size, embedding_size]
        return embed_features.unsqueeze(1)
    def get_item_weight_embeddings(self, interaction):
        """
        使用每个item学习的L维权重对每层特征进行加权
        Args:
            interaction: 包含tracks_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size] 或 [batch_size, L, embedding_size]
        """
        track_ids = interaction['tracks_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 对每层特征投影到embedding_size维度
        projected_features = []
        for layer_idx in range(L):
            layer_features = features[:, layer_idx, :]  # [batch_size, H]
            projected_feature = self.layer_projection_mlps[layer_idx](layer_features)  # [batch_size, embedding_size]
            projected_features.append(projected_feature)
        
        projected_features = torch.stack(projected_features, dim=1)  # [batch_size, L, embedding_size]
        
        # 获取每个item的L维权重
        item_weights = self.item_layer_weights(track_ids)  # [batch_size, L]
        
        # 根据归一化方法处理权重
        if self.weight_norm_method == 'softmax':
            # 使用softmax确保权重和为1
            if self.weight_temperature != 1.0:
                item_weights = item_weights / self.weight_temperature
            item_weights = torch.softmax(item_weights, dim=-1)  # [batch_size, L]
        elif self.weight_norm_method == 'sigmoid':
            # 使用sigmoid将权重限制在[0,1]范围
            item_weights = torch.sigmoid(item_weights)  # [batch_size, L]
        elif self.weight_norm_method == 'l1_norm':
            # L1归一化，权重和为1
            item_weights = torch.abs(item_weights)  # 确保非负
            item_weights = item_weights / (torch.sum(item_weights, dim=-1, keepdim=True) + 1e-8)  # [batch_size, L]
        elif self.weight_norm_method == 'l2_norm':
            # L2归一化
            item_weights = item_weights / (torch.norm(item_weights, p=2, dim=-1, keepdim=True) + 1e-8)  # [batch_size, L]
        elif self.weight_norm_method == 'none':
            # 不进行归一化，直接使用原始权重
            pass
        else:
            raise ValueError(f"Unknown weight normalization method: {self.weight_norm_method}")
        
        # 根据输出模式返回结果
        if self.item_weight_output_mode == 'weighted_sum':
            # 使用item权重对投影后的特征进行加权求和
            item_weights = item_weights.unsqueeze(-1)  # [batch_size, L, 1]
            weighted_features = projected_features * item_weights  # [batch_size, L, embedding_size]
            final_embedding = torch.sum(weighted_features, dim=1)  # [batch_size, embedding_size]
            
            # 通过最终融合层（可选）
            final_embedding = self.final_fusion(final_embedding)  # [batch_size, embedding_size]
            return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
            
        else:  # concat mode
            # 返回每层的加权特征
            item_weights = item_weights.unsqueeze(-1)  # [batch_size, L, 1]
            weighted_features = projected_features * item_weights  # [batch_size, L, embedding_size]
            
            # 通过输出投影层（可选）
            weighted_features = self.output_projection(weighted_features)  # [batch_size, L, embedding_size]
            
            return weighted_features  # [batch_size, L, embedding_size]
    def get_attention_embeddings(self, interaction):
        """
        使用用户注意力机制对每层特征进行加权聚合
        Args:
            interaction: 包含tracks_id和user_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size] 或 [batch_size, L, embedding_size]
        """
        track_ids = interaction['tracks_id']
        user_ids = interaction['user_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 获取用户embedding（直接是embedding_size维度）
        user_query = self.user_attention_embedding(user_ids)  # [batch_size, embedding_size]
        
        # 对每层特征投影到embedding_size维度
        projected_features = []
        for layer_idx in range(L):
            layer_features = features[:, layer_idx, :]  # [batch_size, H]
            projected_feature = self.layer_projection_mlps[layer_idx](layer_features)  # [batch_size, embedding_size]
            projected_features.append(projected_feature)
        
        projected_features = torch.stack(projected_features, dim=1)  # [batch_size, L, embedding_size]
        
        if self.num_attention_heads == 1:
            # 单头注意力
            # 投影到注意力空间（可选，也可以直接使用原始特征）
            query = self.query_projection(user_query)  # [batch_size, embedding_size]
            keys = self.key_projection(projected_features)  # [batch_size, L, embedding_size]
            values = self.value_projection(projected_features)  # [batch_size, L, embedding_size]
            
            # 计算注意力分数
            # query: [batch_size, embedding_size] -> [batch_size, 1, embedding_size]
            query = query.unsqueeze(1)
            attention_scores = torch.bmm(query, keys.transpose(1, 2))  # [batch_size, 1, L]
            attention_scores = attention_scores / (self.embedding_size ** 0.5)  # 缩放
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, L]
            
            # 应用dropout
            if self.training and self.attention_dropout > 0:
                attention_weights = torch.dropout(attention_weights, self.attention_dropout, train=True)
            
            # 加权聚合
            attended_features = torch.bmm(attention_weights, values)  # [batch_size, 1, embedding_size]
            attended_features = attended_features.squeeze(1)  # [batch_size, embedding_size]
            
        else:
            # 多头注意力
            query = self.multi_head_query_projection(user_query)  # [batch_size, embedding_size]
            keys = self.multi_head_key_projection(projected_features)  # [batch_size, L, embedding_size]
            values = self.multi_head_value_projection(projected_features)  # [batch_size, L, embedding_size]
            
            # 重塑为多头格式
            # [batch_size, num_heads, head_dim]
            query = query.view(batch_size, self.num_attention_heads, self.head_dim)
            # [batch_size, num_heads, L, head_dim]
            keys = keys.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            values = values.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            
            # 计算多头注意力
            attention_heads = []
            for head in range(self.num_attention_heads):
                # 每个头的查询、键、值
                head_query = query[:, head:head+1, :]  # [batch_size, 1, head_dim]
                head_keys = keys[:, head, :, :]  # [batch_size, L, head_dim]
                head_values = values[:, head, :, :]  # [batch_size, L, head_dim]
                
                # 计算注意力分数
                head_scores = torch.bmm(head_query, head_keys.transpose(1, 2))  # [batch_size, 1, L]
                head_scores = head_scores / (self.head_dim ** 0.5)
                head_weights = torch.softmax(head_scores, dim=-1)
                
                # 加权聚合
                head_attended = torch.bmm(head_weights, head_values)  # [batch_size, 1, head_dim]
                attention_heads.append(head_attended)
            
            # 拼接所有头的输出
            attended_features = torch.cat(attention_heads, dim=-1)  # [batch_size, 1, embedding_size]
            attended_features = attended_features.squeeze(1)  # [batch_size, embedding_size]
            
            # 最终投影
            attended_features = self.multi_head_output_projection(attended_features)  # [batch_size, embedding_size]
        
        # 根据输出模式返回结果
        if self.attention_output_mode == 'weighted_sum':
            # 直接使用注意力加权的结果
            final_embedding = self.final_fusion(attended_features)  # [batch_size, embedding_size]
            return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
            
        else:  # concat mode
            # 重新计算注意力权重并应用到投影后的特征上
            query = user_query.unsqueeze(1)  # [batch_size, 1, embedding_size]
            attention_scores = torch.bmm(query, projected_features.transpose(1, 2))  # [batch_size, 1, L]
            attention_scores = attention_scores / (self.embedding_size ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, L]
            
            # 将注意力权重应用到每个投影后的特征上
            attention_weights = attention_weights.transpose(1, 2)  # [batch_size, L, 1]
            weighted_embeddings = projected_features * attention_weights  # [batch_size, L, embedding_size]
            
            # 通过输出投影层
            weighted_embeddings = self.output_projection(weighted_embeddings)  # [batch_size, L, embedding_size]
            
            return weighted_embeddings  # [batch_size, L, embedding_size]
    def get_pre_moe_embeddings(self, interaction):
        """
        使用Pre-MoE方法：用user_id学习L维权重，选择top1特征进入MLP
        Args:
            interaction: 包含tracks_id和user_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size]
        """
        track_ids = interaction['tracks_id']
        user_ids = interaction['user_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 第一步：获取user embedding
        if hasattr(self, 'user_id_field_idx') and self.user_id_field_idx is not None:
            # 使用现有的token_embedding_table获取用户embedding
            user_token_field = user_ids.unsqueeze(1)  # [batch_size, 1]
            user_offset = self.token_field_offsets[self.user_id_field_idx]
            user_embedding_input = user_token_field + user_offset  # [batch_size, 1]
            user_embedding = self.token_embedding_table.embedding(user_embedding_input).squeeze(1)  # [batch_size, embedding_size]
        else:
            # 备用方法：通过完整的token fields获取
            token_fields = []
            for field_name in self.token_field_names:
                if field_name == self.USER_ID:
                    token_fields.append(interaction[field_name].unsqueeze(1))
                else:
                    if field_name in interaction:
                        token_fields.append(interaction[field_name].unsqueeze(1))
                    else:
                        dummy_field = torch.zeros_like(user_ids).unsqueeze(1)
                        token_fields.append(dummy_field)
            
            if len(token_fields) > 0:
                token_fields_tensor = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
                token_embeddings = self.token_embedding_table(token_fields_tensor)  # [batch_size, num_token_field, embed_dim]
                
                user_field_idx = None
                for i, field_name in enumerate(self.token_field_names):
                    if field_name == self.USER_ID:
                        user_field_idx = i
                        break
                
                if user_field_idx is not None:
                    user_embedding = token_embeddings[:, user_field_idx, :]  # [batch_size, embedding_size]
                else:
                    raise ValueError(f"USER_ID field {self.USER_ID} not found in token fields")
            else:
                raise ValueError("No token fields available to get user embedding")
        
        # 第二步：通过gate网络学习用户特定的L维权重
        user_gate_weights = self.gate_network(user_embedding)  # [batch_size, L]
        
        # 第三步：结合全局权重和用户权重（如果启用）
        if self.use_global_weights:
            # 将全局权重扩展到batch维度
            global_weights_expanded = self.global_weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, L]
            
            if self.weight_combination == 'add':
                # 简单相加
                combined_weights = user_gate_weights + global_weights_expanded  # [batch_size, L]
            elif self.weight_combination == 'weighted_add':
                # 加权相加，使用可学习的alpha参数
                alpha = torch.sigmoid(self.combination_alpha)  # 将alpha限制在[0,1]范围
                combined_weights = alpha * global_weights_expanded + (1 - alpha) * user_gate_weights  # [batch_size, L]
            elif self.weight_combination == 'concat':
                # 拼接后通过网络融合
                concatenated_weights = torch.cat([global_weights_expanded, user_gate_weights], dim=-1)  # [batch_size, 2*L]
                combined_weights = self.weight_fusion(concatenated_weights)  # [batch_size, L]
            else:
                raise ValueError(f"Unknown weight_combination: {self.weight_combination}")
        else:
            # 不使用全局权重，直接使用用户权重
            combined_weights = user_gate_weights  # [batch_size, L]
        
        # 应用温度参数
        if self.gate_temperature != 1.0:
            combined_weights = combined_weights / self.gate_temperature
        
        # 应用gate dropout（训练时）
        if self.training and self.gate_dropout > 0:
            combined_weights = torch.dropout(combined_weights, self.gate_dropout, train=True)
        
        # 第四步：选择top1特征
        if self.use_gumbel and self.training:
            # 使用Gumbel-Softmax进行可微分的离散选择
            from torch.nn.functional import gumbel_softmax
            # Gumbel-Softmax会返回one-hot向量的软近似
            gumbel_weights = gumbel_softmax(combined_weights, tau=self.gumbel_temperature, hard=self.gumbel_hard, dim=-1)  # [batch_size, L]
            
            if self.gumbel_hard:
                # Hard模式下，选中的特征权重为1，其他为0
                selected_features = torch.sum(features * gumbel_weights.unsqueeze(-1), dim=1)  # [batch_size, H]
            else:
                # Soft模式下，使用软权重
                selected_features = torch.sum(features * gumbel_weights.unsqueeze(-1), dim=1)  # [batch_size, H]
        else:
            # 使用硬选择：直接选择top1
            _, top_indices = torch.max(combined_weights, dim=-1)  # [batch_size] - 每个样本的top1索引
            
            # 向量化选择：使用gather操作
            batch_indices = torch.arange(batch_size, device=features.device).unsqueeze(1)  # [batch_size, 1]
            top_indices = top_indices.unsqueeze(1)  # [batch_size, 1]
            
            # 选择对应的特征
            selected_features = features[batch_indices, top_indices].squeeze(1)  # [batch_size, H]
        
        # 第五步：通过最终MLP得到embedding
        final_embedding = self.final_mlp(selected_features)  # [batch_size, embedding_size]
        
        return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
    def get_attention_origin_embeddings(self, interaction):
        """
        使用用户注意力机制对每层特征进行加权聚合
        Args:
            interaction: 包含tracks_id和user_id的交互数据
        Returns:
            embeddings: tensor of shape [batch_size, 1, embedding_size] 或 [batch_size, L, embedding_size]
        """
        track_ids = interaction['tracks_id']
        user_ids = interaction['user_id']
        
        # 获取所有层特征
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 获取用户embedding（直接是embedding_size维度）
        user_query = self.user_attention_embedding(user_ids)  # [batch_size, embedding_size]
        
        # 对每层特征投影到embedding_size维度
        projected_features = []
        for layer_idx in range(L):
            layer_features = features[:, layer_idx, :]  # [batch_size, H]
            projected_feature = self.layer_projection_mlps[layer_idx](layer_features)  # [batch_size, embedding_size]
            projected_features.append(projected_feature)
        
        projected_features = torch.stack(projected_features, dim=1)  # [batch_size, L, embedding_size]
        
        if self.num_attention_heads == 1:
            # 单头注意力
            # 投影到注意力空间（可选，也可以直接使用原始特征）
            query = self.query_projection(user_query)  # [batch_size, embedding_size]
            keys = self.key_projection(projected_features)  # [batch_size, L, embedding_size]
            values = self.value_projection(projected_features)  # [batch_size, L, embedding_size]
            
            # 计算注意力分数
            # query: [batch_size, embedding_size] -> [batch_size, 1, embedding_size]
            query = query.unsqueeze(1)
            attention_scores = torch.bmm(query, keys.transpose(1, 2))  # [batch_size, 1, L]
            attention_scores = attention_scores / (self.embedding_size ** 0.5)  # 缩放
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, L]
            
            # 应用dropout
            if self.training and self.attention_dropout > 0:
                attention_weights = torch.dropout(attention_weights, self.attention_dropout, train=True)
            
            # 加权聚合
            attended_features = torch.bmm(attention_weights, values)  # [batch_size, 1, embedding_size]
            attended_features = attended_features.squeeze(1)  # [batch_size, embedding_size]
            
        else:
            # 多头注意力
            query = self.multi_head_query_projection(user_query)  # [batch_size, embedding_size]
            keys = self.multi_head_key_projection(projected_features)  # [batch_size, L, embedding_size]
            values = self.multi_head_value_projection(projected_features)  # [batch_size, L, embedding_size]
            
            # 重塑为多头格式
            # [batch_size, num_heads, head_dim]
            query = query.view(batch_size, self.num_attention_heads, self.head_dim)
            # [batch_size, num_heads, L, head_dim]
            keys = keys.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            values = values.view(batch_size, L, self.num_attention_heads, self.head_dim).transpose(1, 2)
            
            # 计算多头注意力
            attention_heads = []
            for head in range(self.num_attention_heads):
                # 每个头的查询、键、值
                head_query = query[:, head:head+1, :]  # [batch_size, 1, head_dim]
                head_keys = keys[:, head, :, :]  # [batch_size, L, head_dim]
                head_values = values[:, head, :, :]  # [batch_size, L, head_dim]
                
                # 计算注意力分数
                head_scores = torch.bmm(head_query, head_keys.transpose(1, 2))  # [batch_size, 1, L]
                head_scores = head_scores / (self.head_dim ** 0.5)
                head_weights = torch.softmax(head_scores, dim=-1)
                
                # 加权聚合
                head_attended = torch.bmm(head_weights, head_values)  # [batch_size, 1, head_dim]
                attention_heads.append(head_attended)
            
            # 拼接所有头的输出
            attended_features = torch.cat(attention_heads, dim=-1)  # [batch_size, 1, embedding_size]
            attended_features = attended_features.squeeze(1)  # [batch_size, embedding_size]
            
            # 最终投影
            attended_features = self.multi_head_output_projection(attended_features)  # [batch_size, embedding_size]
        
        # 根据输出模式返回结果
        if self.attention_output_mode == 'weighted_sum':
            # 直接使用注意力加权的结果
            final_embedding = self.final_fusion(attended_features)  # [batch_size, embedding_size]
            return final_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
            
        else:  # concat mode
            # 重新计算注意力权重并应用到投影后的特征上
            query = user_query.unsqueeze(1)  # [batch_size, 1, embedding_size]
            attention_scores = torch.bmm(query, projected_features.transpose(1, 2))  # [batch_size, 1, L]
            attention_scores = attention_scores / (self.embedding_size ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, L]
            
            # 将注意力权重应用到每个投影后的特征上
            attention_weights = attention_weights.transpose(1, 2)  # [batch_size, L, 1]
            weighted_embeddings = projected_features * attention_weights  # [batch_size, L, embedding_size]
            
            # 通过输出投影层
            weighted_embeddings = self.output_projection(weighted_embeddings)  # [batch_size, L, embedding_size]
            
            return weighted_embeddings  # [batch_size, L, embedding_size]
    def get_moe_embeddings(self, interaction):
        track_ids = interaction['tracks_id']
        user_ids = interaction['user_id']
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 计算每层特征的平均值作为门控网络的输入
        # 可以使用不同的策略，这里使用所有层的平均
        if self.gate_type == 'user':
            user_routing_emb = self.user_routing_embedding(user_ids)  # [batch_size, user_routing_embedding_size]
            gate_input = user_routing_emb  # [batch_size, user_routing_embedding_size]
        elif self.gate_type == 'item':
            gate_input = features.mean(dim=1)  # [batch_size, H]
        elif self.gate_type == 'item_id':
            item_routing_emb = self.item_id_embedding(track_ids)  # [batch_size, item_routing_embedding_size]
            gate_input = item_routing_emb  # [batch_size, item_routing_embedding_size]
        elif self.gate_type == 'both':
            user_routing_emb = self.user_routing_embedding(user_ids)  # [batch_size, user_routing_embedding_size]
            item_feature = features.mean(dim=1)  # [batch_size, H]
            gate_input = torch.cat([user_routing_emb, item_feature], dim=-1)  # [batch_size, user_routing_embedding_size + H]
        elif self.gate_type == 'both_id':
            user_routing_emb = self.user_routing_embedding(user_ids)  # [batch_size, user_routing_embedding_size]
            item_routing_emb = self.item_id_embedding(track_ids)  # [batch_size, item_routing_embedding_size]
            gate_input = torch.cat([user_routing_emb, item_routing_emb], dim=-1)  # [batch_size, user_routing_embedding_size + item_routing_embedding_size]
        elif self.gate_type == 'dot_product':
            user_routing_emb = self.user_routing_embedding(user_ids)  # [batch_size, L]
            
            # item feature平均后通过MLP投影到L维
            item_feature = features.mean(dim=1)  # [batch_size, H]
            item_proj = self.item_projection_mlp(item_feature)  # [batch_size, L]
            
            # 计算元素级点积作为每个专家的分数
            gate_scores = user_routing_emb * item_proj  # [batch_size, L]
            
            # 应用温度缩放
            if self.dot_temperature != 1.0:
                gate_scores = gate_scores / self.dot_temperature

            # 应用softmax得到概率分布
            gate_scores = torch.softmax(gate_scores, dim=-1)
        elif self.gate_type == 'dot_product_mlp':
            # 新增：点积 + MLP 组合方法
            user_routing_emb = self.user_routing_embedding(user_ids)  # [batch_size, L]
            
            # item feature投影到L维
            item_feature = features.mean(dim=1)  # [batch_size, H]
            item_proj = self.item_projection_mlp(item_feature)  # [batch_size, L]
            
            # 方法1：点积
            dot_scores = user_routing_emb * item_proj  # [batch_size, L]
            
            # 方法2：concat后通过MLP
            concat_input = torch.cat([user_routing_emb, item_proj], dim=-1)  # [batch_size, 2L]
            mlp_scores = self.concat_mlp(concat_input)  # [batch_size, L]
            
            # 将两种方法的结果相加
            gate_scores = dot_scores + mlp_scores  # [batch_size, L]
            
            # 应用温度缩放和softmax
            if self.dot_temperature != 1.0:
                gate_scores = gate_scores / self.dot_temperature
            gate_scores = torch.softmax(gate_scores, dim=-1)
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")
        # 通过门控网络计算每个专家的权重
        if self.gate_type not in ['dot_product', 'dot_product_mlp']:
            gate_scores = self.gate_network(gate_input)  # [batch_size, L]
        
            # 应用temperature
            if self.temperature != 1.0:
                gate_scores = gate_scores / self.temperature
                gate_scores = torch.softmax(gate_scores, dim=-1)
        
        if self.use_hard_selection:
            if self.moe_topk == 1:
                # Hard selection: 选择top-1专家（向量化版本）
                _, top_expert_idx = torch.max(gate_scores, dim=-1)  # [batch_size]
                
                # 向量化选择特征和专家
                batch_indices = torch.arange(batch_size, device=features.device)
                selected_features = features[batch_indices, top_expert_idx]  # [batch_size, H]
                
                # 预计算所有专家的输出
                all_expert_outputs = []
                for expert in self.experts:
                    expert_output = expert(selected_features)  # [batch_size, embedding_size]
                    all_expert_outputs.append(expert_output)
                all_expert_outputs = torch.stack(all_expert_outputs, dim=1)  # [batch_size, L, embedding_size]
                
                # 根据选择的专家索引提取对应输出
                final_embeddings = all_expert_outputs[batch_indices, top_expert_idx]  # [batch_size, embedding_size]
            else:
                # Top-k选择
                topk = min(self.moe_topk, L)  # 确保topk不超过专家数量
                topk_scores, topk_indices = torch.topk(gate_scores, k=topk, dim=-1)  # [batch_size, topk]
                
                # 重新归一化topk分数
                topk_scores = torch.softmax(topk_scores, dim=-1)  # [batch_size, topk]
                
                # 计算选中专家的输出 - 向量化版本
                # 为每个batch样本和每个topk专家计算输出
                expert_outputs = []
                for k in range(topk):
                    expert_indices = topk_indices[:, k]  # [batch_size]
                    batch_indices = torch.arange(batch_size, device=features.device)
                    
                    # 选择对应的特征
                    selected_features = features[batch_indices, expert_indices]  # [batch_size, H]
                    
                    # 计算所有专家的输出
                    all_expert_outputs = []
                    for expert in self.experts:
                        expert_output = expert(selected_features)  # [batch_size, embedding_size]
                        all_expert_outputs.append(expert_output)
                    all_expert_outputs = torch.stack(all_expert_outputs, dim=1)  # [batch_size, L, embedding_size]
                    
                    # 根据专家索引选择输出
                    selected_outputs = all_expert_outputs[batch_indices, expert_indices]  # [batch_size, embedding_size]
                    expert_outputs.append(selected_outputs)
                # 加权聚合topk专家的输出
                expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, topk, embedding_size]
                topk_scores = topk_scores.unsqueeze(-1)  # [batch_size, topk, 1]
                final_embeddings = torch.sum(topk_scores * expert_outputs, dim=1)  # [batch_size, embedding_size]
        else:
            # Soft selection: 加权组合所有专家的输出
            expert_outputs = []
            for layer_idx in range(L):
                layer_features = features[:, layer_idx, :]  # [batch_size, H]
                expert_output = self.experts[layer_idx](layer_features)  # [batch_size, embedding_size]
                expert_outputs.append(expert_output)
            
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, L, embedding_size]
            
            # 加权求和
            gate_scores = gate_scores.unsqueeze(-1)  # [batch_size, L, 1]
            final_embeddings = torch.sum(gate_scores * expert_outputs, dim=1)  # [batch_size, embedding_size]
        
        return final_embeddings.unsqueeze(1)
    def get_cluster_embeddings(self, interaction):
        """
        获取track_ids对应的L个cluster embeddings
        Args:
            track_ids: tensor of shape [batch_size] (这些是token2id['tracks_id']中的索引值)
        Returns:
            embeddings: tensor of shape [batch_size, L, embedding_size]
        """
        # track_ids已经是token2id['tracks_id']中的索引，直接使用
        # 但需要减去padding_idx(通常是0)来对齐到我们的映射表
        track_ids = interaction['tracks_id']
        track_indices = track_ids  # 假设padding_idx=0，实际track索引从1开始
        
        # 获取聚类ID [batch_size, L]
        cluster_ids = self.track_to_cluster_map[track_indices]
        
        # 批量获取embeddings
        embeddings = []
        for l in range(self.L):
            emb = self.embedding_tables[l](cluster_ids[:, l])  # [batch_size, embedding_size]
            embeddings.append(emb)
        
        return torch.stack(embeddings, dim=1)  # [batch_size, L, embedding_size]

    def get_all_embeddings(self, interaction):
        track_ids = interaction['tracks_id']
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        if self.all_mlp_mode == 'separate':
            # 使用独立的MLP处理每层
            layer_embeddings = []
            for layer_idx in range(L):
                layer_features = features[:, layer_idx, :]  # [batch_size, H]
                layer_embedding = self.layer_mlps[layer_idx](layer_features)  # [batch_size, embedding_size]
                layer_embeddings.append(layer_embedding)
            
            all_embeddings = torch.stack(layer_embeddings, dim=1)  # [batch_size, L, embedding_size]
            
        elif self.all_mlp_mode == 'shared':
            # 使用共享的MLP处理所有层
            # 将所有层的特征reshape为 [batch_size * L, H]，然后一次性处理
            features_flat = features.view(batch_size * L, H)  # [batch_size * L, H]
            embeddings_flat = self.shared_mlp(features_flat)  # [batch_size * L, embedding_size]
            all_embeddings = embeddings_flat.view(batch_size, L, self.embedding_size)  # [batch_size, L, embedding_size]
            
        else:
            raise ValueError(f"Unknown all_mlp_mode: {self.all_mlp_mode}")
        
        # 根据输出模式返回结果
        if self.all_output_mode == 'mean':
            # 对所有层取平均，返回单个特征
            mean_embedding = torch.mean(all_embeddings, dim=1, keepdim=True)  # [batch_size, 1, embedding_size]
            return mean_embedding
        elif self.all_output_mode == 'mlp':
            # 新增：使用MLP融合L个embedding
            
            # 可选：添加位置编码
            if self.use_position_encoding:
                all_embeddings = all_embeddings + self.position_embeddings.unsqueeze(0)  # [batch_size, L, embedding_size]
            
            # 可选：layer normalization
            if self.use_layer_norm:
                all_embeddings = self.layer_norm(all_embeddings)  # [batch_size, L, embedding_size]
            
            # 将L个embedding拼接成一个长向量
            concatenated_embeddings = all_embeddings.view(batch_size, -1)  # [batch_size, L * embedding_size]
            
            # 通过融合MLP得到最终的单个embedding
            fused_embedding = self.fusion_mlp(concatenated_embeddings)  # [batch_size, embedding_size]
            
            return fused_embedding.unsqueeze(1)  # [batch_size, 1, embedding_size]
        else:
            # 返回所有L个特征
            return all_embeddings  # [batch_size, L, embedding_size]
    def get_rq_cluster_embeddings(self, interaction):
        """
        获取track_ids对应的RQ编码embeddings
        Args:
            track_ids: tensor of shape [batch_size]
        Returns:
            embeddings: tensor of shape [batch_size, L*n_stage, embedding_size]
        """
        track_ids = interaction['tracks_id']
        track_indices = track_ids
        
        # 获取编码 [batch_size, L, n_stage]
        cluster_ids = self.track_to_cluster_map[track_indices]  # [batch_size, L, n_stage]
        
        # 获取所有层所有stage的embeddings
        embeddings = []
        for layer in range(self.L):
            for stage in range(self.n_stage):
                # 使用对应层的embedding table
                emb = self.embedding_tables[layer][stage](cluster_ids[:, layer, stage])  # [batch_size, embedding_size]
                embeddings.append(emb)
        
        # print(torch.stack(embeddings, dim=1).shape)
        return torch.stack(embeddings, dim=1)  # [batch_size, L*n_stage, embedding_size]