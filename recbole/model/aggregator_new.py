import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioAttentionAggregator(nn.Module):
    """音频特征注意力聚合器 使用Wq和Wk矩阵计算层级注意力权重"""
    
    def __init__(self, embedding_size, config, feature_dict, token2id, **kwargs):
        super(AudioAttentionAggregator, self).__init__()
        
        self.embedding_size = embedding_size
        self.config = config
        self.feature_dict = feature_dict
        self.token2id = token2id
        self.device = config.get('device', 'cuda')
        
        # 分析音频特征结构 - 适配 (L, H) 格式
        self._analyze_feature_structure()
        
        # 学习的Wq和Wk矩阵
        self.attention_dim = config.get('attention_dim', embedding_size)
        
        # 为每一层创建Wq和Wk矩阵
        self.Wq_layers = nn.ModuleList([
            nn.Linear(self.layer_feature_dim, self.attention_dim, bias=False)
            for _ in range(self.num_layers)
        ])
        
        self.Wk_layers = nn.ModuleList([
            nn.Linear(self.layer_feature_dim, self.attention_dim, bias=False)
            for _ in range(self.num_layers)
        ])
        
        # 温度参数，用于控制注意力分布的锐度
        self.temperature = config.get('attention_temperature', 1.0)
        
        # 最终的MLP，将加权后的特征映射到embedding_size
        mlp_input_dim = self.num_layers * self.layer_feature_dim
        mlp_hidden_sizes = config.get('audio_mlp_sizes', [256, 128])
        
        mlp_layers = []
        input_dim = mlp_input_dim
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.get('audio_dropout', 0.2))
            ])
            input_dim = hidden_size
        
        # 最后一层映射到embedding_size
        mlp_layers.append(nn.Linear(input_dim, embedding_size))
        self.final_mlp = nn.Sequential(*mlp_layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _analyze_feature_structure(self):
        """分析音频特征结构 - 适配 (L, H) 格式"""
        # 获取一个样本特征来分析结构
        sample_item_id = next(iter(self.feature_dict.keys()))
        sample_feature = self.feature_dict[sample_item_id]  # (L, H)
        
        if isinstance(sample_feature, torch.Tensor):
            # 特征格式为 (L, H)
            self.num_layers = sample_feature.size(0)  # L
            self.layer_feature_dim = sample_feature.size(1)  # H
        elif isinstance(sample_feature, (list, tuple)) and len(sample_feature) == 2:
            # 如果是 numpy array 或其他格式
            sample_tensor = torch.tensor(sample_feature) if not isinstance(sample_feature, torch.Tensor) else sample_feature
            self.num_layers = sample_tensor.size(0)  # L
            self.layer_feature_dim = sample_tensor.size(1)  # H
        else:
            raise ValueError(f"Unsupported feature format: {type(sample_feature)}, shape: {sample_feature.shape if hasattr(sample_feature, 'shape') else 'unknown'}")
        
        # 创建层名列表
        self.layer_names = [f'layer_{i}' for i in range(self.num_layers)]
        
        print(f"Audio feature structure: {self.num_layers} layers, {self.layer_feature_dim} dimensions per layer")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def get_audio_features(self, item_ids):
        """获取item的原始音频特征 - 适配 (L, H) 格式"""
        batch_size = len(item_ids)
        # 初始化特征字典，每层一个key
        layer_features = {f'layer_{i}': [] for i in range(self.num_layers)}
        
        for item_id in item_ids:
            item_id = item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            
            if item_id in self.feature_dict:
                features = self.feature_dict[item_id]  # (L, H)
                
                # 转换为tensor如果需要
                if not isinstance(features, torch.Tensor):
                    features = torch.tensor(features, dtype=torch.float32)
                
                # 确保特征形状正确
                if features.dim() == 2 and features.size(0) == self.num_layers:
                    # 分解每一层的特征
                    for layer_idx in range(self.num_layers):
                        layer_feat = features[layer_idx]  # (H,)
                        layer_features[f'layer_{layer_idx}'].append(layer_feat)
                else:
                    # 如果形状不对，用零向量填充
                    for layer_idx in range(self.num_layers):
                        layer_features[f'layer_{layer_idx}'].append(
                            torch.zeros(self.layer_feature_dim, dtype=torch.float32)
                        )
            else:
                # 如果item不存在，用零向量填充所有层
                for layer_idx in range(self.num_layers):
                    layer_features[f'layer_{layer_idx}'].append(
                        torch.zeros(self.layer_feature_dim, dtype=torch.float32)
                    )
        
        # 转换为tensor并移到正确的设备
        for layer_name in layer_features:
            layer_features[layer_name] = torch.stack(layer_features[layer_name]).to(self.device)  # [batch_size, layer_feature_dim]
        
        return layer_features
    
    def compute_attention_weights(self, target_features, history_features):
        """
        计算注意力权重
        target_features: dict, 每层的target item特征 {layer_i: [1, layer_feature_dim]}
        history_features: dict, 每层的history items特征 {layer_i: [seq_len, layer_feature_dim]}
        返回: [num_layers, seq_len] 每一层对应每个history item的注意力权重
        """
        seq_len = next(iter(history_features.values())).size(0)
        attention_weights = []
        
        for i in range(self.num_layers):
            layer_name = f'layer_{i}'
            
            # 获取当前层的特征
            target_feat = target_features[layer_name]  # [1, layer_feature_dim]
            history_feat = history_features[layer_name]  # [seq_len, layer_feature_dim]
            
            # 计算query和key
            q = self.Wq_layers[i](target_feat)  # [1, attention_dim]
            k = self.Wk_layers[i](history_feat)  # [seq_len, attention_dim]
            
            # 计算注意力分数
            scores = torch.matmul(q, k.transpose(0, 1)) / self.temperature  # [1, seq_len]
            scores = scores.squeeze(0)  # [seq_len]
            
            # 应用softmax得到注意力权重
            weights = F.softmax(scores, dim=0)  # [seq_len]
            attention_weights.append(weights)
        
        return torch.stack(attention_weights)  # [num_layers, seq_len]
    
    def forward(self, interaction, target_item_id=None, history_item_ids=None):
        """
        前向传播
        interaction: 包含item信息的字典
        target_item_id: 目标item的ID（可选，用于注意力计算）
        history_item_ids: 历史item的ID列表（可选，用于注意力计算）
        """
        if 'tracks_id' in interaction:
            item_ids = interaction['tracks_id']
            if isinstance(item_ids, torch.Tensor):
                item_ids = item_ids.cpu().tolist()
            elif not isinstance(item_ids, list):
                item_ids = [item_ids]
        else:
            raise ValueError("No tracks_id found in interaction")
        
        # 获取音频特征
        layer_features = self.get_audio_features(item_ids)
        
        # 如果提供了target和history，使用注意力机制
        if target_item_id is not None and history_item_ids is not None:
            # 获取target item的特征
            target_features = self.get_audio_features([target_item_id])
            
            # 获取history items的特征
            history_features = self.get_audio_features(history_item_ids)
            
            # 计算注意力权重
            attention_weights = self.compute_attention_weights(target_features, history_features)
            
            # 使用注意力权重对history特征进行加权
            weighted_features = []
            for i in range(self.num_layers):
                layer_name = f'layer_{i}'
                history_feat = history_features[layer_name]  # [seq_len, layer_feature_dim]
                weights = attention_weights[i].unsqueeze(1)  # [seq_len, 1]
                weighted_feat = torch.sum(history_feat * weights, dim=0, keepdim=True)  # [1, layer_feature_dim]
                weighted_features.append(weighted_feat)
            
            # 拼接所有层的加权特征
            concatenated_features = torch.cat(weighted_features, dim=1)  # [1, num_layers * layer_feature_dim]
        else:
            # 没有注意力机制时，直接拼接所有层的特征
            batch_size = len(item_ids)
            concatenated_features = []
            
            for i in range(batch_size):
                item_features = []
                for layer_idx in range(self.num_layers):
                    layer_name = f'layer_{layer_idx}'
                    item_features.append(layer_features[layer_name][i])  # [layer_feature_dim]
                
                item_concat = torch.cat(item_features, dim=0)  # [num_layers * layer_feature_dim]
                concatenated_features.append(item_concat)
            
            concatenated_features = torch.stack(concatenated_features)  # [batch_size, num_layers * layer_feature_dim]
        
        # 通过最终的MLP得到embedding
        final_embeddings = self.final_mlp(concatenated_features)  # [batch_size, embedding_size]
        
        # 增加一个维度以匹配RecBole的期望格式
        final_embeddings = final_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_size]
        
        return final_embeddings