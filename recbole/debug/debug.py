import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
import numpy as np

class MLPLayers(nn.Module):
    """多层感知机层"""
    def __init__(self, layers, dropout=0.0, activation='relu', bn=False):
        super(MLPLayers, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.use_bn = bn
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if bn and i < len(layers) - 2:  # 最后一层不加BN
                self.layers.append(nn.BatchNorm1d(layers[i + 1]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                # 最后一层不加激活函数
                if i < len(self.layers) - 1:
                    if self.activation == 'relu':
                        x = F.relu(x)
                    elif self.activation == 'sigmoid':
                        x = torch.sigmoid(x)
                    elif self.activation == 'tanh':
                        x = torch.tanh(x)
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
        return x

class AFN(nn.Module):
    """AFN (Attentional Factorization Network) 完全自包含版本"""
    
    def __init__(self, config, num_features=None, feature_dims=None):
        super(AFN, self).__init__()
        
        # 基本参数
        self.embedding_size = config.get("embedding_size", 32)
        self.mlp_hidden_size = config.get("mlp_hidden_size", [128, 64])
        self.dropout_prob = config.get("dropout_prob", 0.2)
        self.logarithmic_neurons = config.get("logarithmic_neurons", 16)
        self.ensemble_dnn = config.get("ensemble_dnn", False)
        self.reg_weight = config.get("reg_weight", 1e-4)
        
        # 特征配置
        self.num_features = num_features or 8
        self.feature_dims = feature_dims or [1000] * self.num_features
        
        # 标签字段名
        self.LABEL = 'label'
        
        # 创建embedding层
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, self.embedding_size) 
            for dim in self.feature_dims
        ])
        
        # AFN特有的层
        self.num_feature_field = len(self.feature_dims)
        
        # MLP层配置
        size_list = [
            self.logarithmic_neurons * self.embedding_size
        ] + self.mlp_hidden_size + [1]
        
        self.dense_layers = MLPLayers(
            size_list, self.dropout_prob, activation='relu', bn=True
        )
        
        # AFN核心组件
        self.coefficient_W = nn.Linear(
            self.num_feature_field, self.logarithmic_neurons, bias=False
        )
        
        self.log_batch_norm = nn.BatchNorm1d(self.num_feature_field)
        self.exp_batch_norm = nn.BatchNorm1d(self.logarithmic_neurons)
        
        # 激活函数和损失函数
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                xavier_normal_(module.weight.data)
            elif isinstance(module, nn.Linear):
                xavier_normal_(module.weight.data)
                if module.bias is not None:
                    constant_(module.bias.data, 0)
    
    def concat_embed_input_fields(self, interaction):
        """将所有特征字段的embedding拼接"""
        embeddings = []
        
        # 获取所有特征字段
        feature_names = [f'field_{i}' for i in range(self.num_features)]
        
        for i, field_name in enumerate(feature_names):
            if field_name in interaction:
                # 获取特征值并进行embedding
                feature_values = interaction[field_name]
                emb = self.embeddings[i](feature_values)
                embeddings.append(emb)
        
        # 如果没有足够的特征，用零填充
        while len(embeddings) < self.num_feature_field:
            batch_size = embeddings[0].size(0) if embeddings else 1
            zero_emb = torch.zeros(batch_size, self.embedding_size)
            embeddings.append(zero_emb)
        
        # 拼接所有embedding [batch_size, num_field, embed_dim]
        concat_embeddings = torch.stack(embeddings, dim=1)
        return concat_embeddings
    
    def log_net(self, emb):
        """AFN的对数网络部分"""
        # emb: [batch_size, num_field, embed_dim]
        emb = torch.abs(emb)
        emb = torch.clamp(emb, min=1e-5)
        log_emb = torch.log(emb)
        log_emb = self.log_batch_norm(log_emb)
        log_out = self.coefficient_W(log_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(log_out)
        cross_out = self.exp_batch_norm(cross_out)
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out
    
    def forward(self, interaction):
        """前向传播"""
        embeddings = self.concat_embed_input_fields(interaction)
        dnn_input = self.log_net(embeddings)
        afn_out = self.dense_layers(dnn_input)
        return afn_out.squeeze(-1)
    
    def reg_emb_loss(self):
        """计算embedding正则化损失"""
        reg_loss = 0
        for embedding in self.embeddings:
            reg_loss += torch.norm(embedding.weight, p=2)
        return self.reg_weight * reg_loss
    
    def calculate_loss(self, interaction):
        """计算总损失"""
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        bce_loss = self.loss_fn(output, label)
        total_loss = bce_loss + self.reg_emb_loss()
        print(f"BCE Loss: {bce_loss.item():.4f}, Reg Loss: {self.reg_emb_loss().item():.6f}")
        return total_loss
    
    def predict(self, interaction):
        """预测"""
        result = self.forward(interaction)
        probs = torch.sigmoid(result)
        print(f'Logits - max: {torch.max(result).item():.4f}, avg: {torch.mean(result).item():.4f}')
        print(f'Probs - max: {torch.max(probs).item():.4f}, avg: {torch.mean(probs).item():.4f}')
        return result

# 生成k对正负样本的训练数据函数
def generate_k_pairs_data(k=5, num_features=8, vocab_sizes=None):
    """生成k对正负样本"""
    if vocab_sizes is None:
        vocab_sizes = [500] * num_features
    
    batch_size = 2 * k  # k个正例 + k个负例
    interaction = {}
    
    # 生成特征数据
    for i in range(num_features):
        field_name = f'field_{i}'
        feature_values = []
        
        # 生成k个正例
        for j in range(k):
            if i < num_features // 2:
                # 前半部分特征：正例用较大值范围
                pos_val = torch.randint(vocab_sizes[i]//2, vocab_sizes[i], (1,))
            else:
                # 后半部分特征：正例用中高值范围
                pos_val = torch.randint(vocab_sizes[i]//3, vocab_sizes[i], (1,))
            feature_values.append(pos_val)
        
        # 生成k个负例
        for j in range(k):
            if i < num_features // 2:
                # 前半部分特征：负例用较小值范围
                neg_val = torch.randint(0, vocab_sizes[i]//2, (1,))
            else:
                # 后半部分特征：负例用低值范围
                neg_val = torch.randint(0, vocab_sizes[i]//3, (1,))
            feature_values.append(neg_val)
        
        interaction[field_name] = torch.cat(feature_values)
    
    # 标签：前k个为正例(1.0)，后k个为负例(0.0)
    labels = torch.cat([
        torch.ones(k),   # k个正例
        torch.zeros(k)   # k个负例
    ])
    interaction['label'] = labels
    
    return interaction

# 批量训练函数
def train_afn_with_batches(model, k_pairs=5, num_features=8, vocab_sizes=None, num_epochs=50):
    """使用k对样本进行批量训练"""
    if vocab_sizes is None:
        vocab_sizes = [500] * num_features
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("=" * 60)
    print(f"开始训练AFN模型 - 使用{k_pairs}对正负样本")
    print("=" * 60)
    
    # 生成训练数据
    training_data = generate_k_pairs_data(k_pairs, num_features, vocab_sizes)
    
    print(f"训练数据概况:")
    print(f"  总样本数: {2 * k_pairs} (正例: {k_pairs}, 负例: {k_pairs})")
    print(f"  特征数量: {num_features}")
    
    # 显示部分训练数据
    print(f"\n前3个样本的特征值:")
    for i in range(min(3, 2 * k_pairs)):
        sample_features = []
        for j in range(num_features):
            field_name = f'field_{j}'
            sample_features.append(training_data[field_name][i].item())
        label = training_data['label'][i].item()
        print(f"  样本{i+1} (标签:{label}): {sample_features}")
    
    print("\n" + "-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播和损失计算
        loss = model.calculate_loss(training_data)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}, Total Loss: {loss.item():.4f}")
        
        # 每10个epoch预测一次
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model.predict(training_data)
                probs = torch.sigmoid(predictions)
                
                # 计算正例和负例的平均预测概率
                pos_probs = probs[:k_pairs].mean().item()
                neg_probs = probs[k_pairs:].mean().item()
                
                print(f"  平均预测概率: 正例={pos_probs:.4f}, 负例={neg_probs:.4f}")
                print(f"  概率差距: {pos_probs - neg_probs:.4f}")
                print("-" * 40)
    
    return training_data

# 评估函数
def evaluate_model(model, test_data, k_pairs):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        predictions = model.predict(test_data)
        probs = torch.sigmoid(predictions)
        
        # 分离正例和负例的预测
        pos_probs = probs[:k_pairs]
        neg_probs = probs[k_pairs:]
        
        print("\n" + "=" * 60)
        print("模型评估结果:")
        print("=" * 60)
        
        print(f"正例预测概率:")
        for i, prob in enumerate(pos_probs):
            print(f"  正例{i+1}: {prob.item():.4f}")
        
        print(f"\n负例预测概率:")
        for i, prob in enumerate(neg_probs):
            print(f"  负例{i+1}: {prob.item():.4f}")
        
        # 计算统计指标
        pos_mean = pos_probs.mean().item()
        neg_mean = neg_probs.mean().item()
        pos_std = pos_probs.std().item()
        neg_std = neg_probs.std().item()
        
        print(f"\n统计指标:")
        print(f"  正例: 均值={pos_mean:.4f}, 标准差={pos_std:.4f}")
        print(f"  负例: 均值={neg_mean:.4f}, 标准差={neg_std:.4f}")
        print(f"  分离度: {pos_mean - neg_mean:.4f}")
        
        # 计算准确率（简单阈值0.5）
        pos_correct = (pos_probs > 0.5).sum().item()
        neg_correct = (neg_probs <= 0.5).sum().item()
        accuracy = (pos_correct + neg_correct) / (2 * k_pairs)
        
        print(f"  准确率 (阈值0.5): {accuracy:.4f} ({pos_correct + neg_correct}/{2 * k_pairs})")

# 主函数
if __name__ == "__main__":
    # 配置参数
    config = {
        'embedding_size': 32,
        'mlp_hidden_size': [128, 64],
        'dropout_prob': 0.1,
        'logarithmic_neurons': 16,
        'ensemble_dnn': False,
        'reg_weight': 1e-5
    }
    
    # 特征配置
    num_features = 6
    feature_dims = [300, 200, 150, 400, 250, 350]  # 每个特征的词汇表大小
    k_pairs = 10000  # 生成8对正负样本
    
    # 创建模型
    model = AFN(config, num_features=num_features, feature_dims=feature_dims)
    print(f"模型创建成功！参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 开始训练
    training_data = train_afn_with_batches(
        model, 
        k_pairs=k_pairs, 
        num_features=num_features, 
        vocab_sizes=feature_dims, 
        num_epochs=100
    )
    
    # 评估模型
    evaluate_model(model, training_data, k_pairs)
    
    print("\n" + "=" * 60)
    print("训练和评估完成！")
    
    # 生成新的测试数据进行验证
    # print("\n测试新样本:")
    # test_data = generate_k_pairs_data(3, num_features, feature_dims)
    # evaluate_model(model, test_data, 3)