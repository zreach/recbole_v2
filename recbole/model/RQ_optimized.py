import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import faiss  # 如果可用的话，用于快速近邻搜索
from collections import Counter

class OptimizedRQKMeans:
    def __init__(self, n_stages=2, n_clusters=256, max_iter=100, random_state=42, 
                 use_gpu=False, batch_size=1000, use_faiss=True):
        """
        优化的RQ-Kmeans实现
        
        Args:
            n_stages: 量化阶段数
            n_clusters: 每个阶段的聚类数（码本大小）
            max_iter: kmeans最大迭代次数
            random_state: 随机种子
            use_gpu: 是否使用GPU（需要torch CUDA支持）
            batch_size: 批处理大小
            use_faiss: 是否使用FAISS进行快速搜索
        """
        self.n_stages = n_stages
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.use_faiss = use_faiss
        
        self.codebooks = []
        self.faiss_indexes = [] if use_faiss else None
        self.is_fitted = False
        
        # 统计信息
        self.stage_usage_stats = []  # 每个阶段的码本使用统计
        self.collision_stats = {}    # 碰撞统计
        
        # 设备选择
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
    def _build_faiss_index(self, codebook):
        """为码本构建FAISS索引"""
        if not self.use_faiss:
            return None
            
        try:
            import faiss
            d = codebook.shape[1]
            index = faiss.IndexFlatIP(d)  # 内积索引，速度更快
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # 归一化码本用于内积搜索
            codebook_norm = codebook / (np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-8)
            index.add(codebook_norm.astype(np.float32))
            return index, codebook_norm
        except ImportError:
            return None
        
    def fit(self, X):
        """
        训练RQ-Kmeans
        
        Args:
            X: 输入向量，shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        
        self.codebooks = []
        self.faiss_indexes = [] if self.use_faiss else None
        self.stage_usage_stats = []
        
        # 转换为torch张量以利用GPU加速
        if self.use_gpu:
            residual = torch.from_numpy(X).to(self.device)
        else:
            residual = X.copy()
            
        for stage in range(self.n_stages):
            print(f"Training stage {stage + 1}/{self.n_stages}...")
            
            # 转换回numpy进行kmeans训练（sklearn目前不支持GPU）
            if self.use_gpu:
                residual_np = residual.cpu().numpy()
            else:
                residual_np = residual
                
            # 使用MiniBatchKMeans加速大数据集训练
            if n_samples > 10000:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    random_state=self.random_state + stage,
                    batch_size=min(self.batch_size, n_samples // 10),
                    n_init=3  # 减少初始化次数
                )
            else:
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    random_state=self.random_state + stage,
                    n_init=5
                )
            
            # 训练kmeans
            cluster_labels = kmeans.fit_predict(residual_np)
            codebook = kmeans.cluster_centers_.astype(np.float32)
            
            # 存储码本
            self.codebooks.append(codebook)
            
            # 统计训练时的码本使用情况
            usage_counter = Counter(cluster_labels)
            stage_stats = {
                'total_codes': self.n_clusters,
                'used_codes': len(usage_counter),
                'usage_ratio': len(usage_counter) / self.n_clusters,
                'usage_distribution': dict(usage_counter),
                'max_usage': max(usage_counter.values()),
                'min_usage': min(usage_counter.values()),
                'avg_usage': np.mean(list(usage_counter.values())),
                'std_usage': np.std(list(usage_counter.values()))
            }
            self.stage_usage_stats.append(stage_stats)
            
            # 构建FAISS索引
            if self.use_faiss:
                faiss_result = self._build_faiss_index(codebook)
                self.faiss_indexes.append(faiss_result)
            
            # 计算量化向量并更新残差
            if self.use_gpu:
                codebook_tensor = torch.from_numpy(codebook).to(self.device)
                quantized = codebook_tensor[cluster_labels]
                residual = residual - quantized
            else:
                quantized = codebook[cluster_labels]
                residual = residual - quantized
                
        self.is_fitted = True
        print("Training completed!")
        
    def encode_batch(self, X, collect_stats=True):
        """
        批量编码向量
        
        Args:
            X: 输入向量，shape (n_samples, n_features)
            collect_stats: 是否收集统计信息
            
        Returns:
            codes: 量化码，shape (n_samples, n_stages)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
            
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.n_stages), dtype=np.int32)
        
        if self.use_gpu:
            residual = torch.from_numpy(X).to(self.device)
        else:
            residual = X.copy()
        
        # 统计信息收集
        stage_code_usage = []
        
        for stage in range(self.n_stages):
            if self.use_faiss and self.faiss_indexes[stage] is not None:
                # 使用FAISS进行快速搜索
                index, codebook_norm = self.faiss_indexes[stage]
                if self.use_gpu:
                    query = residual.cpu().numpy()
                else:
                    query = residual
                    
                # 归一化查询向量
                query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
                _, cluster_ids = index.search(query_norm.astype(np.float32), 1)
                cluster_ids = cluster_ids.flatten()
            else:
                # 使用距离计算找最近聚类
                if self.use_gpu:
                    residual_np = residual.cpu().numpy()
                else:
                    residual_np = residual
                    
                distances = cdist(residual_np, self.codebooks[stage])
                cluster_ids = np.argmin(distances, axis=1)
            
            codes[:, stage] = cluster_ids
            
            # 收集当前阶段的码本使用统计
            if collect_stats:
                stage_code_usage.append(cluster_ids)
            
            # 更新残差
            if self.use_gpu:
                codebook_tensor = torch.from_numpy(self.codebooks[stage]).to(self.device)
                quantized = codebook_tensor[cluster_ids]
                residual = residual - quantized
            else:
                quantized = self.codebooks[stage][cluster_ids]
                residual = residual - quantized
        
        # 更新统计信息
        if collect_stats:
            self._update_encoding_stats(codes, stage_code_usage)
                
        return codes
    
    def _update_encoding_stats(self, codes, stage_code_usage):
        """更新编码统计信息"""
        n_samples = codes.shape[0]
        
        # 计算码本组合碰撞率
        code_combinations = []
        for i in range(n_samples):
            combo = tuple(codes[i])
            code_combinations.append(combo)
        
        combo_counter = Counter(code_combinations)
        unique_combinations = len(combo_counter)
        collision_rate = 1.0 - (unique_combinations / n_samples)
        
        # 更新全局统计
        self.collision_stats = {
            'total_samples': n_samples,
            'unique_combinations': unique_combinations,
            'collision_rate': collision_rate,
            'most_common_combinations': combo_counter.most_common(10),
            'theoretical_max_combinations': self.get_codebook_size(),
            'combination_utilization': unique_combinations / self.get_codebook_size()
        }
        
        # 更新每个阶段的使用统计
        for stage, cluster_ids in enumerate(stage_code_usage):
            usage_counter = Counter(cluster_ids)
            self.stage_usage_stats[stage].update({
                'encoding_used_codes': len(usage_counter),
                'encoding_usage_ratio': len(usage_counter) / self.n_clusters,
                'encoding_usage_distribution': dict(usage_counter)
            })
    
    def get_collision_stats(self):
        """获取碰撞统计信息"""
        return self.collision_stats
    
    def get_codebook_usage_stats(self):
        """获取码本使用效率统计"""
        return self.stage_usage_stats
    
    def print_detailed_stats(self):
        """打印详细的统计信息"""
        print("\n" + "="*60)
        print("RQ-KMeans 详细统计报告")
        print("="*60)
        
        # 基本信息
        print(f"量化阶段数: {self.n_stages}")
        print(f"每阶段聚类数: {self.n_clusters}")
        print(f"理论最大组合数: {self.get_codebook_size():,}")
        
        # 码本使用效率
        print("\n码本使用效率分析:")
        print("-" * 40)
        for stage, stats in enumerate(self.stage_usage_stats):
            print(f"阶段 {stage + 1}:")
            print(f"  训练时使用的码本数: {stats['used_codes']}/{stats['total_codes']} "
                  f"({stats['usage_ratio']:.2%})")
            
            if 'encoding_used_codes' in stats:
                print(f"  编码时使用的码本数: {stats['encoding_used_codes']}/{stats['total_codes']} "
                      f"({stats['encoding_usage_ratio']:.2%})")
            
            print(f"  使用次数统计: 最大={stats['max_usage']}, 最小={stats['min_usage']}, "
                  f"平均={stats['avg_usage']:.1f}, 标准差={stats['std_usage']:.1f}")
        
        # 碰撞分析
        if self.collision_stats:
            print("\n碰撞分析:")
            print("-" * 40)
            print(f"总样本数: {self.collision_stats['total_samples']:,}")
            print(f"唯一组合数: {self.collision_stats['unique_combinations']:,}")
            print(f"碰撞率: {self.collision_stats['collision_rate']:.2%}")
            print(f"组合空间利用率: {self.collision_stats['combination_utilization']:.2%}")
            
            print("\n最常见的组合 (前5个):")
            for i, (combo, count) in enumerate(self.collision_stats['most_common_combinations'][:5]):
                print(f"  {i+1}. {combo}: {count} 次 ({count/self.collision_stats['total_samples']:.2%})")
    
    def analyze_codebook_balance(self):
        """分析码本负载均衡情况"""
        balance_analysis = {}
        
        for stage, stats in enumerate(self.stage_usage_stats):
            if 'encoding_usage_distribution' in stats:
                usage_counts = list(stats['encoding_usage_distribution'].values())
            else:
                usage_counts = list(stats['usage_distribution'].values())
            
            # 计算基尼系数衡量不平衡程度
            sorted_counts = sorted(usage_counts)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n
            
            # 计算变异系数
            cv = np.std(usage_counts) / np.mean(usage_counts)
            
            balance_analysis[f'stage_{stage + 1}'] = {
                'gini_coefficient': gini,
                'coefficient_of_variation': cv,
                'balance_score': 1.0 - gini,  # 平衡分数，越接近1越平衡
                'interpretation': 'balanced' if gini < 0.2 else 'moderately_unbalanced' if gini < 0.4 else 'highly_unbalanced'
            }
        
        return balance_analysis
    
    def encode(self, x):
        """单个向量编码（兼容原接口）"""
        codes = self.encode_batch(x.reshape(1, -1) if x.ndim == 1 else x, collect_stats=False)
        return codes[0].tolist() if codes.shape[0] == 1 else codes
    
    def decode_batch(self, codes):
        """
        批量解码量化码
        
        Args:
            codes: 量化码，shape (n_samples, n_stages)
            
        Returns:
            reconstructed: 重构向量，shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
            
        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
            
        n_samples = codes.shape[0]
        n_features = self.codebooks[0].shape[1]
        
        if self.use_gpu:
            reconstructed = torch.zeros(n_samples, n_features, device=self.device)
            for stage in range(self.n_stages):
                codebook_tensor = torch.from_numpy(self.codebooks[stage]).to(self.device)
                reconstructed += codebook_tensor[codes[:, stage]]
            return reconstructed.cpu().numpy()
        else:
            reconstructed = np.zeros((n_samples, n_features), dtype=np.float32)
            for stage in range(self.n_stages):
                reconstructed += self.codebooks[stage][codes[:, stage]]
            return reconstructed
    
    def decode(self, codes):
        """单个码解码（兼容原接口）"""
        if isinstance(codes, list):
            codes = np.array(codes)
        result = self.decode_batch(codes.reshape(1, -1) if codes.ndim == 1 else codes)
        return result[0] if result.shape[0] == 1 else result
    
    def get_codebook_size(self):
        """返回总码本大小"""
        return self.n_clusters ** self.n_stages
    
    def get_compression_ratio(self, original_dim):
        """计算压缩比"""
        original_bits = original_dim * 32
        compressed_bits = self.n_stages * np.log2(self.n_clusters)
        return original_bits / compressed_bits

# 保持原类名的兼容性
class RQKMeans(OptimizedRQKMeans):
    """向后兼容的原类名"""
    pass

# 性能测试和比较
if __name__ == "__main__":
    import time
    
    # 生成测试数据
    np.random.seed(42)
    n_samples, n_features = 10000, 768  # 更实际的维度
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    test_vectors = np.random.randn(5000, n_features).astype(np.float32)
    
    print("Testing optimized RQ-Kmeans with collision analysis...")
    
    # 测试优化版本
    start_time = time.time()
    rq_opt = OptimizedRQKMeans(n_stages=2, n_clusters=256, use_gpu=False, use_faiss=False)
    rq_opt.fit(X)
    fit_time = time.time() - start_time
    print(f"Optimized fit time: {fit_time:.2f}s")
    
    # 测试批量编码（带统计）
    start_time = time.time()
    codes_batch = rq_opt.encode_batch(test_vectors, collect_stats=True)
    encode_time = time.time() - start_time
    print(f"Batch encode time (5000 vectors): {encode_time:.4f}s")
    print(f"Average encode time per vector: {encode_time/5000*1000:.4f}ms")
    
    # 测试解码
    start_time = time.time()
    reconstructed_batch = rq_opt.decode_batch(codes_batch)
    decode_time = time.time() - start_time
    print(f"Batch decode time: {decode_time:.4f}s")
    
    # 计算重构误差
    error = np.mean(np.linalg.norm(test_vectors - reconstructed_batch, axis=1))
    print(f"Average reconstruction error: {error:.4f}")
    print(f"Compression ratio: {rq_opt.get_compression_ratio(n_features):.2f}x")
    
    # 打印详细统计
    rq_opt.print_detailed_stats()
    
    # 分析码本平衡性
    balance_analysis = rq_opt.analyze_codebook_balance()
    print("\n码本负载均衡分析:")
    print("-" * 40)
    for stage, analysis in balance_analysis.items():
        print(f"{stage}: 基尼系数={analysis['gini_coefficient']:.3f}, "
              f"平衡分数={analysis['balance_score']:.3f}, "
              f"状态={analysis['interpretation']}")