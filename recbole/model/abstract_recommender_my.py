# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/16, 2020/8/6, 2020/8/25, 2023/4/24
# @Author : Zhen Tian, Shanlei Mu, Yupeng Hou, Chenglong Ma
# @Email  : chenyuwuxinn@gmail.com, slmu@ruc.edu.cn, houyupeng@ruc.edu.cn, chenglong.m@outlook.com

"""
recbole.model.abstract_recommender
##################################
"""

from logging import getLogger

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
from recbole.utils import ModelType, InputType, FeatureSource, FeatureType, set_color
from recbole.model.loss import RegLoss

from tqdm import tqdm

class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def __init__(self):
        self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )


# class GeneralRecommender(AbstractRecommender):
#     """This is a abstract general recommender. All the general model should implement this class.
#     The base general recommender class provide the basic dataset and parameters information.
#     """

#     type = ModelType.GENERAL

#     def __init__(self, config, dataset):
#         super(GeneralRecommender, self).__init__()

#         # load dataset info
#         self.USER_ID = config["USER_ID_FIELD"]
#         self.ITEM_ID = config["ITEM_ID_FIELD"]
#         self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
#         self.n_users = dataset.num(self.USER_ID)
#         self.n_items = dataset.num(self.ITEM_ID)

#         # load parameters info
#         self.device = config["device"]

class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]
        
        # multimodal
        self.token2id = dataset.field2token_id
        self.id2token = {}
        self.use_cb = config['use_cb']

        self.use_audio = config['use_audio']
        self.use_text = config['use_text']
        self.embedding_size = config["embedding_size"]

        self.a_feats = None 
        self.t_feats = None
        self.id2afeats = None 
        self.id2tfeats = None


        if self.use_cb:
            if self.use_sem_id:
                pass
            if self.use_audio:
                a_feature_path = config['a_feature_path']
                with open(a_feature_path, 'rb') as fp:
                    music_features_array = pickle.load(fp)
                
                self.wav_embedding_size = list(music_features_array.values())[0].shape[-1]
                music_features_array['[PAD]'] = np.zeros((self.wav_embedding_size))
                music_features = torch.zeros((len(self.token2id['tracks_id']), self.wav_embedding_size ))

            if self.use_text:
                t_feature_path = config['t_feature_path']
                with open(t_feature_path, 'rb') as fp:
                    text_features_array = pickle.load(fp)

                self.text_embedding_size = list(text_features_array.values())[0].shape[-1]
                text_features_array['[PAD]'] = np.zeros((self.text_embedding_size))
                text_features = torch.zeros((len(self.token2id['tracks_id']), self.text_embedding_size ))
            
            # if config['norm_audio']:
            #     for k, v in music_features_array.items():
            #         norm = np.linalg.norm(v, axis=1, keepdims=True)
            #         norm[norm == 0] = 1e-12
            #         music_features_array[k] = v / norm

            if config['dataset'] in [ 'm4a-fil']:
                
                map_path = config['map_path']
                with open(map_path, 'rb') as fp:
                    self.id2msd = pickle.load(fp)
                self.id2msd = {str(k): v for k, v in self.id2msd.items()}
                self.id2msd['[PAD]'] = '[PAD]'
                for k, v in self.token2id['tracks_id'].items():
                    # if config['dataset'] == 'm4a-fil':
                    #     k = str(k)
                    k = self.id2msd[k]
                    self.id2token[v] = k
                    if k == '[PAD]':
                        if self.use_audio:
                            wav_feature = np.zeros((self.wav_embedding_size))
                        if self.use_text:
                            text_feature = np.zeros((self.text_embedding_size))
                    else:
                        if self.use_text:
                            text_feature = text_features_array[k]
                        if self.use_audio:
                            layer = config['afeat_layer']
                            
                            if layer is not None:
                                wav_feature = music_features_array[k][layer]
                            else:
                                wav_feature = music_features_array[k]
                    # print('layer', layer)
                    if self.use_audio:
                        music_features[v] = torch.Tensor(wav_feature)
                    if self.use_text:
                        text_features[v] = torch.Tensor(text_feature)
            elif config['dataset'] in ['m4a', 'm4a-seq', 'lfm2b-fil', 'lfm1b-fil',]: # 这个数据没有时间维度， 而且不需要map
                for k, v in self.token2id['tracks_id'].items():
                    k = str(k)
                    self.id2token[v] = k
                    if k == '[PAD]':
                        if self.use_audio:
                            wav_feature = np.zeros((self.wav_embedding_size))
                        if self.use_text:
                            text_feature = np.zeros((self.text_embedding_size))
                    else:
                        if self.use_text:
                            if k in text_features_array: 
                                text_feature = text_features_array[k]
                            else:
                                print(1)
                                text_feature = np.zeros((self.text_embedding_size))
                        if self.use_audio:
                            layer = config['afeat_layer']
                            
                            if layer is not None:
                                if layer == 'mean':
                                    wav_feature = np.mean(music_features_array[k], axis=0)
                                else:
                                    wav_feature = music_features_array[k][layer]
                            else:
                                wav_feature = music_features_array[k]
                    # print('layer', layer)
                    if self.use_audio:
                        music_features[v] = torch.Tensor(wav_feature)
                    if self.use_text:
                        text_features[v] = torch.Tensor(text_feature)
            # music_features = torch.load('/user/zhouyz/rec/myRec/wav2feature.pt')
            if self.use_audio:
                self.a_feats = music_features
                self.id2afeats = nn.Embedding.from_pretrained(music_features)
                self.id2afeats.requires_grad_(False)
                if self.embedding_size is not None:
                    size_list = [
                        self.wav_embedding_size 
                    ] + config['wav_mlp_sizes'] + [self.embedding_size]
                    self.wav_mlp = MLPLayers(size_list, 0.2, bn=True)
                

            if self.use_text:
                self.t_feats = text_features
                self.id2tfeats = nn.Embedding.from_pretrained(text_features)
                self.id2tfeats.requires_grad_(False)
                if self.embedding_size is not None:
                    size_list = [
                        self.text_embedding_size 
                    ] + config['text_mlp_sizes'] + [self.embedding_size]
                    self.text_mlp = MLPLayers(size_list, 0.2, bn=True)


    
    def get_wav_embedding(self, track_ids):

        wav_features = self.id2afeats(track_ids)
        # print(wav_features[0])
        embed_features = self.wav_mlp(wav_features)

        return embed_features

    def get_text_embedding(self, track_ids):
        text_features = self.id2tfeats(track_ids)
        # print(text_features[0])
        embed_features = self.text_mlp(text_features)

        return embed_features

class AutoEncoderMixin(object):
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1, device=self.device).repeat(
            user.shape[0], self.n_items
        )
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten()
        )
        return rating_matrix


class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """

    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class KnowledgeRecommender(AbstractRecommender):
    """This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
    The base knowledge-based recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.KNOWLEDGE

    def __init__(self, config, dataset):
        super(KnowledgeRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.ENTITY_ID = config["ENTITY_ID_FIELD"]
        self.RELATION_ID = config["RELATION_ID_FIELD"]
        self.HEAD_ENTITY_ID = config["HEAD_ENTITY_ID_FIELD"]
        self.TAIL_ENTITY_ID = config["TAIL_ENTITY_ID_FIELD"]
        self.NEG_TAIL_ENTITY_ID = config["NEG_PREFIX"] + self.TAIL_ENTITY_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)

        # load parameters info
        self.device = config["device"]


class Aggregator(nn.Module):
    def __init__(self, embedding_size, token2id, feature_dict, config, proj_method="linear", layer=-1, n_clusters=2, n_stage=2, mlp_dropout=0.2, mlp_size_list=None):
        super().__init__()
        self.layer = layer
        self.proj_method = proj_method
        self.embedding_size = embedding_size
        self.token2id = token2id
        self.feature_dict = feature_dict
        self.has_time = False
        feature_shape = list(feature_dict.values())[0].shape
        self.config = config
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

        if layer == 'weighted_sum' or proj_method in ['transformer', 'moe']:
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

        if proj_method in ['mlp', 'linear', 'transformer', 'moe']:
            self.num_feature_filed = 1
        elif proj_method in ['cluster']:
            self.num_feature_filed = self.L
        elif proj_method in ['rq-kmeans']:
            self.num_feature_filed = n_stage * self.L
        # 线性聚合参数
        if proj_method == 'linear':
            self.net = nn.Linear(H, embedding_size, bias=True)
        # MLP聚合参数
        elif proj_method == 'mlp':
            size_list = [
                H
            ] + mlp_size_list + [self.embedding_size]
            self.net = MLPLayers(size_list, mlp_dropout)

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

        elif proj_method == 'moe':
            self.gate_type = config['moe_gate_type'] if 'moe_gate_type' in config else 'user'
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
                gate_input_dim = H + embedding_size
            elif self.gate_type == 'user':
                gate_input_dim = embedding_size
            elif self.gate_type == 'item':
                gate_input_dim = H
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, gate_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(gate_input_dim // 2, L),
                nn.Softmax(dim=-1)
            )

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
            return self.get_cluster_embeddings(interaction['tracks_id'])
        elif self.proj_method == 'rq-kmeans':
            return self.get_rq_cluster_embeddings(interaction['tracks_id'])
        elif self.proj_method == 'transformer':
            return self.get_transformer_embeddings(interaction['tracks_id'])
        elif self.proj_method == 'moe':
            return self.get_moe_embeddings(interaction['tracks_id'])
        else:
            raise ValueError(f"Unknown aggregation method: {self.proj_method}")

    def get_transformer_embeddings(self, track_ids):
        # track_ids = interaction['tracks_id']
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
    def get_moe_embeddings(self, track_ids):
        all_features = self.id2feats(track_ids)  # [batch_size, L*H]
        batch_size = all_features.size(0)
        L = self.L
        H = self.feature_dim
        features = all_features.view(batch_size, L, H)  # [batch_size, L, H]
        
        # 计算每层特征的平均值作为门控网络的输入
        # 可以使用不同的策略，这里使用所有层的平均
        gate_input = features.mean(dim=1)  # [batch_size, H]
        
        # 通过门控网络计算每个专家的权重
        gate_scores = self.gate_network(gate_input)  # [batch_size, L]
        
        # 应用temperature
        if self.temperature != 1.0:
            gate_scores = gate_scores / self.temperature
            gate_scores = torch.softmax(gate_scores, dim=-1)
        
        if self.use_hard_selection:
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
    def get_cluster_embeddings(self, track_ids):
        """
        获取track_ids对应的L个cluster embeddings
        Args:
            track_ids: tensor of shape [batch_size] (这些是token2id['tracks_id']中的索引值)
        Returns:
            embeddings: tensor of shape [batch_size, L, embedding_size]
        """
        # track_ids已经是token2id['tracks_id']中的索引，直接使用
        # 但需要减去padding_idx(通常是0)来对齐到我们的映射表
        track_indices = track_ids  # 假设padding_idx=0，实际track索引从1开始
        
        # 获取聚类ID [batch_size, L]
        cluster_ids = self.track_to_cluster_map[track_indices]
        
        # 批量获取embeddings
        embeddings = []
        for l in range(self.L):
            emb = self.embedding_tables[l](cluster_ids[:, l])  # [batch_size, embedding_size]
            embeddings.append(emb)
        
        
        return torch.stack(embeddings, dim=1)  # [batch_size, L, embedding_size]
    
    def get_rq_cluster_embeddings(self, track_ids):
        """
        获取track_ids对应的RQ编码embeddings
        Args:
            track_ids: tensor of shape [batch_size]
        Returns:
            embeddings: tensor of shape [batch_size, L*n_stage, embedding_size]
        """
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
class ContextRecommender(AbstractRecommender):
    """This is a abstract context-aware recommender. All the context-aware model should implement this class.
    The base context-aware recommender class provide the basic embedding function of feature fields which also
    contains a first-order part of feature fields.
    """

    type = ModelType.CONTEXT
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(ContextRecommender, self).__init__()

        if config['no_itemid'] is True:
            self.field_names = dataset.fields(
                source=[
                    FeatureSource.INTERACTION,
                    FeatureSource.USER,
                    FeatureSource.USER_ID,
                    FeatureSource.ITEM,
                    # FeatureSource.ITEM_ID, 
                ]
            )
        else:
            self.field_names = dataset.fields(
                source=[
                    FeatureSource.INTERACTION,
                    FeatureSource.USER,
                    FeatureSource.USER_ID,
                    FeatureSource.ITEM,
                    FeatureSource.ITEM_ID, 
                ]
            )
        # 如果
        self.config = config
        self.LABEL = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.double_tower = config["double_tower"]
        self.numerical_features = config["numerical_features"]
        if self.double_tower is None:
            self.double_tower = False
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.float_seq_field_names = []
        self.float_seq_field_dims = []
        self.num_feature_field = 0

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        
        self.token2id = dataset.field2token_id
        self.id2token = {}

        self.use_cb = config['use_cb'] if 'use_cb' in config else False

        self.use_audio = config['use_audio']
        self.use_text = config['use_text']

        if self.use_cb:
            
            if self.use_audio:
                a_feature_path = config['a_feature_path']
                with open(a_feature_path, 'rb') as fp:
                    music_features_array = pickle.load(fp)
                
                self.a_aggregator = Aggregator(
                    embedding_size = self.embedding_size,
                    proj_method = config['proj_method'],
                    token2id = self.token2id,
                    feature_dict = music_features_array,
                    config=config,
                    layer=config['afeat_layer'] if 'afeat_layer' in config else -1,
                    mlp_dropout= config['wav_dropout'] if 'wav_dropout' in config else 0.2,
                    mlp_size_list= config['wav_mlp_sizes'] if 'wav_mlp_sizes' in config else [512, 32],
                    n_clusters = config['n_clusters'] if 'n_clusters' in config else 16,
                    n_stage = config['n_stage'] if 'n_stage' in config else 2,
                )

            if self.use_text:
                t_feature_path = config['t_feature_path']
                with open(t_feature_path, 'rb') as fp:
                    text_features_array = pickle.load(fp)

                self.text_embedding_size = list(text_features_array.values())[0].shape[-1]
                text_features_array['[PAD]'] = np.zeros((self.text_embedding_size))
                text_features = torch.zeros((len(self.token2id['tracks_id']), self.text_embedding_size ))
            

            if self.use_text:
                self.t_feats = text_features
                self.id2tfeats = nn.Embedding.from_pretrained(text_features)
                self.id2tfeats.requires_grad_(False)
        
        if self.double_tower:
            self.user_field_names = dataset.fields(
                source=[FeatureSource.USER, FeatureSource.USER_ID]
            )
            self.item_field_names = dataset.fields(
                source=[FeatureSource.ITEM, FeatureSource.ITEM_ID]
            )
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += 1
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += 1

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT
                and field_name in self.numerical_features
            ):
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT_SEQ
                and field_name in self.numerical_features
            ):
                self.float_seq_field_names.append(field_name)
                self.float_seq_field_dims.append(dataset.num(field_name))
            else:
                continue

            self.num_feature_field += 1
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long
            )
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size
            )
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array(
                (0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.long
            )
            self.float_embedding_table = FLEmbedding(
                self.float_field_dims, self.float_field_offsets, self.embedding_size
            )
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, self.embedding_size)
                )
        if len(self.float_seq_field_dims) > 0:
            self.float_seq_embedding_table = nn.ModuleList()
            for float_seq_field_dim in self.float_seq_field_dims:
                self.float_seq_embedding_table.append(
                    nn.Embedding(float_seq_field_dim, self.embedding_size)
                )
        self.use_firstorder_mlp = config['use_firstorder_mlp']
        if self.use_firstorder_mlp:
            self.first_order_linear = FMFirstOrderLinear(config, dataset, id2afeats=self.id2afeats, id2tfeats=self.id2tfeats)
        else:
            self.first_order_linear = FMFirstOrderLinear(config, dataset)
        
        if self.use_audio:
            # wav_dropout = config['wav_dropout'] if 'wav_dropout' in config else 0.2
            # size_list = [
            #     self.wav_embedding_size 
            # ] + config['wav_mlp_sizes'] + [self.embedding_size]
            # self.wav_mlp = MLPLayers(size_list, wav_dropout, last_activation = None)
            # self.wav_fc = nn.Linear(1024, self.embedding_size)
            self.num_feature_field += self.a_aggregator.num_feature_filed
        if self.use_text:
            text_dropout = config['text_dropout'] if 'text_dropout' in config else 0.2
            size_list = [
                self.text_embedding_size 
            ] + config['text_mlp_sizes'] + [self.embedding_size]
            self.text_mlp = MLPLayers(size_list, text_dropout, last_activation = None )
            # self.text_fc = nn.Linear(1024, self.embedding_size)
            self.num_feature_field += 1

    def reg_emb_loss(self):
        # 先默认是给embedding的正则化
        reg_term = 0
        reg_pairs = [(2, self.config['reg_emb'])]
        for m_name, module in self.named_modules():
                if type(module) in [FMEmbedding, FLEmbedding]:
                    for p_name, param in module.named_parameters():
                        if param.requires_grad:
                            for emb_p, emb_lambda in reg_pairs:
                                reg_term += (emb_lambda) * torch.norm(param, emb_p) ** emb_p
        return reg_term
    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return None
        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(float_fields)

        return float_embedding

    def embed_float_seq_fields(self, float_seq_fields, mode="mean"):
        """Embed the float feature columns

        Args:
            float_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len, 2]
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            float_seq_embedding = base * embedding_table(
                index.long()
            )  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                float_seq_embedding
            )  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_float_seq_embedding = (
                    float_seq_embedding - (1 - mask) * 1e9
                )  # [batch_size, seq_len, embed_dim]
                result = torch.max(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1
                )  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result, dim=1
            )  # [batch_size, num_token_seq_field, embed_dim]

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode="mean"):
        """Embed the token feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(
                token_seq_field
            )  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                token_seq_embedding
            )  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_token_seq_embedding = (
                    token_seq_embedding - (1 - mask) * 1e9
                )  # [batch_size, seq_len, embed_dim]
                result = torch.max(
                    masked_token_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=1
                )  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result, dim=1
            )  # [batch_size, num_token_seq_field, embed_dim]

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.

        """
        if not self.double_tower:
            raise RuntimeError(
                "Please check your model hyper parameters and set 'double tower' as True"
            )
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = torch.split(
                dense_embedding,
                [self.user_float_field_num, self.item_float_field_num],
                dim=1,
            )
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [
                self.user_token_seq_field_num,
                self.item_token_seq_field_num,
                self.user_token_field_num,
                self.item_token_field_num,
            ]
            (
                first_token_seq_embedding,
                second_token_seq_embedding,
                first_token_embedding,
                second_token_embedding,
            ) = torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat(
                [first_token_seq_embedding, first_token_embedding], dim=1
            )
            second_sparse_embedding = torch.cat(
                [second_token_seq_embedding, second_token_embedding], dim=1
            )
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (
            first_sparse_embedding,
            first_dense_embedding,
            second_sparse_embedding,
            second_dense_embedding,
        )

    def get_wav_embedding(self, interaction):
        track_ids = interaction['tracks_id']
        wav_features = self.id2afeats(track_ids)
        # print(wav_features[0])
        embed_features = self.wav_mlp(wav_features)

        return embed_features.unsqueeze(1)

    def get_text_embedding(self, interaction):
        track_ids = interaction['tracks_id']
        text_features = self.id2tfeats(track_ids)
        # print(text_features[0])
        embed_features = self.text_mlp(text_features)

        return embed_features.unsqueeze(1)
    
    def concat_embed_input_fields(self, interaction):
        
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if self.use_audio:
            # wav_embedding = self.get_wav_embedding(interaction) 
            wav_embeddings = self.a_aggregator(interaction)
            # for emb in wav_embeddings:
            #     print(emb.shape)
            #     all_embeddings.append(emb)
            all_embeddings.append(wav_embeddings)
        if self.use_text:
            text_embedding = self.get_text_embedding(interaction)
            all_embeddings.append(text_embedding)
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        
        return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(
                float_fields, dim=1
            )  # [batch_size, num_float_field, 2]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])

        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)

        if float_fields_embedding is None:
            dense_embedding = float_seq_fields_embedding
        else:
            if float_seq_fields_embedding is None:
                dense_embedding = float_fields_embedding
            else:
                dense_embedding = torch.cat(
                    [float_seq_fields_embedding, float_fields_embedding], dim=1
                )

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(
                token_fields, dim=1
            )  # [batch_size, num_token_field, 2]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat(
                    [token_seq_fields_embedding, token_fields_embedding], dim=1
                )

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field, 2] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding
