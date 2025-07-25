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

import numpy as np
import torch
import torch.nn as nn
import pickle

from recbole.model.layers import FMEmbedding, FMFirstOrderLinear, FLEmbedding
from recbole.utils import ModelType, InputType, FeatureSource, FeatureType, set_color
from recbole.model.layers import MLPLayers
from recbole.model.loss import RegLoss

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

        if self.use_cb:
            
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

            if config['dataset'] in ['lfm1b-fil', 'm4a-fil']:
                
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
            elif config['dataset'] in ['m4a', 'lfm2b-fil']: # 这个数据没有时间维度， 而且不需要map
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
        # self.id2token = dataset.field2id_token

        self.use_cb = config['use_cb']
        # self.feature_type = config['wav_feature_type']
        if self.use_cb is None:
            self.use_cb = False

        self.use_audio = config['use_audio']
        self.use_text = config['use_text']

        if self.use_cb:
            

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
            
            
            if config['dataset'] in ['lfm1b-fil', 'm4a-fil']:
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
            elif config['dataset'] in ['m4a', 'lfm2b-fil']: # 这个数据没有时间维度， 而且不需要map
                for k, v in self.token2id['tracks_id'].items():
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

        self.first_order_linear = FMFirstOrderLinear(config, dataset)
        
        
        if self.use_audio:
            size_list = [
                self.wav_embedding_size 
            ] + config['wav_mlp_sizes'] + [self.embedding_size]
            self.wav_mlp = MLPLayers(size_list, 0.2)
            # self.wav_fc = nn.Linear(1024, self.embedding_size)
            self.num_feature_field += 1
        if self.use_text:
            size_list = [
                self.text_embedding_size 
            ] + config['text_mlp_sizes'] + [self.embedding_size]
            self.text_mlp = MLPLayers(size_list, 0.2)
            # self.text_fc = nn.Linear(1024, self.embedding_size)
            self.num_feature_field += 1
    
    def get_music_features(self, track_ids):
        # 返回Tensor的feature
        # TODO : 利用并行加速
        features = []
        for id in track_ids:
            try:
                numpy_feature = self.music_features[id.item()]
            except:
                numpy_feature = np.zeros((1,self.wav_embedding_size))
                # raise ValueError(f"Could not find {id}")
                print(f"Could not find {id}")
            features.append(torch.tensor(numpy_feature))
        features_tensor = torch.stack(features)

        return features_tensor.to(torch.float32).to(self.device)

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
            wav_embedding = self.get_wav_embedding(interaction) 
            all_embeddings.append(wav_embedding)
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
