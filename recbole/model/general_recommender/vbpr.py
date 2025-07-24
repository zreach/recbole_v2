
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender_my import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from tqdm import tqdm
import torch.nn.functional as F



class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataloader):

        super(VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.a_feats is not None and self.t_feats is not None:
            self.item_raw_features = torch.cat((self.t_feats, self.a_feats), -1)
        elif self.a_feats is not None:
            self.item_raw_features = self.a_feats
        else:
            self.item_raw_features = self.t_feats
        
        self.item_raw_features = self.item_raw_features.to(self.device)

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        for name, submodule in self.named_modules():
            self._init_weights(name, submodule)

    def _init_weights(self, name, module):
        if name not in ['id2afeats', 'id2tfeats']:
            if isinstance(module, nn.Embedding):
                xavier_normal_initialization(module.weight.data)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_embeddings = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[users, :]
        pos_e = item_embeddings[pos_items, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_items, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score