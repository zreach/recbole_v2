# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender_my import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from tqdm import tqdm
import pickle


class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.has_iid = True
        if config['no_itemid']:
            self.has_iid = False
        if self.has_iid:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        # self.apply(xavier_normal_initialization)
        for name, submodule in self.named_modules():
            self._init_weights(name, submodule)

    def _init_weights(self, name, module):
        if name not in ['id2afeats', 'id2tfeats']:
            if isinstance(module, nn.Embedding):
                xavier_normal_initialization(module.weight.data)


    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        if self.has_iid:
            item_e = self.item_embedding(item)
        else:
            item_e = torch.zeros((item.shape[0], self.embedding_size), device=item.device)
        if self.use_audio:
            audio_e = self.get_wav_embedding(item)
            item_e = item_e + audio_e
        if self.use_text:
            text_e = self.get_text_embedding(item)
            item_e = item_e + text_e
        return item_e
    
    def compute_all_item_embeddings(self, batch_size=1024):
        """
        Compute all item embeddings in batches to avoid OOM.
        The result is stored in self.all_item_embeddings_cached.
        """
        self.all_item_embeddings_cached = []
        all_items = torch.arange(self.n_items, device=self.device)
        for start_idx in tqdm(range(0, self.n_items, batch_size)):
            item_batch = all_items[start_idx: start_idx + batch_size]
            item_batch_embedding = self.get_item_embedding(item_batch)
            # print(item_batch_embedding.shape)
            self.all_item_embeddings_cached.append(item_batch_embedding.cpu())
        self.all_item_embeddings_cached = torch.cat(self.all_item_embeddings_cached, dim=0).to(self.device)

    def compute_all_user_embeddings(self, batch_size=1024):
        """
        Compute all user embeddings in batches to avoid OOM.
        The result is stored in self.all_user_embeddings_cached.
        """
        self.all_user_embeddings_cached = []
        all_users = torch.arange(self.n_users, device=self.device)
        for start_idx in tqdm(range(0, self.n_users, batch_size)):
            user_batch = all_users[start_idx: start_idx + batch_size]
            user_batch_embedding = self.get_user_embedding(user_batch)
            self.all_user_embeddings_cached.append(user_batch_embedding.cpu())
        self.all_user_embeddings_cached = torch.cat(self.all_user_embeddings_cached, dim=0).to(self.device)
    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        # Check if item embeddings are cached, if not, compute them in batches
        if not hasattr(self, 'all_item_embeddings_cached'):
            print("Computing all item embeddings...")
            self.compute_all_item_embeddings()
            self.compute_all_user_embeddings()
            embed_dict = {
                'all_item_embeddings_cached': self.all_item_embeddings_cached.detach().cpu().numpy(),
                'user_embedding': self.all_user_embeddings_cached.detach().cpu().numpy()
            }
            with open('embeddings_cached.pkl', 'wb') as f:
                pickle.dump(embed_dict, f)
        # print(self.all_user_embeddings_cached.shape)
        

        all_item_e = self.all_item_embeddings_cached

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
