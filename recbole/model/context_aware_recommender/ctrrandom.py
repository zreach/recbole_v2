# -*- coding: utf-8 -*-
# @Time   : 2020/7/8 10:09
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : fm.py

# UPDATE:
# @Time   : 2020/8/13,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
FM
################################################
Reference:
    Steffen Rendle et al. "Factorization Machines." in ICDM 2010.
"""

import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch 

from recbole.model.abstract_recommender_my import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine


class CTRRandom(ContextRecommender):
    """Factorization Machine considers the second-order interaction with features to predict the final score."""

    def __init__(self, config, dataset):
        super(CTRRandom, self).__init__(config, dataset)
        self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCEWithLogitsLoss()
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))



    def forward(self, interaction):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        return torch.rand(len(interaction), device=self.device).squeeze(-1)
