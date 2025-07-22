import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender_my import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers


class AFN(ContextRecommender):
    r"""AFN (Attentional Factorization Network) is a context-aware recommender model that uses attention mechanisms
    to learn the interactions between user and item features.

    Args:
        config (Config): The configuration of the model.
    """

    def __init__(self, config, dataset):
        super(AFN, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.logarithmic_neurons = config["logarithmic_neurons"]
        self.ensemble_dnn = config["ensemble_dnn"]

        # self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.logarithmic_neurons * self.embedding_size
        ] + self.mlp_hidden_size + [1]
        # self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.dense_layers = MLPLayers(
            size_list, self.dropout_prob, activation='relu', bn=False
        )
        self.coefficient_W = nn.Linear(self.num_feature_field, self.logarithmic_neurons, bias=False)

        self.log_batch_norm = nn.BatchNorm1d(self.num_feature_field)
        self.exp_batch_norm = nn.BatchNorm1d(self.logarithmic_neurons)
        # if self.ensemble_dnn:
        #     self.embedding)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        for name, submodule in self.named_modules():
            self._init_weights(name, submodule)
            
    def _init_weights(self, name, module):
        if name not in ['id2afeats', 'id2tfeats']:
            if isinstance(module, nn.Embedding):
                xavier_normal_(module.weight.data)
            elif isinstance(module, nn.Linear):
                xavier_normal_(module.weight.data)
                if module.bias is not None:
                    constant_(module.bias.data, 0)
    
    def log_net(self, emb):
        # emb: [batch_size, num_field, embed_dim]
        emb = torch.abs(emb)
        emb = torch.clamp(emb, min=1e-5)
        log_emb = torch.log(emb)
        # log_emb = self.log_batch_norm(log_emb)
        log_out = self.coefficient_W(log_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(log_out)
        # cross_out = self.exp_batch_norm(cross_out)
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out
    
    def forward(self, interaction):
        embeddings = self.concat_embed_input_fields(
            interaction
        ) # [batch_size, num_field, embed_dim]
        dnn_input = self.log_net(embeddings)
        # print(torch.max(dnn_input, ))
        afn_out = self.dense_layers(dnn_input)
        

        # if self.ensemble_dnn:
        # TODO 修改结构，能接受两个embedding\
        y = afn_out
        # print(y.shape)
        return y.squeeze(-1)
    
    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        print(self.loss(output, label).item())
        return self.loss(output, label) + self.reg_emb_loss()
    
    def predict(self, interaction):
        result = self.forward(interaction)
        print('max', str(torch.max(result, )))
        print('avg', str(torch.mean(result, )))
        return result