import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


from recbole.model.utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood
from recbole.model.abstract_recommender_my import GeneralRecommender
from recbole.utils import InputType
# from recbole.model.layers import MLPLayers
# from recbole.model.loss import RegLoss
class LATTICE(GeneralRecommender):
    
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LATTICE, self).__init__(config, dataset)

        

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.weight_size = config['weight_size']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.batch_size = config['train_batch_size']

        self.build_item_graph = True
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.item_adj = None

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if config['cf_model'] == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            dropout_list = config['mess_dropout']
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        dataset_path = os.path.abspath(config['data_path'])
        audio_adj_file = os.path.join(dataset_path, 'audio_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k))

        if self.a_feats is not None:
            self.audio_embedding = self.a_feats
            if os.path.exists(audio_adj_file):
                audio_adj = torch.load(audio_adj_file)
            else:
                audio_adj = build_sim(self.id2afeats.weight.detach())
                audio_adj = build_knn_neighbourhood(audio_adj, topk=self.knn_k)
                audio_adj = compute_normalized_laplacian(audio_adj)
                torch.save(audio_adj, audio_adj_file)
            self.audio_original_adj = audio_adj.cuda()

        if self.t_feats is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feats, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj)
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.a_feats is not None:
            self.audio_trs = nn.Linear(self.a_feats.shape[1], self.feat_embed_dim)
        # if self.t_feats is not None:
        #     self.text_trs = nn.Linear(self.t_feats.shape[1], self.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
    
    def pre_epoch_processing(self):
        self.build_item_graph = True
    
    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            #print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, adj, build_item_graph=False):
        if self.a_feats is not None:
            audio_feats = self.audio_trs(self.id2afeats.weight)
        
        if build_item_graph:
            weight = self.softmax(self.modal_weight)

            if self.a_feats is not None:
                self.audio_adj = build_sim(audio_feats)
                self.audio_adj = build_knn_neighbourhood(self.audio_adj, topk=self.knn_k)
                learned_adj = self.audio_adj
                original_adj = self.audio_original_adj

            # if self.a_feats is not None and self.t_feats is not None:
            #     learned_adj = weight[0] * self.audio_adj + weight[1] * self.text_adj
            #     original_adj = weight[0] * self.audio_original_adj + weight[1] * self.text_original_adj

            learned_adj = compute_normalized_laplacian(learned_adj)
            if self.item_adj is not None:
                del self.item_adj
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()
        pass
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.mm(self.item_adj, h)

        if self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        return batch_mf_loss + batch_emb_loss + batch_reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, build_item_graph=True)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores