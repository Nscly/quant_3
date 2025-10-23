import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool, global_max_pool


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=-1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention_block = AttentionBlock(time_step, hidden_dim) 
        self.dropout = nn.Dropout(0.2)
        self.dim = hidden_dim
    
    def forward(self, seq):
        seq_vector, _ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)
        attention_vec = attention_vec.view(-1, 1, self.dim)
        return attention_vec


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gat = GATConv(hidden_dim, hidden_dim)
        self.dim = hidden_dim
    
    def forward(self, seq, edge_index):
        seq_vector, _ = self.encoder(seq)
        seq_vector = seq_vector[:, -1, :]
        attention_vec = self.gat(seq_vector, edge_index)
        return seq_vector, attention_vec


class CategoricalGraph(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, inner_edge, outer_edge, input_num, device):
        super(CategoricalGraph, self).__init__()

        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.device = device

        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, time_step, hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim, hidden_dim)
        self.weekly_attention = AttentionBlock(input_num, hidden_dim)
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)

        self.reg_layer = nn.Linear(hidden_dim, 1)
        self.cls_layer = nn.Linear(hidden_dim, 1)

    def forward(self, weekly_batch):
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1, self.time_step, self.input_dim))

        for week_idx in range(1, self.input_num):
            weekly_inp = weekly_batch[week_idx]
            weekly_inp = weekly_inp.view(-1, self.time_step, self.input_dim)
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp)
            weekly_embedding = torch.cat((weekly_embedding, week_stock_embedding), dim=1)

        weekly_att_vector, _ = self.weekly_attention(weekly_embedding)
        category_vectors = self.cat_gat(weekly_att_vector, self.inner_edge)

        fusion_vec = torch.cat((weekly_att_vector, category_vectors), dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self, test_data, device, top_k=5):
        y_pred_all_reg, y_pred_all_cls = [], []
        test_w1, test_w2, test_w3, test_w4 = test_data
        for idx, _ in enumerate(test_w2):
            batch_x1, batch_x2, batch_x3, batch_x4 = test_w1[idx].to(self.device), \
                                        test_w2[idx].to(self.device),\
                                        test_w3[idx].to(self.device),\
                                        test_w4[idx].to(self.device)
            batch_weekly = [batch_x1, batch_x2, batch_x3, batch_x4][-self.input_num:]
            pred_reg, pred_cls = self.forward(batch_weekly)
            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
            y_pred_all_reg.extend(pred_reg.tolist())
            y_pred_all_cls.extend(pred_cls.tolist())
        return y_pred_all_reg, y_pred_all_cls


class CategoricalGraphAtt(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, inner_edge, outer_edge, input_num, use_gru, device):
        super(CategoricalGraphAtt, self).__init__()

        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.use_gru = use_gru
        self.device = device

        if self.use_gru:
            self.weekly_encoder = nn.GRU(hidden_dim, hidden_dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, time_step, hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim, hidden_dim)
        self.inner_gat = GATConv(hidden_dim, hidden_dim)
        self.weekly_attention = AttentionBlock(input_num, hidden_dim)
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)

        self.reg_layer = nn.Linear(hidden_dim, 1)
        self.cls_layer = nn.Linear(hidden_dim, 1)

    def forward(self, weekly_batch):
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1, self.time_step, self.input_dim))

        for week_idx in range(1, self.input_num):
            weekly_inp = weekly_batch[week_idx]
            weekly_inp = weekly_inp.view(-1, self.time_step, self.input_dim)
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp)
            weekly_embedding = torch.cat((weekly_embedding, week_stock_embedding), dim=1)

        if self.use_gru:
            weekly_embedding, _ = self.weekly_encoder(weekly_embedding)
        weekly_att_vector, _ = self.weekly_attention(weekly_embedding)

        inner_graph_embedding = self.inner_gat(weekly_att_vector, self.inner_edge)
        
        fusion_vec = torch.cat((weekly_att_vector, inner_graph_embedding), dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self, test_data, device, top_k=5):
        y_pred_all_reg, y_pred_all_cls = [], []
        test_w1, test_w2, test_w3, test_w4 = test_data
        for idx, _ in enumerate(test_w2):
            batch_x1, batch_x2, batch_x3, batch_x4 = test_w1[idx].to(self.device), \
                                        test_w2[idx].to(self.device),\
                                        test_w3[idx].to(self.device),\
                                        test_w4[idx].to(self.device)
            batch_weekly = [batch_x1, batch_x2, batch_x3, batch_x4][-self.input_num:]
            pred_reg, pred_cls = self.forward(batch_weekly)
            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
            y_pred_all_reg.extend(pred_reg.tolist())
            y_pred_all_cls.extend(pred_cls.tolist())
        return y_pred_all_reg, y_pred_all_cls


class CategoricalGraphPool(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, inner_edge, inner20_edge, outer_edge, input_num, use_gru, device):
        super(CategoricalGraphPool, self).__init__()

        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.inner20_edge = inner20_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.use_gru = use_gru
        self.device = device

        if self.use_gru:
           self.weekly_encoder = nn.GRU(hidden_dim, hidden_dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, time_step, hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim, hidden_dim)
        self.inner_gat = GATConv(hidden_dim, hidden_dim)
        self.pooling_gcn = SAGPooling(hidden_dim, ratio=0.5)
        self.weekly_attention = AttentionBlock(input_num, hidden_dim)
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)

        self.reg_layer = nn.Linear(hidden_dim, 1)
        self.cls_layer = nn.Linear(hidden_dim, 1)

    def forward(self, weekly_batch):
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1, self.time_step, self.input_dim))

        for week_idx in range(1, self.input_num):
            weekly_inp = weekly_batch[week_idx]
            weekly_inp = weekly_inp.view(-1, self.time_step, self.input_dim)
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp)
            weekly_embedding = torch.cat((weekly_embedding, week_stock_embedding), dim=1)

        if self.use_gru:
            weekly_embedding, _ = self.weekly_encoder(weekly_embedding)
        weekly_att_vector, _ = self.weekly_attention(weekly_embedding)

        inner_graph_embedding = self.inner_gat(weekly_att_vector, self.inner_edge)

        fusion_vec = torch.cat((weekly_att_vector, inner_graph_embedding), dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self, test_data, device, top_k=5):
        y_pred_all_reg, y_pred_all_cls = [], []
        test_w1, test_w2, test_w3, test_w4 = test_data
        for idx, _ in enumerate(test_w2):
            batch_x1, batch_x2, batch_x3, batch_x4 = test_w1[idx].to(self.device), \
                                        test_w2[idx].to(self.device),\
                                        test_w3[idx].to(self.device),\
                                        test_w4[idx].to(self.device)
            batch_weekly = [batch_x1, batch_x2, batch_x3, batch_x4][-self.input_num:]
            pred_reg, pred_cls = self.forward(batch_weekly)
            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
            y_pred_all_reg.extend(pred_reg.tolist())
            y_pred_all_cls.extend(pred_cls.tolist())
        return y_pred_all_reg, y_pred_all_cls
