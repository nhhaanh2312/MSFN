from torch.nn import Linear as Lin, Sequential as Seq, Parameter
from dense import MLP
from sparse import MultiSeq, GraphConv, ResGraphBlock,DenseGraphBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input, adj):

        h = torch.matmul(input, self.W)
        bs, N, _ = h.size()

        a_input = torch.cat([h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(bs, N, -1, 2 * self.out_features)
        print("h size:", a_input.shape)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        batch_adj = torch.unsqueeze(adj, 0).repeat(bs, 1, 1)


        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(batch_adj > 0, e, zero_vec)
        attention = self.dropout_layer(F.softmax(attention, dim=-1)) # [bs, N, N]
        h_prime = torch.bmm(attention, h)# [bs, N, F]


        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feature,adj):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

class ResDeepGCN(nn.Module):
    def __init__(self, in_channels, channels1,channels2,n_blocks,n_classes, act,norm, bias, dropout,conv,heads):
        super(ResDeepGCN, self).__init__()
        c_growth = channels1
        self.n_blocks=n_blocks
        self.head = GraphConv(in_channels, channels1,conv)
        self.gcn = GraphConv(channels1*n_blocks, n_classes,conv)

        self.backbone = MultiSeq(*[ResGraphBlock(channels1,channels2) for _ in range(n_blocks - 1)])

        self.fusion_block = MLP([channels1*n_blocks, 256], act, None, bias)

        self.prediction = Seq(*[MLP([256, 64], act, norm, bias),
                                torch.nn.Dropout(p=dropout),
                                MLP([64, n_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True



    def forward(self, x, adj):
        feats = [self.head(x,adj)]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], adj))

        feats = torch.cat(feats, 1)
        fusion= self.fusion_block(feats)
        feature=fusion
        out = self.prediction(fusion)
        return out,feature

