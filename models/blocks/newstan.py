import torch
from torch import nn
from torch.nn import GELU
from nncore.nn import MultiHeadAttention, MODELS, build_norm_layer, build_linear_modules


@MODELS.register()
class VETMo(nn.Module):

    def __init__(self):

        super(VETMo, self).__init__()

        self.T = 75

        self.con_model = nn.ModuleList([
            VETMoLayer(T=self.T)
            for _ in range(1)
        ])

    def forward(self, inner_features, output, text_features, mask=None):
        # inner_features: batch, layers, 75, 768
        # output: batch, 75, 512

        x2 = None
        is_last_layer = False
        # 暂时只写一层
        for layer in range(len(self.con_model)):
            if layer == len(self.con_model) - 1:
                is_last_layer = True
            x2 = self.con_model[layer](inner_features[:, layer], text_features, x2, is_last_layer, mask=mask)

        x = x2
        return torch.cat((output, x), dim=2)

@MODELS.register()
class VETMoLayer(nn.Module):
    def __init__(self, T, embed_dim=512):

        super(VETMoLayer, self).__init__()

        self.embed_dim = embed_dim if embed_dim else 768
        self.vnn = nn.Linear(768, self.embed_dim)
        # self.qnn = nn.Linear(512, 768)

        self.mapping = nn.Linear(self.embed_dim, 256)
        # self.position_embedding = nn.Embedding(75, self.embed_dim)

        # 模拟mini clip transformer块
        self.self_attn = MultiHeadAttention(self.embed_dim)
        # self.q_attn = nn.TransformerEncoderLayer(512, nhead=8)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        # self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 4)
        self.gelu = GELU()
        self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.norm = build_norm_layer(dict(type='LN'), 256)
        self.norm2 = build_norm_layer(dict(type='LN'), embed_dim)


    def tmp_forward(self, x, text_features, mask=None):
        x1 = x

        x1 = x1 + self.self_attn(x1, mask=mask)
        x1 = self.layer_norm1(x1)

        x1 = x1 + self.fc2(self.gelu(self.fc1(x1)))

        x1 = self.mapping(x1)
        x1 = self.norm(x1)

        return x1

    def forward(self, x, query, x2=None, is_last_layer=False, mask=None):
        q = torch.zeros((query.shape[0], 512)).cuda()
        for i in range(len(query)):
            for t in range(len(query[i])):
                if query[i][t][0] == float('inf'):
                    q[i] = torch.mean(query[i, :t], dim=0)
                    break

        # batch, 1, 512
        q = torch.unsqueeze(q, dim=1)

        mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), dim=1)

        # batch, 75, 768
        x1 = x

        # batch, 75, 512
        x1 = self.vnn(x1)
        if x2 != None:
            x1 = x1 + x2
        # batch, 76, 512
        x1 = torch.cat((q, x1), dim=1)
        # x1 = self.layer_norm2(x1)
        # x1 = self.layer_norm1(x1)
        x1 = x1 + self.self_attn(x1, mask=mask)
        x1 = self.layer_norm1(x1[:, 1:])

        x1 = x1 + self.fc2(self.gelu(self.fc1(x1)))

        if is_last_layer:
            x1 = self.mapping(x1)
            x1 = self.norm(x1)
        else:
            x1 = self.norm2(x1)

        # return x1 + x
        return x1

