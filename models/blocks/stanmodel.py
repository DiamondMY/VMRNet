import torch
import torch.nn as nn
from torch.nn import GELU

from nncore.nn import (MODELS, MultiHeadAttention, build_norm_layer)
from einops import rearrange

from transformers import CLIPConfig, CLIPModel
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP
import nncore

# 整个模块还是要改成直接读数据
@MODELS.register()
class VITCLIP_STAN(nn.Module):
    def __init__(self):

        super(VITCLIP_STAN, self).__init__()

        pretrained_model = r'../../data/mydata/huggingface_model/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb'
        configuration = CLIPConfig().from_json_file(nncore.join(pretrained_model, 'config.json'))
        clip_model = CLIPModel.from_pretrained(nncore.join(pretrained_model, 'pytorch_model.bin'), config=configuration)


        self.config = configuration
        # print(self.config.vision_config)

        self.num_patches = (configuration.vision_config.image_size // configuration.vision_config.patch_size) ** 2
        self.embed_dim = configuration.vision_config.hidden_size

        self.class_embedding = clip_model.vision_model.embeddings.class_embedding
        self.patch_embedding = clip_model.vision_model.embeddings.patch_embedding
        self.position_embedding = clip_model.vision_model.embeddings.position_embedding

        self.pre_layrnorm = clip_model.vision_model.pre_layrnorm
        self.post_layernorm = clip_model.vision_model.post_layernorm
        self.layers = clip_model.vision_model.encoder.layers
        # self.return_mean = return_mean

        self.visual_projection = clip_model.visual_projection
        del clip_model


        self.T = 75
        self.clip_stan = nn.ModuleList([
            VITCLIP_STANLayer(T=self.T, config=self.config)
            for _ in range(1)
        ])

    def forward(self, inner_feature, pooler_output, mask):

        batch_num = mask.shape[0]
        max_length = mask.shape[1]

        # 此时的inner_feature为batch_num, 3 * frames_num, 50, 768，模块会把除了batch_num的前两维做压缩，要人为把他拆开来
        # inner_feature:batch_num, 3, frames_num, 50, 768
        # inner_feature = rearrange(inner_feature, 'b (n t) l d -> b n t l d', n=3)

        # for i in range(inner_feature.shape[0]):
        #     if inner_feature[i][0].shape[0] != max_length:
        #         # print('跑模型时存在没有扩维的数据！')
        #         # 手动扩维一下
        #         print('目前的维度：{}'.format(inner_feature[i][0].shape[0]))
        #         for num in len(range(inner_feature.shape[1])):
        #             inner_feature[i][num] = torch.nn.functional.pad(inner_feature[i][num],
        #                                                     (0, 0, 0, max_length - inner_feature[i][0].shape[0]))  # 左右上下
        #         print('扩维后的维度:{}'.format(inner_feature[i][0].shape[0]))


        x2 = None

        for layer_num in range(len(self.clip_stan)):
            x2 = self.clip_stan[layer_num](inner_feature[:, layer_num], x2)

        # return x_tokens, x_cls
        output = torch.cat((pooler_output, x2[:, 0].unsqueeze(1).repeat(1, max_length, 1)), dim=2)

        # batch, frames_num, 512 + 128
        return output

    def result_process(self, x, x2):
        # 两个cls相加
        cls_token = x[:, 0] + x2[:, 0].repeat(1, self.T).view(x2.size(0) * self.T, -1)
        cls_token = self.post_layernorm(cls_token)


        # batch * frames_num, 50, 768
        x_tokens = x
        # batch * frames_num, 768
        x_cls = x2

        # B * num_frames, 512
        x_cls = self.visual_projection(x_cls)

        # batch, frames_num, 512
        x_cls = rearrange(x_cls, '(b t) d -> b t d', t=self.T)

        # batch, frames_num * 50, 768
        x_tokens = rearrange(x_tokens, '(b t) l d -> b (t l) d', t=self.T)

        if self.return_mean:
            # batch, 512
            x_cls = x_cls.mean(1)
        return x_tokens, x_cls

@MODELS.register()
class VITCLIP_STANLayer(nn.Module):
    def __init__(self, T, config):

        super(VITCLIP_STANLayer, self).__init__()

        self.embed_dim = config.vision_config.hidden_size
        # self.embed_dim = 128

        self.T = T

        self.timesFPN_S_layers = CLIPLayer_Spatial(T=self.T, config=config.vision_config, embed_dim=self.embed_dim)
        self.timesFPN_T_layers = CLIPLayer_AttnTime(T=self.T, config=config.vision_config, embed_dim=self.embed_dim)

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.position_embedding = nn.Embedding(50, self.embed_dim)
        self.drop_after_pos = nn.Dropout(p=0)
        self.drop_after_time = nn.Dropout(p=0)

        # 这里 64是他给的常数
        self.timesFPN_time_embed = nn.Embedding(self.T, self.embed_dim)
        # self.down_visual_features = nn.Linear(768, self.embed_dim)
        # self.visual_projection = nn.Linear(768, 512, bias=False)

        # self.init_weights()

        self.T = 75
        self.return_mean = True

    def forward_patch(self, x, x2=None):

        # 给x降个维，768太可怕了
        # x = self.down_visual_features(x)

        # batch, cls(1) + patch_num(49) * frames_num, 768
        x2 = self.forword_timeModule(x, x2, self.T)

        return x2

    def forword_timeModule(self, x1, x2, T):
        x1 = rearrange(x1, '(b t) l d -> b t l d', t=T)

        self.cls_residue = False
        self.timesFPN_T_layers.t = T
        self.timesFPN_S_layers.t = T

        if x2 is not None:
            cls_token_ori = x1[:, :, 0, :]
            cls_token = cls_token_ori.mean(dim=1).unsqueeze(1)
            x1 = x1[:, :, 1:, :]
            x1 = rearrange(x1, 'b t l d -> b (l t) d')
            x1 = torch.cat((cls_token, x1), dim=1)
            if not self.cls_residue:
                x = x2 + x1
            else:
                if self.training:
                    cls_token1 = cls_token_ori[:, 0::2, :].mean(dim=1).unsqueeze(1)
                else:
                    cls_token1 = cls_token_ori.mean(dim=1).unsqueeze(1)

                x1 = torch.cat((cls_token1.repeat(1, self.num_cls, 1), x1[:, 1:, :]), dim=1)
                x = x2 + x1
        else:
            x = x1
            x = self.input_ini(x)

        x = self.timesFPN_T_layers(x)
        x = self.timesFPN_S_layers(x)
        return x

    def input_ini(self, x):
        cls_old = x[:, :, 0, :].mean(dim=1).unsqueeze(1)
        x = x[:,:,1:,:]
        B,T,L,D = x.size()
        x = rearrange(x, 'b t l d -> (b t) l d')
        cls_tokens = self.class_embedding.expand(x.size(0), 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.position_embedding(position_ids)
        x = x + pos_embed
        x = self.drop_after_pos(x)

        cls = x[:B, 0, :].unsqueeze(1)
        # cls = torch.zeros((B, 1, self.embed_dim)).cuda()
        # for i in range(B):
        #     cls[i] = x[T * i, 0]

        x = rearrange(x[:, 1:, :], '(b t) l d -> (b l) t d', b=B)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        time_embed = self.timesFPN_time_embed(position_ids)
        x = x + time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b l) t d -> b (l t) d', b=B)
        cls = (cls_old + cls) / 2
        x = torch.cat((cls, x), dim=1)

        # batch, cls(1) + patch_num(49) * frames_num, 768
        return x

    def forward(self, x, x2=None, **kwargs):

        # B, num_frames, 50, 768
        B, T, L, M = x.shape
        # B * num_frames, 50, 768
        # x = x.reshape((-1,) + x.shape[2:])
        x = rearrange(x, 'b t l m -> (b t) l m', b=B)

        self.T = T

        x2 = self.forward_patch(x, x2)
        # x_tokens, x_cls = self.result_process(vision_outputs)

        return x2

@MODELS.register()
class CLIPLayer_Spatial(nn.Module):
    def __init__(self, T, config, embed_dim,layer_num=0.1, num_cls=1):

        super(CLIPLayer_Spatial, self).__init__()

        # self.embed_dim = embed_dim
        self.embed_dim = config.hidden_size

        # self.self_attn = MultiHeadAttention(self.embed_dim)
        self.self_attn = CLIPAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)

        # self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 4)
        # self.gelu = GELU()
        # self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.mlp = CLIPMLP(config)

        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        dropout_layer = dict(type='drop', p=layer_num) if layer_num > 0 else None
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = build_norm_layer(dropout_layer) if dropout_layer else nn.Identity()
        self.t = T

        self.num_cls = num_cls

    def forward(self, hidden_states: torch.Tensor):

        # batch, cls(1) + patch_num(49) * frames_num, 768
        residual = hidden_states

        # B, 1, 768
        init_cls_token = hidden_states[:, :self.num_cls, :]
        # B, 49, 768
        query_s = hidden_states[:, self.num_cls:, :]

        b, pt, m = query_s.size()
        p = pt // self.t
        # 相同的cls，为每个帧复制了一份(一共75)
        cls_token = init_cls_token.unsqueeze(1).repeat(1, self.t, 1, 1).reshape(b * self.t, self.num_cls, m)  # can I do?
        # cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=self.t)
        hidden_states = torch.cat((cls_token, query_s), 1)
        # hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(hidden_states)

        res_spatial = self.dropout_layer(self.proj_drop(hidden_states[0]))

        cls_token = res_spatial[:, :self.num_cls, :].reshape(b, self.t, self.num_cls, m)
        cls_token = torch.mean(cls_token, 1)
        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, self.num_cls:, :], '(b t) p m -> b (p t) m', p=p, t=self.t)
        hidden_states = torch.cat((cls_token, res_spatial), 1)
        # hidden_states = self.process.after(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)
        # hidden_states = self.fc1(hidden_states)
        # hidden_states = self.gelu(hidden_states)
        # hidden_states = self.fc2(hidden_states)

        hidden_states = residual + hidden_states

        outputs = hidden_states

        # cls头也做了处理，对msa和mlp部分都做了残差
        # batch, cls(1) + patch_num(49) * frames_num, 768
        return outputs

@MODELS.register()
class CLIPLayer_AttnTime(nn.Module):
    def __init__(self, T, embed_dim, config, layer_num=0.1, num_cls=1):

        super(CLIPLayer_AttnTime, self).__init__()

        # self.embed_dim = embed_dim
        self.embed_dim = config.hidden_size

        # self.self_attn = MultiHeadAttention(self.embed_dim)
        self.self_attn = CLIPAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.num_cls = num_cls

        dropout_layer = dict(type='drop', p=layer_num) if layer_num > 0 else None
        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = build_norm_layer(dropout_layer)
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        # 唯一改动（理论上）
        # constant_init(self.temporal_fc, val=0, bias=0)
        self.t = T

    def forward(self, hidden_states: torch.Tensor):

        residual = hidden_states[:, self.num_cls:, :]

        init_cls_token = hidden_states[:, :self.num_cls, :]
        query_t = hidden_states[:, self.num_cls:, :]

        b, pt, m = query_t.size()
        p = pt // self.t
        # hidden_states = query_t.reshape(b * p, t, m)
        hidden_states = rearrange(query_t, 'b (p t) m -> (b p) t m', p=p, t=self.t)

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(hidden_states)

        res_temporal = self.dropout_layer(self.proj_drop(hidden_states[0]))
        res_temporal = self.temporal_fc(res_temporal)

        # hidden_states = res_temporal.reshape(b, p * self.t, m)
        hidden_states = rearrange(res_temporal, '(b p) t m -> b (p t) m', p=p, t=self.t)

        hidden_states = residual + hidden_states
        hidden_states = torch.cat((init_cls_token, hidden_states), 1)
        outputs = hidden_states

        # cls部分未做改动，后面为原参数与经过模块处理的残差和
        # batch, cls(1) + patch_num(49) * frames_num, 768
        return outputs
