import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.non_local_simple_version import NONLocalBlock2D
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange


def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

# class CosineSimilarity(nn.Module):
#     pass
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def dwt_init(self,x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

    def forward(self, x):
        return self.dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def iwt_init(self,x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, :out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height,
                         out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    def forward(self, x):
        return self.iwt_init(x)


class TransformerDWTFusion(nn.Module):


    def __init__(self,inplanes=64*3, outplanes=576,kernel_size=2,stride=2,
                 act_layer=nn.GELU,norm_layer=partial(nn.LayerNorm, eps=1e-6,),
                 embed_dim=576,  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,trans_dpr=0. ):
        super(TransformerDWTFusion, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.conv_project = nn.Conv2d(inplanes // 3 // 4 , outplanes, kernel_size=1, stride=1, padding=0)
        # self.conv_project0 = nn.Conv2d(inplanes // 3 , outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

        self.transformer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=trans_dpr,
                                  )

        self.resnet = ConvBlock(inplanes=inplanes // 3 , outplanes=inplanes // 3 , stride=1, res_conv=True,
                                      act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                                      drop_block=None, drop_path=None)

    def forward(self, x, x_t):
        x,x2 = self.resnet(x,return_x_2=True)
        x_t = self.transformer(x_t)

        B,C,H,W = x.size()
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)

        cls = x_t[:, 0][:, None, :]   # [4,1,576]
        cls_kernel = cls.reshape(B, C, 3, 3).permute(1,0,2,3)
        cls_kernel = torch.mean(cls_kernel, dim=1).unsqueeze(1).repeat(1, C, 1, 1)
        x_HH_new = F.conv2d(x_HH, cls_kernel, stride=(1, 1), padding=1)  # B，64,28,28
        x_new = torch.cat((x_LL, x_HL, x_LH, x_HH_new), dim=1)
        x_new = self.iwt(x_new)

        x_L = self.conv_project(x2)
        x_L = self.sample_pooling(x_L).flatten(2).transpose(1, 2)
        x_L = self.act(self.ln(x_L))    # [B,196,576]
        #
        # x_t_new = torch.cat((cls, x_L), dim=1)
        return x_new,x_t#_new


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

        # self.apply(self._init_weights)



    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=False):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class CrossAttentionWithLearnableQuery(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossAttentionWithLearnableQuery, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))  # Learnable query  [1,1,C]
        for i in range(4):
            self.add_module('attention_{}'.format(i), nn.MultiheadAttention(feature_dim, num_heads))

    def forward(self, key_value):
        # key_value shape: (T, B, C)  4个B,T,C
        B,T,C = key_value[0].shape
        query = self.query.expand(T, B, -1)  # Expanding query to match T and B
        outputs = []
        for i in range(4):
            kv = key_value[i].permute(1, 0, 2)
            output, _ = eval('self.attention_' + str(i))(query, kv, kv)
            outputs.append(output)
        r = torch.cat(outputs, dim=-1)
        return r.permute(1, 0, 2)

class FusionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads    # 这里是多个头
        self.scale = qk_scale or head_dim ** -0.5

        for i in range(1,5):
            self.add_module('proj_' + str(i),
                            nn.Linear(dim, dim)
                            )
            self.add_module('qkv_' + str(i),
                            nn.Linear(dim, dim * 3, bias=qkv_bias))

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # self.queryEnhancedSemanticModule = QueryEnhancedSemanticModule(dim, num_heads)

    def forward(self, X):
        B, V, N, C = X.shape
        Q = []
        K = []
        V = []
        for i in range(1,5):
            qkv = eval('self.qkv_'+str(i))(X[:,i-1]).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q1, k1, v1 = qkv[0], qkv[1], qkv[2]  # 1,9,127,64     576=64*9
            Q.append(q1)
            K.append(k1)
            V.append(v1)

        Reaults = []
        for i in range(len(Q)):
            X = []
            for j in range(len(K)):
                attn = (Q[i] @ K[j].transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = (attn @ V[j]).transpose(1, 2).reshape(B, N, C)
                x = eval('self.proj_' + str(i+1))(x)
                x = self.proj_drop(x)          #  (1,127,576)
                X.append(x)
            r = torch.stack(X)      # (4,1,127,576)
            # r = self.queryEnhancedSemanticModule(r)       # 这里可以换成 non_local
            r = r.mean(dim=0)      # 这里可以换成 non_local
            Reaults.append(r)       # 1,127,576

        return Reaults

class jointLayer(nn.Module):
    def __init__(self, in_channels=1536):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding="same", bias=True)
        self.NONLocalBlock2D = NONLocalBlock2D(in_channels=in_channels, sub_sample=True)

    def joint(self, x):
        x = rearrange(x, '(b v) c h w -> b v c h w', v=4)
        b, v, c, h, w = x.shape
        arr = x.clone()
        arr = arr.view(b, c, 2 * h, 2 * w)
        arr[:, :, :h, :w] = x[:, 0, :, :, :]
        arr[:, :, :h, w:2 * w] = x[:, 1, :, :, :]
        arr[:, :, h:2 * h, :w] = x[:, 2, :, :, :]
        arr[:, :, h:2 * h, w:2 * w] = x[:, 3, :, :, :]
        return arr

    def forward(self, x):
        x = self.joint(x)
        x = self.NONLocalBlock2D(x)

        return x

class FusionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.attn = FusionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        for i in range(1,5):
            self.add_module('mlp_' + str(i),Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop))
            self.add_module('norm2_' + str(i),norm_layer(dim))

        self.crossAttentionWithLearnableQuery = CrossAttentionWithLearnableQuery(dim, num_heads)

    def forward(self, X):  # x_t = rearrange(x_t, '(b v) c e -> b v c e', v=4)   # (B,197,576)
        X = rearrange(X, '(b v) c e -> b v c e', v=4)  # (B,197,576)
        X = self.attn(self.norm1(X))

        RESULTS = []
        for i in range(1,5):
            x = X[i-1]
            x = x + self.drop_path(x)
            x = x + self.drop_path(eval('self.mlp_'+str(i))(eval('self.norm2_'+str(i))(x)))
            RESULTS.append(x)

        r = self.crossAttentionWithLearnableQuery(RESULTS)
        return r

class WMIMVDR(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_views=4):
        print('new-model! ')
        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # assert depth % 3 == 0
        self.num_view = 4
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim )
        self.trans_cls_head = nn.Linear(embed_dim * 4, num_classes) if num_classes > 0 else nn.Identity()

        self.trans_norm_1 = nn.LayerNorm(embed_dim )
        self.trans_cls_head_1 = nn.Linear(embed_dim * 4, 2) if num_classes > 0 else nn.Identity()  # else nn.Identity() 不需要分类的话就直接保存原来的


        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        trans_dw_stride = patch_size // 4
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)

        stage_1_channel = int(base_channel * channel_ratio)
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)  # 这里是使用了残差

        for i in range(2,9):
            self.add_module('transformerDWTFusion_' + str(i),
                            TransformerDWTFusion(inplanes=64 * 3, outplanes=embed_dim,kernel_size=2,stride=2,
                             act_layer=nn.GELU,norm_layer=partial(nn.LayerNorm, eps=1e-6,),
                             embed_dim=embed_dim,  num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
                             drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,trans_dpr=self.trans_dpr[i - 1])
                            )

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.FusionBlock = FusionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=self.trans_dpr[i - 1])

        # 卷积

        self.conv_cls_head = nn.Linear(int(256 * 2), num_classes)
        self.jointLayer = jointLayer(512)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.dwt = DWT()

        for i in range(self.num_view):
            self.add_module('mv_conv1_' + str(i),
                            ConvBlock(inplanes=256, outplanes=512, stride=2, res_conv=True,
                                      act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                                      drop_block=None, drop_path=None)
                            )



        self.alpha = nn.Parameter(torch.ones(1))



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def joint(self, x):
        x = rearrange(x, '(b v) c h w -> b v c h w', v=4)
        b, v, c, h, w = x.shape
        arr = x
        arr = arr.view(b, c, 2 * h, 2 * w)
        arr[:, :, :h, :w] = x[:, 0, :, :, :]
        arr[:, :, :h, w:2 * w] = x[:, 1, :, :, :]
        arr[:, :, h:2 * h, :w] = x[:, 2, :, :, :]
        arr[:, :, h:2 * h, w:2 * w] = x[:, 3, :, :, :]
        return arr

    def _add(self, x):
        x = rearrange(x, 'b v c e -> b c (v e)', v=4)
        return x

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x = x_base

        # ---------------------------------- transformer ----------------
        x_t = self.trans_patch_conv(x_base)
        x_t = x_t.flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)

        for i in range(2, 9):
            x,x_t = eval('self.transformerDWTFusion_' + str(i))(x,x_t)

        x_t = self.FusionBlock(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        # ----------------------------------- 卷积的分支 -----------------------
        x_c = self.dwt(x)
        x_c = torch.cat(x_c, dim=1)
        x_c =  rearrange(x_c, '(b v) c h w -> b v c h w', v=4)
        mv_c = []
        for i in range(self.num_view):
            mv_ = eval('self.mv_conv1_' + str(i))(x_c[:, i])
            mv_c.append(mv_)
        x_c = torch.stack(mv_c, 1)
        x_c = rearrange(x_c, 'b v c h w -> (b v) c h w', v=4)
        x_c = self.jointLayer(x_c)
        # conv classification
        x_c = self.pooling(x_c).flatten(1)
        conv_cls = self.conv_cls_head(x_c)
        # --------- combine --------------
        cls = tran_cls + self.alpha * conv_cls

        return cls


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    model = WMIMVDR(patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=6, num_med_block=0,
                 embed_dim=576, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_views=4)
    cls = model.forward(x)
    print(cls.shape)
