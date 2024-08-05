# YOLOv5 common modules

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from torch.nn import init

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


###################################################################################
####################### clw note: 只用红外的输出
class CrossTransformerBlockForOnlyTir(nn.Module):    # clw note: CFE模块, 
    # def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlockForOnlyTir, self).__init__()
        self.loops = loops_num
        # self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.crossatt = CrossAttentionForOnlyTir(d_model, h, attn_pdrop, resid_pdrop)  # clw modify
        # self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
        #                              # nn.SiLU(),  # changed from GELU
        #                              nn.GELU(),  # changed from GELU
        #                              nn.Linear(block_exp * d_model, d_model),
        #                              nn.Dropout(resid_pdrop),
        #                              )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )

        # Layer norm
        # self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        # self.coefficient1 = LearnableCoefficient()
        # self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        # self.coefficient5 = LearnableCoefficient()
        # self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        # bs, nx, c = rgb_fea_flat.size()
        # h = w = int(math.sqrt(nx))

        for _ in range(self.loops):
            # with Learnable Coefficient
            #rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])   # clw modify: CFE模块只输出红外
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return ir_fea_flat
    


    
class TransformerFusionBlockForOnlyTir(nn.Module):              # clw modify： 只用红外特征分支（把可见光向红外融合）
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlockForOnlyTir, self).__init__()

        self.n_embd = d_model   # 512
        self.vert_anchors = vert_anchors   # 20
        self.horz_anchors = horz_anchors   # 20
        # d_k = d_model  # 512          # clw delete: no use, and value wrong，可以查看self.d_k
        # d_v = d_model  # 512

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))   # 20*20, 512 
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        # self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   # clw note: this has bug ??
        # self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        #self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlockForOnlyTir(d_model, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])  # clw modify

        # Concat
        # self.concat = Concat(dimension=1)

        # conv1x1
        #self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv1x1_out = Conv(c1=d_model, c2=d_model, k=1, s=1, p=0, g=1, act=True)   # clw modify TODO: 只用红外分支的CFE输出 
        

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):   # clw modify: 只使用红外分支的CFE模块
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        # import pdb; pdb.set_trace()
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        # import pdb; pdb.set_trace()
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis   # (h*w, c)，论文里的T_R

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  # (h*w, c)，论文里的T_

        # rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  # clw note: 这里相当于CFE模块在FFN前面的部分；
        ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  

        # rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        # if self.training == True:
        #     rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        # else:
        #     rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        # new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        # new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_ir_fea)
        
        return new_fea




class CrossAttentionForOnlyTir(nn.Module):       # clw modify: 只输出红外部分
    #def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
    def __init__(self, d_model, h, attn_pdrop=.1, resid_pdrop=.1):   # clw modify TODO
        '''
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        '''
        super(CrossAttentionForOnlyTir, self).__init__()
        # assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h   # clw note: Dimensionality of queries and keys,  512 // 8 = 64
        self.d_v = d_model // h   #             Dimensionality of values
        assert self.d_k % h == 0
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)   # clw note TODO：论文里没有写LN
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        ir_fea_flat = self.LN2(ir_fea_flat)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)

        # get attention matrix      
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)   # clw note TODO: 默认dropout参数0.1 ??

        # output
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)    # clw note TODO: 默认dropout参数0.1 ??

        return out_ir
#############################################################




#####################################################
################ clw note: 原版实现
class CrossAttention(nn.Module):   
    #def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
    def __init__(self, d_model, h, attn_pdrop=.1, resid_pdrop=.1):   # clw modify TODO
        '''
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        # assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h   # clw note: Dimensionality of queries and keys,  512 // 8 = 64
        self.d_v = d_model // h   #             Dimensionality of values
        assert self.d_k % h == 0
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)   # clw note TODO：论文里没有写LN
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)          # clw note TODO: 默认dropout参数0.1 ??
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):    # clw note: CFE模块
    # def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        # self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.crossatt = CrossAttention(d_model, h, attn_pdrop, resid_pdrop)  # clw modify
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        # bs, nx, c = rgb_fea_flat.size()
        # h = w = int(math.sqrt(nx))

        for _ in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            # rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN1(rgb_att_out)))   # clw modify TODO
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]


    
class TransformerFusionBlock(nn.Module):         # clw note: DMFF，原版实现； yolov5原版代码里面，三个stride特征层分别加了该模块，分别设置 [[512, 20, 20], [1024, 16, 16], [2048, 10, 10]]
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model   # 512
        self.vert_anchors = vert_anchors   # 20
        self.horz_anchors = horz_anchors   # 20
        # d_k = d_model  # 512          # clw delete: no use, and value wrong，可以查看self.d_k
        # d_v = d_model  # 512

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))   # 20*20, 512 
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   # clw note: origin version
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        #self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])  # clw modify

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, rgb_fea, ir_fea):
    # def forward(self, x):
        # rgb_fea = x[0]
        # ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis   # (h*w, c)，论文里的T_R

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  # (h*w, c)，论文里的T_

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  # clw note: 这里相当于CFE模块在FFN前面的部分；

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        # import pdb; pdb.set_trace()
        if self.training == True:
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')   # clw note TODO：为啥训练和推理插值方式不一样？并且训练用的低精度插值方式 nearest ?? 
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')     #              实测：训练时使用bilinear，推理时使用bilinear或者nearest精度基本没有差别
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#
        
        return new_fea




 
class TransformerFusionBlockV2(nn.Module):         # clw note: DMFF，原版实现； yolov5原版代码里面，三个stride特征层分别加了该模块，分别设置 [[512, 20, 20], [1024, 16, 16], [2048, 10, 10]]
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlockV2, self).__init__()

        self.n_embd = d_model   # 512
        self.vert_anchors = vert_anchors   # 20
        self.horz_anchors = horz_anchors   # 20

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))   # 20*20, 512 
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')   # clw note: origin version
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        #self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])  # clw modify


    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, rgb_fea, ir_fea):
    # def forward(self, x):
        # rgb_fea = x[0]
        # ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis   # (h*w, c)，论文里的T_R

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  # (h*w, c)，论文里的T_

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])  # clw note: 这里相当于CFE模块在FFN前面的部分；

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        # import pdb; pdb.set_trace()
        if self.training == True:
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')   # clw note TODO：为啥训练和推理插值方式不一样？并且训练用的低精度插值方式 nearest ?? 
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')     #              实测：训练时使用bilinear，推理时使用bilinear或者nearest精度基本没有差别
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
            # rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
            # ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        new_ir_fea = ir_fea_CFE + ir_fea

        return new_rgb_fea, new_ir_fea
############################################################################


    

class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

# class SE_Block(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(SE_Block, self).__init__()
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.ReLU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, h, w = x.size()
#         y = self.gap(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)

#         return x * y.expand_as(x)


# 通道注意力模块
# class Channel_Attention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         '''
#         :param in_channels: 输入通道数
#         :param reduction_ratio: 输出通道数量的缩放系数
#         :param pool_types: 池化类型
#         '''

#         super(Channel_Attention, self).__init__()

#         self.pool_types = pool_types
#         self.in_channels = in_channels
#         self.shared_mlp = nn.Sequential(nn.Flatten(),
#                                         nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
#                                         nn.ReLU(),
#                                         nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels)
#                                         )

#     def forward(self, x):
#         channel_attentions = []

#         for pool_types in self.pool_types:
#             if pool_types == 'avg':
#                 pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
#                 avg_pool = pool_init(x)
#                 channel_attentions.append(self.shared_mlp(avg_pool))
#             elif pool_types == 'max':
#                 pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
#                 max_pool = pool_init(x)
#                 channel_attentions.append(self.shared_mlp(max_pool))

#         pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
#         output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

#         return x * output


# 空间注意力模块
# class Spatial_Attention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(Spatial_Attention, self).__init__()

#         self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
#                                                nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
#                                                )

#     def forward(self, x):
#         x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
#         x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
#         scaled = nn.Sigmoid()(x_output)

#         return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
#         super(CBAM, self).__init__()

#         self.spatial = spatial
#         self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

#         if self.spatial:
#             self.spatial_attention = Spatial_Attention(kernel_size=7)

#     def forward(self, x):
#         x_out = self.channel_attention(x)
#         if self.spatial:
#             x_out = self.spatial_attention(x_out)

#         return x_out




#########################################################################  C2Former
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import einops


# class C2Former(nn.Module):

#     def __init__(
#             self, q_size, kv_size, n_heads, n_head_channels, n_groups,
#             attn_drop, proj_drop, stride,
#             offset_range_factor,
#             no_off, stage_idx
#     ):

#         super(C2Former,self).__init__()
#         self.n_head_channels = n_head_channels
#         self.scale = self.n_head_channels ** -0.5
#         self.n_heads = n_heads
#         self.q_h, self.q_w = q_size
#         self.kv_h, self.kv_w = kv_size
#         self.nc = n_head_channels * n_heads
#         self.qnc = n_head_channels * n_heads * 2
#         self.n_groups = n_groups
#         self.n_group_channels = self.nc // self.n_groups
#         self.n_group_heads = self.n_heads // self.n_groups
#         self.no_off = no_off
#         self.offset_range_factor = offset_range_factor

#         ksizes = [9, 7, 5, 3]
#         kk = ksizes[stage_idx]

#         self.conv_offset = nn.Sequential(
#             nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
#             LayerNormProxy(self.n_group_channels),
#             nn.GELU(),
#             nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
#         )

#         self.proj_q_lwir = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.proj_q_vis = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.proj_combinq = nn.Conv2d(
#             self.qnc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )

#         self.proj_k_lwir = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.proj_k_vis = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.proj_v_lwir = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.proj_v_vis = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )

#         self.proj_out_lwir = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.proj_out_vis = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#         self.vis_proj_drop = nn.Dropout(proj_drop, inplace=True)
#         self.lwir_proj_drop = nn.Dropout(proj_drop, inplace=True)
#         self.vis_attn_drop = nn.Dropout(attn_drop, inplace=True)
#         self.lwir_attn_drop = nn.Dropout(attn_drop, inplace=True)

#         self.vis_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)
#         self.lwir_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)


#         ########### clw modify
#         self.concat = Concat(dimension=1) 
#         self.conv1x1_out = Conv(c1=n_heads * n_head_channels * 2, c2=n_heads * n_head_channels, k=1, s=1, p=0, g=1, act=True)


#     @torch.no_grad()
#     def _get_ref_points(self, H_key, W_key, B, dtype, device):

#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
#             torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
#         )
#         ref = torch.stack((ref_y, ref_x), -1)
#         ref[..., 1].div_(W_key).mul_(2).sub_(1)
#         ref[..., 0].div_(H_key).mul_(2).sub_(1)
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

#         return ref

#     def forward(self, vis_x, lwir_x):

#         B, C, H, W = vis_x.size()
#         dtype, device = vis_x.dtype, vis_x.device
#         # concat two tensor
#         x = torch.cat([vis_x,lwir_x],1)
#         combin_q = self.proj_combinq(x)

#         q_off = einops.rearrange(combin_q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
#         offset = self.conv_offset(q_off) 
#         Hk, Wk = offset.size(2), offset.size(3)
#         n_sample = Hk * Wk

#         if self.offset_range_factor > 0:
#             offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
#             offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

#         offset = einops.rearrange(offset, 'b p h w -> b h w p')
#         vis_reference = self._get_ref_points(Hk, Wk, B, dtype, device)
#         lwir_reference = self._get_ref_points(Hk, Wk, B, dtype, device)

#         if self.no_off:
#             offset = offset.fill(0.0)

#         if self.offset_range_factor >= 0:
#             vis_pos = vis_reference + offset
#             lwir_pos = lwir_reference
#         else:
#             vis_pos = (vis_reference + offset).tanh()
#             lwir_pos = lwir_reference.tanh()

#         vis_x_sampled = F.grid_sample(
#             input=vis_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
#             grid=vis_pos[..., (1, 0)],  
#             mode='bilinear', align_corners=True)  

#         lwir_x_sampled = F.grid_sample(
#             input=lwir_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
#             grid=lwir_pos[..., (1, 0)],  
#             mode='bilinear', align_corners=True)  

#         vis_x_sampled = vis_x_sampled.reshape(B, C, 1, n_sample)
#         lwir_x_sampled = lwir_x_sampled.reshape(B, C, 1, n_sample)
        
#         q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
#         q_lwir = q_lwir.reshape(B * self.n_heads, self.n_head_channels, H * W)
#         k_vis = self.proj_k_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
#         v_vis = self.proj_v_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

#         q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
#         q_vis = q_vis.reshape(B * self.n_heads, self.n_head_channels, H * W)
#         k_lwir = self.proj_k_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
#         v_lwir = self.proj_v_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

#         attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis)  
#         attn_vis = attn_vis.mul(self.scale)
#         attn_vis = F.softmax(attn_vis, dim=2)
#         attn_vis = self.vis_attn_drop(attn_vis)
#         out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis)
#         out_vis = out_vis.reshape(B, C, H, W)
#         out_vis = self.vis_proj_drop(self.proj_out_vis(out_vis))

#         attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir)  
#         attn_lwir = attn_lwir.mul(self.scale)
#         attn_lwir = F.softmax(attn_lwir, dim=2)
#         attn_lwir = self.lwir_attn_drop(attn_lwir)
#         out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir)
#         out_lwir = out_lwir.reshape(B, C, H, W)
#         out_lwir = self.lwir_proj_drop(self.proj_out_lwir(out_lwir))

#         # return out_vis, out_lwir

#         ########### clw modify 
#         new_fea = self.concat([out_vis, out_lwir])
#         new_fea = self.conv1x1_out(new_fea)
#         return new_fea
    

# class LayerNormProxy(nn.Module):

#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         x = einops.rearrange(x, 'b c h w -> b h w c')
#         x = self.norm(x)
#         return einops.rearrange(x, 'b h w c -> b c h w')

# # Modality Norm
# class ModalityNorm(nn.Module):
#     def __init__(self, nf, use_residual=True, learnable=True):
#         super(ModalityNorm, self).__init__()

#         self.learnable = learnable
#         self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

#         if self.learnable:
#             self.conv= nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
#                                              nn.ReLU(inplace=True))
#             self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#             self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

#             self.use_residual = use_residual

#             # initialization
#             self.conv_gamma.weight.data.zero_()
#             self.conv_beta.weight.data.zero_()
#             self.conv_gamma.bias.data.zero_()
#             self.conv_beta.bias.data.zero_()

#     def forward(self, lr, ref):
#         ref_normed = self.norm_layer(ref)
#         if self.learnable:
#             x = self.conv(lr)
#             gamma = self.conv_gamma(x)
#             beta = self.conv_beta(x)

#         b, c, h, w = lr.size()
#         lr = lr.view(b, c, h * w)
#         lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
#         lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

#         if self.learnable:
#             if self.use_residual:
#                 gamma = gamma + lr_std
#                 beta = beta + lr_mean
#             else:
#                 gamma = 1 + gamma
#         else:
#             gamma = lr_std
#             beta = lr_mean

#         out = ref_normed * gamma + beta

#         return out