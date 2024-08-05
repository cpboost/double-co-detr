
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .layers import *
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
from .irnn import irnn
os.environ['CUDA_VISIBLE_DEVICES']='2'
class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)


class DispEstimator(nn.Module):
    def __init__(self,channel,depth=4,norm=nn.BatchNorm2d,dilation=1):
        super(DispEstimator,self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel,channel,3,act=None,norm=None,dilation=dilation,padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,act=None))
        #self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel+self.corrks**2
        dilation = 1
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
        #self.corrpropcessor = Conv2d(9+channel,channel,3,padding=1,bias=True,norm=nn.InstanceNorm2d)
        #self.AP3=nn.AvgPool2d(3,stride=1,padding=1)

    # def localcorr(self,feat1,feat2):
    #     feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
    #     feat1 = F.normalize(feat1,dim=1)
    #     feat2 = F.normalize(feat2,dim=1)
    #     b,c,h,w = feat2.shape
    #     feat2_smooth = KF.gaussian_blur2d(feat2,[9,9],[3,3])
    #     feat2_loc_blk = F.unfold(feat2_smooth,kernel_size=self.corrks,dilation=4,padding=4*(self.corrks-1)//2,stride=1).reshape(b,c,-1,h,w)
    #     localcorr = (feat1.unsqueeze(2)*feat2_loc_blk).sum(dim=1)
    #     localcorr = self.localcorrpropcessor(localcorr)
    #     corr = torch.cat([feat,localcorr],dim=1)
    #     return corr
    def localcorr(self,feat1,feat2):
        feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
        b,c,h,w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1,(13,13),(3,3),border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth,kernel_size=self.corrks,dilation=4,padding=2*(self.corrks-1),stride=1).reshape(b,c,-1,h,w)
        localcorr = (feat2.unsqueeze(2)-feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat,localcorr],dim=1)
        return corr

    def forward(self,feat1,feat2):
        b,c,h,w = feat1.shape
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0,1,0,0] != w-1 or self.scale[0,0,0,0] != h-1:
            self.scale = torch.FloatTensor([w,h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1,feat2)
        for i,layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr,(13,13),(3,3),border_type='replicate')
        disp = corr.clamp(min=-300,max=300)
        # print(disp.shape)
        # print(feat1.shape)
        return disp/self.scale

class DispRefiner(nn.Module):
    def __init__(self,channel,dilation=1,depth=4):
        super(DispRefiner,self).__init__()
        self.preprocessor = nn.Sequential(Conv2d(channel,channel,3,dilation=dilation,padding=dilation,norm=None,act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,norm=None,act=None))
        oc = channel
        ic = channel+2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)
    def forward(self,feat1,feat2,disp):
        
        b=feat1.shape[0]
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b],feat[b:]],dim=1))
        corr = torch.cat([feat,disp],dim=1)
        delta_disp = self.estimator(corr)
        disp = disp+delta_disp
        return disp 
        

class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x


class DenseMatcher(nn.Module):
    def __init__(self,unshare_depth=4,matcher_depth=4,num_pyramids=2):
        super(DenseMatcher,self).__init__()
        self.num_pyramids=num_pyramids
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth,base_ic=3,base_oc=8,base_dilation=1,norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth,base_ic=3,base_oc=8,base_dilation=1,norm=nn.InstanceNorm2d)
        #self.feature_extractor_unshare2 = self.feature_extractor_unshare1
        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        base_dilation = self.feature_extractor_unshare1.dilation
        self.feature_extractor_share1 = nn.Sequential(Conv2d(base_oc,base_oc*2,kernel_size=3,stride=1,padding=1,dilation=1, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*2,base_oc*2,kernel_size=3,stride=2,padding=1,dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(Conv2d(base_oc*2,base_oc*4,kernel_size=3,stride=1,padding=2,dilation=2, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*4,base_oc*4,kernel_size=3,stride=2,padding=2,dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(Conv2d(base_oc*4,base_oc*8,kernel_size=3,stride=1,padding=4,dilation=4, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*8,base_oc*8,kernel_size=3,stride=2,padding=4,dilation=4, norm=nn.InstanceNorm2d))
        self.matcher1 = DispEstimator(base_oc*4,matcher_depth,dilation=4)
        self.matcher2 = DispEstimator(base_oc*8,matcher_depth,dilation=2)
        self.refiner = DispRefiner(base_oc*2,1)
        self.grid_down = KU.create_meshgrid(64,64).cuda()
        self.grid_full = KU.create_meshgrid(128,128).cuda()
        self.scale = torch.FloatTensor([128,128]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1

    def match(self,feat11,feat12,feat21,feat22,feat31,feat32):
        #compute scale (w,h)
        if self.scale[0,1,0,0]*2 != feat11.shape[2]-1 or self.scale[0,0,0,0]*2 != feat11.shape[3]-1:
            self.h,self.w = feat11.shape[2],feat11.shape[3]
            self.scale = torch.FloatTensor([self.w,self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1 
            self.scale = self.scale.to(feat11.device)

        #estimate disp src(feat1) to tgt(feat2) in low resolution
        disp2_raw = self.matcher2(feat31,feat32) 
        
        #upsample disp and grid
        disp2 = F.interpolate(disp2_raw,[feat21.shape[2],feat21.shape[3]],mode='bilinear')
        if disp2.shape[2] != self.grid_down.shape[1] or disp2.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat21.shape[2],feat21.shape[3]).cuda()

        #warp the last src(fea1) to tgt(feat2) with disp2
        feat21 = F.grid_sample(feat21,self.grid_down+disp2.permute(0,2,3,1))

        #estimate disp src(feat1) to tgt(feat2) in low resolution
        disp1_raw = self.matcher1(feat21,feat22)

        #upsample
        disp1 = F.interpolate(disp1_raw,[feat11.shape[2],feat11.shape[3]],mode='bilinear')
        disp2 = F.interpolate(disp2,[feat11.shape[2],feat11.shape[3]],mode='bilinear')
        if disp1.shape[2] != self.grid_full.shape[1] or disp1.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2],feat11.shape[3]).cuda()

        #warp
        feat11 = F.grid_sample(feat11,self.grid_full+(disp1+disp2).permute(0,2,3,1))

        #finetune
        disp_scaleup = (disp1+disp2)*self.scale
        disp = self.refiner(feat11,feat12,disp_scaleup)
        disp = KF.gaussian_blur2d(disp,(17,17),(5,5),border_type='replicate')/self.scale
        if self.training:
            return disp,disp_scaleup/self.scale,disp2
        return disp,None,None    
        
    def forward(self,src,tgt,type='ir2vis'):
        b,c,h,w = tgt.shape
        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)
        feat0 = torch.cat([feat01,feat02])
        feat1 = self.feature_extractor_share1(feat0)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat11,feat12 = feat1[0:b],feat1[b:]
        feat21,feat22 = feat2[0:b],feat2[b:]
        feat31,feat32 = feat3[0:b],feat3[b:]
        disp_12 = None
        disp_21 = None
        if type == 'bi':
            disp_12,disp_12_down4,disp_12_down8 = self.match(feat11,feat12,feat21,feat22,feat31,feat32)
            disp_21,disp_21_down4,disp_21_down8 = self.match(feat12,feat11,feat22,feat21,feat32,feat31)
            t = torch.cat([disp_12,disp_21,disp_12_down4,disp_21_down4,disp_12_down8,disp_21_down8])
            t = F.interpolate(t,[h,w],mode='bilinear')
            down2,down4,donw8 = torch.split(t,2*b,dim=0)
            disp_12_,disp_21_ = torch.split(down2,b,dim=0)
        elif type == 'ir2vis':
            disp_12,_,_= self.match(feat11,feat12,feat21,feat22,feat31,feat32)
            disp_12 = F.interpolate(disp_12,[h,w],mode='bilinear')
        elif type =='vis2ir':
            disp_21,_,_ = self.match(feat12,feat11,feat22,feat21,feat32,feat31)
            disp_21 = F.interpolate(disp_21,[h,w],mode='bilinear')
        if self.training:
            return {'ir2vis':disp_12_,'vis2ir':disp_21_,
            'down2':down2,
            'down4':down4,
            'down8':donw8}    
        return {'ir2vis':disp_12,'vis2ir':disp_21}

class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=0.2):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.IRNN = irnn()

    def forward(self, input):
        output = self.IRNN.apply(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight,
                      self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias,
                      self.left_weight.bias)
        return output


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        model = []
        out_channels = int(in_channels / 2)
        model += [ConvLeakyRelu2d(2 * in_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, 4, activation='Sigmod', kernel_size=3, padding=1, stride=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels)
        self.irnn2 = Spacial_IRNN(self.out_channels)
        self.conv_in = ConvLeakyRelu2d(2 * in_channels, in_channels, activation=None)
        self.conv2 = ConvLeakyRelu2d(in_channels * 4, in_channels, activation=None, kernel_size=3, padding=1, stride=1)
        self.conv3 = ConvLeakyRelu2d(in_channels * 4, in_channels, kernel_size=3, padding=1, stride=1)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = ConvLeakyRelu2d(in_channels, in_channels, activation='Sigmod', kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            # print('top_up device:', top_up.device, 'weight device:', weight.device)
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        mask = self.conv_out(out)
        return mask

class FusionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FusionNet, self).__init__()
        channels = [8, 16, 16, 32]
        encoder_ir = []
        encoder_ir += [ConvLeakyRelu2d(in_channels, channels[0])]
        encoder_ir += [ConvLeakyRelu2d(channels[0], channels[1])]
        encoder_ir += [ConvLeakyRelu2d(channels[1], channels[2])]
        encoder_ir += [ConvLeakyRelu2d(channels[2], channels[3])]
        self.encoder_ir = nn.Sequential(*encoder_ir)

        encoder_vi = []
        encoder_vi += [ConvLeakyRelu2d(in_channels, channels[0])]
        encoder_vi += [ConvLeakyRelu2d(channels[0], channels[1])]
        encoder_vi += [ConvLeakyRelu2d(channels[1], channels[2])]
        encoder_vi += [ConvLeakyRelu2d(channels[2], channels[3])]
        self.encoder_vi = nn.Sequential(*encoder_vi)

        decoder = []
        decoder += [ConvLeakyRelu2d(channels[3], channels[2])]
        decoder += [ConvLeakyRelu2d(channels[2], channels[1])]
        decoder += [ConvLeakyRelu2d(channels[1], channels[0])]
        decoder += [ConvLeakyRelu2d(channels[0], out_channels,activation='Tanh')]

        self.decoder = nn.Sequential(*decoder)
        self.SAM = SAM(channels[3], channels[3], 1)
        # self.SAM_vi = SAM(channels[3], channels[3], 1)

    def forward(self, image_ir,image_vi, eps=1e-6):
        # split data into RGB and INF
        features_ir = self.encoder_ir(image_ir)
        features_vi = self.encoder_vi(image_vi)        
        attention_ir = self.SAM(torch.cat([features_ir, features_vi], dim=1))
        # attention_vi = self.SAM_vi(features_vi)
        # features_fused = attention_ir * features_ir + (1- attention_ir) * features_vi 
        # features_fused = features_ir * (attention_ir / (attention_vi + attention_ir)) + features_vi * (attention_vi / (attention_vi + attention_ir))
        features_fused = features_ir.mul(attention_ir) + features_vi.mul(1 - attention_ir)
        image_fused = self.decoder(features_fused)
        image_fused = (image_fused+1)/2
        return image_fused

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass


if __name__ == '__main__':
    matcher = DenseMatcher().cuda()
    ir = torch.rand(2,3,512,512).cuda()
    vis = torch.rand(2,3,512,512).cuda()
    disp=matcher(ir,vis,'bi')
