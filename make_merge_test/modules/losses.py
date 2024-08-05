# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
import torchvision
import kornia.filters as KF
# from loss_ssim import ssim
shape = (256, 256)

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    #print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map*mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq

    
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=1):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        mask = torch.logical_and(img1>0,img2>0).float()
        for i in range(self.window_size//2):
            mask = (F.conv2d(mask, window, padding=self.window_size//2, groups=channel)>0.8).float()
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask=mask)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

ssimloss = SSIMLoss(window_size=11)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())

        return contentloss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def forward(self, I, J, win=[15]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims
        sum_filt = torch.ones([1, I.shape[1], *win]).cuda()/I.shape[1]
        pad_no = math.floor(win[0] / 2)
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)
        cc = cross * cross / ((I_var * J_var).clamp(min=1e-3) + 1e-3)
        return -1 * torch.mean(cc)




# def similarity_loss(tgt, warped_img):

#     sizes = np.prod(list(tgt.shape)[1:])
#     flatten1 = torch.reshape(tgt, (-1, sizes))
#     flatten2 = torch.reshape(warped_img, (-1, sizes))

#     mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
#     mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
#     var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
#     var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
#     cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
#     pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))
#     raw_loss = torch.sum(1 - pearson_r)

#     return raw_loss

def l1loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,[3,3],[1,1])*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],[1,1])*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).mean()

def l2loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,[3,3],[1,1])*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],[1,1])*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
    def forward(self,img1,img2,mask=1,eps=1e-2):
        #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        #l = l[...,5:-5,10:-10].mean()
        return l

def smoothloss(disp,img=None):
    smooth_d=[3*3,7*3,15*3]
    b,c,h,w = disp.shape
    grad = KF.spatial_gradient(disp,order=2).abs().sum(dim=2)[:,:,5:-5,5:-5].clamp(min=1e-9).mean()
    local_smooth_re = 0
    for d in smooth_d:
        local_mean = KF.gaussian_blur2d(disp,[d,d],[d//6,d//6],border_type='replicate')
        #local_mean_pow2 = F.avg_pool2d(disp.pow(2),kernel_size=d,stride=1,padding=d//2)
        local_smooth_re += 1/(d*1.0+1)*(disp-local_mean)[:,:,d//2:-d//2,d//2:-d//2].pow(2).mean()
        #local_smooth_re += 1/(d*1.0+1)*(disp.pow(2)-local_mean_pow2)[:,:,5:-5,5:-5].pow(2).mean()
    #global_var = disp[...,2:-2,2:-2].var(dim=[-1,-2]).clamp(1e-5).mean()
    #std = img.std(dim=[-1,-2]).mean().clamp(min=0.003)
    #grad = grad[...,10:-10,10:-10]
    return 5000*local_smooth_re + 500*grad

def l2regularization(img):
    return img.pow(2).mean()
# def l1loss(img1,img2,mask=1,eps=1e-2):
#     img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
#     img2 = KF.gaussian_blur2d(img2,[7,7],[2,2])
#     return ((img1-img2)*mask).abs().clamp(min=eps).mean()

# def l2loss(img1,img2,mask=1,eps=1e-2):
#     img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
#     img2 = KF.gaussian_blur2d(img2,[7,7],[2,2])
#     return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

# class gradientloss(nn.Module):
#     def __init__(self):
#         super(gradientloss,self).__init__()
#         self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
#         self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
#     def forward(self,img1,img2,mask=1,eps=1e-3):
#         #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
#         #img2 = KF.gaussian_blur2d(img2,[7,7],[2,2])
#         grad1 = KF.spatial_gradient(img1,order=2).abs().sum(dim=[1,2])
#         grad2 = KF.spatial_gradient(img2,order=2).abs().sum(dim=[1,2])
#         # grad1 = self.AP5(self.MP5(grad1))
#         # grad2 = self.AP5(self.MP5(grad2))
#         l = ((grad1-grad2)*mask).abs().clamp(min=eps).mean()
#         return l

# def smoothloss(img):
#     grad = KF.spatial_gradient(img,order=2).mean(dim=1).abs().sum(dim=1)
#     return grad.clamp(min=1e-2,max=0.5).mean()
# a = torch.rand(1,2,256,256)
# a[:,1]=0
# smoothloss(a)
def orthogonal_loss(t):
    # C=A'A, a positive semi-definite matrix
    # should be close to I. For this, we require C
    # has eigen values close to 1
    c = torch.matmul(t, t)
    k = torch.linalg.eigvals(c)  # Get eigenvalues of C
    ortho_loss = torch.mean((k[0][0] - 1.0) ** 2) + torch.mean((k[0][1] - 1.0) ** 2)
    ortho_loss = ortho_loss.float()
    return ortho_loss


def determinant_loss(t):
    # Determinant Loss: determinant should be close to 1
    det_value = torch.det(t)
    det_loss = torch.sum((det_value - 1.0) ** 2)/2
    return det_loss

def smoothness_loss(deformation, img=None, alpha=0.0):
    """Calculate the smoothness loss of the given defromation field

    :param deformation: the input deformation
    :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    :param alpha: the alpha coefficient used in the bilateral filtering.
    :return:
    """
    diff_1 = torch.abs(deformation[:, :, 1::, :] - deformation[:, :, 0:-1, :])
    diff_2 = torch.abs((deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1]))
    diff_3 = torch.abs(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1::, 1::])
    diff_4 = torch.abs(deformation[:, :, 0:-1, 1::] - deformation[:, :, 1::, 0:-1])
    if img is not None and alpha > 0.0:
        mask = img
        weight_1 = torch.exp(-alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
        weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
        weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
        weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
        weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    else:
        weight_1 = weight_2 = weight_3 = weight_4 = 1.0
    loss = torch.mean(weight_1 * diff_1) + torch.mean(weight_2 * diff_2) \
           + torch.mean(weight_3 * diff_3) + torch.mean(weight_4 * diff_4)
    return loss


def feat_loss(feat1,feat2,grid=16):
    b,c,h,w=feat1.shape[0],feat1.shape[1],feat1.shape[2],feat1.shape[3]
    shift_x = np.random.randint(1,w//grid)
    shift_y = np.random.randint(1,h//grid)
    x = tuple(np.arange(grid)*w//grid+shift_x)
    y = tuple(np.arange(grid)*w//grid+shift_y)
    feat1_sampled = feat1[:,:,y,:]
    feat1_sampled = F.normalize(feat1_sampled[:,:,:,x],dim=1).view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    feat2_sampled = feat2[:,:,y,:]
    feat2_sampled = F.normalize(feat2_sampled[:,:,:,x],dim=1).view(b,c,-1).permute(0,2,1).contiguous().view(-1,c)
    # .view(b,c,-1).permute(0,2,1).view(-1,c)
    featset = torch.cat([feat1_sampled,feat2_sampled])
    perseed = torch.randperm(featset.shape[0])
    featset = featset[perseed][0:feat1_sampled.shape[0]]
    simi_pos = (feat1_sampled*feat2_sampled).sum(dim=-1)
    simi_neg = (feat1_sampled*featset).sum(dim=-1) if torch.rand(1)>0.5 else (feat2_sampled*featset).sum(dim=-1)
    loss = (simi_neg-simi_pos+0.5).clamp(min=0.0).mean()
    return loss

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes,weight=1)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n



