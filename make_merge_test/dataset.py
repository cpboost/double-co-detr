import os
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import torch
from utils.utils import randflow, randrot, randfilp
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import json
from natsort import natsorted
import kornia.utils as KU


class CityScapes(Dataset):
    def __init__(
        self,
        rootpth,
        cropsize=(640, 480),
        mode='train',
        *args,
        **kwargs
    ):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255

        with open('./utils/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        
        # impth = osp.join(rootpth, Method, mode)
        self.img_dir = os.path.join(rootpth, 'fused')
        self.label_dir = os.path.join(rootpth, 'label')
        self.file_list = natsorted(os.listdir(self.img_dir))       

        ## pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fn = self.file_list[idx]
        impth = os.path.join(self.img_dir, fn)
        lbpth = os.path.join(self.label_dir, fn)
        img = Image.open(impth)
        label = Image.open(lbpth)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label, fn

    def __len__(self):
        return len(self.file_list)
class RegData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, opts, crop=lambda x: x):
        super(RegData, self).__init__()
        self.vis_folder = os.path.join(opts.dataroot, 'vi')
        self.ir_folder = os.path.join(opts.dataroot, 'ir')
        self.crop = torchvision.transforms.RandomCrop(256)
        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        print(len(self.vis_list), len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])

        assert os.path.basename(vis_path) == os.path.basename(ir_path), f"Mismatch ir:{os.path.basename(ir_path)} vi:{os.path.basename(vis_path)}."

        # read image as type Tensor
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)

        vis_ir = torch.cat([vis,ir],dim=1)
        if vis_ir.shape[-1]<=256 or vis_ir.shape[-2]<=256:
            vis_ir=TF.resize(vis_ir,256)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)

        flow,disp,_ = randflow(vis_ir,10,0.1,1)
        vis_ir_warped = F.grid_sample(vis_ir, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([vis_ir,vis_ir_warped,disp.permute(0,3,1,2)], dim=1)
        patch = self.crop(patch)

        vis, ir, vis_warped, ir_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h,w = vis_ir.shape[2],vis_ir.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        #print(self.crop.size[0])
        disp = disp.permute(0,2,3,1)*scale
        #vis_warped_ = self.ST(vis.unsqueeze(0),disp.unsqueeze(0))
        #TF.to_pil_image(((vis_warped-vis_warped_).abs()).squeeze(0)).save('error.png')
        #disp_crop = disp[:,h//2-150:h//2+150,w//2-150:w//2+150,:]*scale
        return ir, vis, ir_warped, vis_warped, disp

    def __len__(self):
        return len(self.vis_list)


    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        # im_cv = cv2.imread(str(path), flags)
        # assert im_cv is not None, f"Image {str(path)} is invalid."
        # im_ts = kornia.utils.image_to_tensor(im_cv / 255.,keepdim=False).type(torch.FloatTensor)
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class MSRSData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, opts, crop=lambda x: x):
        super(MSRSData, self).__init__()
        self.vis_folder = os.path.join(opts.dataroot, 'vi')
        self.ir_folder = os.path.join(opts.dataroot, 'ir')
        self.label_folder = os.path.join(opts.dataroot, 'label')
        self.crop = torchvision.transforms.RandomCrop(256)
        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        self.label_list = sorted(os.listdir(self.label_folder))
        print(len(self.vis_list), len(self.ir_list), len(self.label_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name)
        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        label = self.imread(path=label_path, label=True)

        vis_ir = torch.cat([vis, ir, label], dim=1)
        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)
        patch = self.crop(vis_ir)

        vis, ir, label = torch.split(patch, [3, 3, 1], dim=1)
        h, w = vis_ir.shape[2], vis_ir.shape[3]
        label = label.type(torch.LongTensor)
        return ir, vis, label

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, label=False):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            img = Image.open(path).convert('RGB')
            im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class RoadSceneData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, opts, crop=lambda x: x):
        super(RoadSceneData, self).__init__()
        self.vis_folder = os.path.join(opts.dataroot, 'vi')
        self.ir_folder = os.path.join(opts.dataroot, 'ir')
        self.crop = torchvision.transforms.RandomCrop(256)
        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        print(len(self.vis_list), len(self.ir_list))
        # self.ST = SpatialTransformer(self.crop.size[0],self.crop.size[0],False)

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)

        vis_ir = torch.cat([vis, ir], dim=1)
        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)
        patch = self.crop(vis_ir)

        vis, ir = torch.split(patch, [3, 3], dim=1)
        return ir, vis

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, label=False):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            img = Image.open(path).convert('RGB')
            im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class TestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_dir=None, vi_dir=None):
        super(TestData, self).__init__()
        self.vis_folder = vi_dir
        self.ir_folder = ir_dir
        # gain infrared and visible images list
        self.ir_list = natsorted(os.listdir(self.ir_folder))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        return ir, vis, image_name

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        im_ts = KU.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.).float()
        im_ts = im_ts.unsqueeze(0)
        return im_ts


def imsave(img, filename):
    img = img.squeeze().cpu()
    img = KU.tensor_to_image(img) * 255.
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
