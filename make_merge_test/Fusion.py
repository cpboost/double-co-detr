import torch
import os
from tqdm import tqdm
from time import time
from utils.utils import *
from dataset import TestData, imsave


def run_fusion(dataset_name='MSRS', model=None, save_dir=None):
    if dataset_name == 'MSRS':
        img_path = "./dataset/test/MSRS"
        if save_dir is not None:
            save_dir = './dataset/test/MSRS/fused'
    elif dataset_name == 'RoadScene':
        img_path = "./dataset/test/RoadScene"
        if save_dir is not None:
            save_dir = './dataset/test/RoadScene/fused'
    ir_path = os.path.join(img_path, 'ir')
    vi_path = os.path.join(img_path, 'vi')
    os.makedirs(save_dir, exist_ok=True)
    test_dataloader = TestData(ir_path, vi_path)
    model.eval()
        
    p_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for idx, [ir, vi, name] in p_bar:
        ir_tensor = ir.cuda()
        vi_tensor = vi.cuda()
        start = time()
        with torch.no_grad():
            fu = model.fusion_forward(ir_tensor,vi_tensor)
        test_time = time() - start
        imsave(fu, os.path.join(save_dir, name))
        p_bar.set_description(f'fusing {name} | time : {str(test_time)}')
    model.train()