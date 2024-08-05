import torch
from collections import OrderedDict

pretrain_pth = '../data/pretrain_model/co_dino_5scale_vit_large_coco.pth'

save_path = pretrain_pth[:-4] + '_for_rgbt.pth'


print('====================== modify pretrained pth for rgbt two stream start !!')

state_dicts = torch.load(pretrain_pth)
if 'state_dict' in state_dicts:
    state_dicts = state_dicts['state_dict']

new_state_dicts = OrderedDict()
for k, v in state_dicts.items():
    if 'backbone.' in k:
        new_state_dicts[k.replace('backbone.', 'backbone_vis.')] = v
        new_state_dicts[k.replace('backbone.', 'backbone_lwir.')] = v
    elif 'neck.' in k:
        new_state_dicts[k.replace('neck.', 'neck_vis.')] = v
        new_state_dicts[k.replace('neck.', 'neck_lwir.')] = v
        new_state_dicts[k] = v
    else:
        new_state_dicts[k] = v

torch.save(new_state_dicts, save_path)


print('====================== modify pretrained pth for rgbt two stream end !!')



