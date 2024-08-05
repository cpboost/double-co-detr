import warnings

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
from mmcv.runner import auto_fp16

from .common import TransformerFusionBlock, TransformerFusionBlockForOnlyTir
# from .common import C2Former

TWO_STREAM_SHARE_NECK = True
# TWO_STREAM_SHARE_NECK = False

@DETECTORS.register_module()
class TwoStreamCoDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',   # clw note: 默认是detr，精度也最优
                 eval_index=0):
        super(TwoStreamCoDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        # Module for evaluation, ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module
        # Module index for evaluation
        self.eval_index = eval_index

        #############################  # clw modify
        self.backbone_vis = build_backbone(backbone)  
        self.backbone_lwir = build_backbone(backbone)




        if TWO_STREAM_SHARE_NECK:
            self.neck = build_neck(neck)   # clw note: 先各自过neck， or 融合后过一个neck
        else:
            ############# 方式1 ###############
            self.neck_vis = build_neck(neck)
            self.neck_lwir = build_neck(neck)
            ##################################

            ############# 方式2： backbone和neck都融合 ###############
            # self.neck = build_neck(neck)
            # self.neck_lwir = build_neck(neck)
            ##################################


        if TWO_STREAM_SHARE_NECK:
            # # fmap_size=(160, 160)
            # fmap_size=(40, 40)   # for VIT，640x640 input
            # # dims_out=[96, 192, 384, 768]
            # dims_out=[1024]
            # # num_heads=[3, 6, 12, 24]
            # # num_heads=[3]
            # num_heads=[8]
            # # cca_strides=[3, 3, 3, 3]
            # cca_strides=[3]
            # # groups=[1, 2, 3, 6]
            # groups=[1]
            # # offset_range_factor=[2, 2, 2, 2]
            # offset_range_factor=[2]
            # # no_offs=[False, False, False, False]
            # no_offs=[False]
            # attn_drop_rate=0.0
            # drop_rate=0.0
            # i = 0
            # hc = dims_out[i] // num_heads[i]
            # self.tfb_blocks = nn.ModuleList([
            #     C2Former(fmap_size, fmap_size, num_heads[i], hc, groups[i], attn_drop_rate, drop_rate, cca_strides[i], offset_range_factor[i], no_offs[i], i)
            # ])
            
            self.tfb_blocks = nn.ModuleList([     # (256, 64, 80), (512, 32, 40), (1024, 16, 20), (2048, 8, 10)           对于yolo640输入，原始特征图80, 40, 20 -> 20, 16, 10
                
                # TransformerFusionBlock(192, 64, 80),   # clw note: 128, 160 may cause oom
                # ##### TransformerFusionBlock(192, 96, 120),  
                # TransformerFusionBlock(384, 64, 80),    #           
                # TransformerFusionBlock(768, 32, 40),    #            
                # TransformerFusionBlock(1536, 16, 20), 

                # TransformerFusionBlock(192, 64, 80),   # backbone s4
                # TransformerFusionBlock(384, 32, 40),    #           s8
                # TransformerFusionBlock(768, 16, 20),    #            s16
                # TransformerFusionBlock(1536, 8, 10),    #             s32



                # TransformerFusionBlockForOnlyTir(192, 64, 80),   # backbone s4
                # TransformerFusionBlockForOnlyTir(384, 32, 40),    #           s8
                # TransformerFusionBlockForOnlyTir(768, 16, 20),    #            s16
                # TransformerFusionBlockForOnlyTir(1536, 8, 10),   

                ############# for VIT, 
                TransformerFusionBlock(1024, 40, 40),    # 640 input
                # TransformerFusionBlock(1024, 80, 80),    # 1280 input
                # TransformerFusionBlock(1024, 48, 48),    # 1536 input
            ])

        else:
            ######################## 方式1：只neck融合 ##########
            self.tfb_blocks = nn.ModuleList([
                TransformerFusionBlock(256, 80, 80),  
                TransformerFusionBlock(256, 40, 40),  
                TransformerFusionBlock(256, 20, 20),  
                TransformerFusionBlock(256, 10, 10),  
                TransformerFusionBlock(256, 5, 5),  
            ])
            ##############################


            ######################## 方式2：backbone, neck都融合 ##########
            # self.tfb_blocks_backbone = nn.ModuleList([     # (256, 64, 80), (512, 32, 40), (1024, 16, 20), (2048, 8, 10)           对于yolo640输入，原始特征图80, 40, 20 -> 20, 16, 10
            #     TransformerFusionBlock(1024, 40, 40),  
            # ])
                        
            # self.tfb_blocks = nn.ModuleList([
            #     TransformerFusionBlock(256, 80, 80),  
            #     TransformerFusionBlock(256, 40, 40),  
            #     TransformerFusionBlock(256, 20, 20),  
            #     TransformerFusionBlock(256, 10, 10),  
            #     TransformerFusionBlock(256, 5, 5),  
            # ])
            ##############################


                        
        ##############################

        # self.backbone = build_backbone(backbone)
        # if neck is not None:
        #     self.neck = build_neck(neck)
            
        head_idx = 0

        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i+head_idx].rcnn if (train_cfg and train_cfg[i+head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i+head_idx].rcnn)
                self.roi_head.append(build_head(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(train_cfg=train_cfg[i+head_idx+len(self.roi_head)] if (train_cfg and train_cfg[i+head_idx+len(self.roi_head)] is not None) else None)
                bbox_head[i].update(test_cfg=test_cfg[i+head_idx+len(self.roi_head)])
                self.bbox_head.append(build_head(bbox_head[i]))  
                self.bbox_head[-1].init_weights() 

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head)>0))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0 and self.roi_head[0].with_mask)
    
    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    ########################
    def extract_visfeat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_vis(img)
        # if self.with_neck:  
        x = self.neck_vis(x)
        return x
    
    def extract_lwirfeat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_lwir(img)   # [(1, 192, 128, 160), (1, 384, 64, 80), ..., (1, 1536, 16, 20)]
        # if self.with_neck:
        x = self.neck_lwir(x)
        return x
    #########################




    def forward_test(self, imgs, img_lwirs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_lwirs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_lwirs, img_metas, **kwargs)


    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_lwir, img_metas, return_loss=True, **kwargs):   # clw modify TODO: img_lwir
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_lwir, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_lwir, img_metas, **kwargs)
        

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.query_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_lwir,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]



        # x = self.extract_feat(img, img_metas)

        ########################## clw modify： 先融合，后过neck
        if TWO_STREAM_SHARE_NECK: 
            x_vis = self.backbone_vis(img)   # 对于VIT backbone： x_vis[0].shape: list -> (1, 1024, 40, 40)
            x_lwir = self.backbone_lwir(img_lwir)
            x = []
            # import pdb; pdb.set_trace()
            for i in range(len(x_vis)):   # VIT只有一层输出，torch.Size([1, 1024, h//16, w//16])
                #x.append(0.5 * (x_vis[i] + x_lwir[i]))
                # x.append(self.tfb_blocks[i]( [x_vis[i], x_lwir[i]] ))
                x.append(self.tfb_blocks[i]( x_vis[i], x_lwir[i] ))
            x = tuple(x)
            x = self.neck(x)    # (1, 256, h//4, w//4) -> (1, 256, h//64, w//64)
            
        else:
            ##################### 方式1：##############
            x_vis = self.extract_visfeat(img)  
            x_lwir = self.extract_lwirfeat(img_lwir)
            x = []
            for i in range(len(x_vis)):
                #x.append(0.5 * (x_vis[i] + x_lwir[i]))
                x.append(self.tfb_blocks[i]( x_vis[i], x_lwir[i] ))
            x = tuple(x)
            ###########################################


            ##################### 方式2：backbone和neck都融合 （实测效果不好........） ##################
            # x_vis = self.backbone_vis(img)
            # x_lwir = self.backbone_lwir(img_lwir)
            # x = []
            # for i in range(len(x_vis)):   # VIT只有一层输出，torch.Size([1, 1024, h//16, w//16])
            #     #x.append(0.5 * (x_vis[i] + x_lwir[i]))
            #     x.append(self.tfb_blocks_backbone[i]( [x_vis[i], x_lwir[i]] ))
            # x = tuple(x)
            
            # x_fusion = self.neck(x)
            # x_lwir = self.neck_lwir(x_lwir)
            # # import pdb; pdb.set_trace()
            # x_after = []
            # for i in range(len(x_lwir)):
            #     x_after.append(self.tfb_blocks[i]( x_fusion[i], x_lwir[i] ))
            # x = tuple(x_after)
            #########################################


       ############################


        losses = dict()
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(x, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)
            

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                              self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else: 
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')     
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
            
        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                        gt_labels, gt_bboxes_ignore)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')          
            bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords)>0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)                    

        return losses


    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_lwir, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        # x = self.extract_feat(img, img_metas)
                
        ########################################## clw modify TODO


        if TWO_STREAM_SHARE_NECK:   # 先融合，后过neck
            x_vis = self.backbone_vis(img)
            x_lwir = self.backbone_lwir(img_lwir)
            x = []
            for i in range(len(x_vis)):
                #x.append(0.5 * (x_vis[i] + x_lwir[i]))
                x.append(self.tfb_blocks[i]( x_vis[i], x_lwir[i] ))
            x = tuple(x)
            x = self.neck(x)
        else:
            ##################### 方式1：##############
            x_vis = self.extract_visfeat(img)  
            x_lwir = self.extract_lwirfeat(img_lwir)
            x = []
            for i in range(len(x_vis)):
                #x.append(0.5 * (x_vis[i] + x_lwir[i]))
                x.append(self.tfb_blocks[i]( x_vis[i], x_lwir[i] ))
            x = tuple(x)
            ###########################################


            ##################### 方式2： backbone和neck都融合 ##################
            # x_vis = self.backbone_vis(img)
            # x_lwir = self.backbone_lwir(img_lwir)
            # x = []
            # for i in range(len(x_vis)):   # VIT只有一层输出，torch.Size([1, 1024, h//16, w//16])
            #     #x.append(0.5 * (x_vis[i] + x_lwir[i]))
            #     x.append(self.tfb_blocks_backbone[i]( [x_vis[i], x_lwir[i]] ))
            # x = tuple(x)
            
            # x_fusion = self.neck(x)
            # x_lwir = self.neck_lwir(x_lwir)
            # x_after = []
            # for i in range(len(x_lwir)):
            #     x_after.append(self.tfb_blocks[i]( x_fusion[i], x_lwir[i] ))
            # x = tuple(x_after)
            #########################################

        ########################################



        results_list = self.query_head.simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask: # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, img, img_lwir, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module=='one-stage':
            return self.simple_test_bbox_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module=='two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_query_head(img, img_lwir, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.query_head.forward_onnx(x, img_metas)[:2]
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        # TODO support NMS
        # det_bboxes, det_labels = self.query_head.onnx_export(
        #     *outs, img_metas, with_nms=with_nms)
        det_bboxes, det_labels = self.query_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels