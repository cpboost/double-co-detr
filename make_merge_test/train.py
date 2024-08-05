import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from options import TrainOptions
from dataset import *
from model import SuperFusion
from utils.saver import Saver
from utils.logger import *
from utils.fullevaluate import MscEval
from modules.TII import *

from torch.utils.data import DataLoader
from Fusion import run_fusion


def evaluate(seg_model=None, logger=None):
    respth = opts.result_dir
    Method = opts.name
    if logger == None:
        logger = logging.getLogger()
    respth = os.path.join(respth, Method)
    # model
    logger.info('\n')
    logger.info('====' * 4)
    logger.info('evaluating the model ...')
    logger.info('setup and restore model')

    # dataset
    batchsize = 1
    n_workers = 2
    dspth = opts.dataroot_val
    dsval = CityScapes(dspth)
    dl = DataLoader(
        dsval,
        batch_size=batchsize,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    # evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(seg_model, dl)
    # eval
    mIOU = evaluator.evaluate(Method=Method)
    logger.info('mIOU is: {:.6f}'.format(mIOU))
    return mIOU

def main_FS(opts):    
    dataset_name = 'MSRS'
    log_dir = os.path.join(opts.result_dir, opts.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')    
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logger_config(log_path=log_path, logging_name='Timer')

    # Load pre-trained segmentation model for computing semantic loss
    n_classes = 9
    seg_model = BiSeNet(n_classes=n_classes)
    pretrained_model_path = './checkpoint/Segmentation/Seg_model.pth'
    seg_model.load_state_dict(torch.load(pretrained_model_path))
    seg_model.cuda(opts.gpu)
    seg_model.eval()
    logger.info('Load Segmentation Model from {} Sucessfully~'.format(pretrained_model_path))
    # daita loader
    print('\n--- load dataset ---')
    if dataset_name == 'MSRS':
        dataset = MSRSData(opts)
    else:
        dataset = RoadSceneData(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = SuperFusion(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    # print('start the training at epoch %d' % (ep0))
    logger.info('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)
    model_save_path = opts.resume
    # train
    print('\n--- train ---')
    max_it = 500000
    best_mIoU = 0
    
    for ep in range(ep0, opts.n_ep):
        if dataset_name == "MSRS":
            for it, (image_ir, image_vi, label) in enumerate(train_loader):
                # input data
                image_ir = image_ir.cuda(opts.gpu).detach()
                image_vi = image_vi.cuda(opts.gpu).detach()
                label = label.cuda(opts.gpu).detach()
                if len(image_ir.shape) > 4:
                    image_ir = image_ir.squeeze(1)
                    image_vi = image_vi.squeeze(1)
                    label = label.squeeze(1)

                # update model
                model.update_FS(image_ir, image_vi, seg_model, label, dataset_name)                
                # save to display file
                if not opts.no_display_img:
                    saver.write_display(total_it, model)

                if (total_it + 1) % 10 == 0:
                    Intensity_loss = model.loss_intensity
                    SSIM_loss = model.loss_ssim
                    Fusion_loss = model.loss_fus
                    Grad_loss = model.loss_grad
                    Seg_loss = model.loss_seg
                    Total_loss = model.loss_total
                    logger.info('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                        total_it, ep, it, model.FN_opt.param_groups[0]['lr'], Total_loss))
                    logger.info(
                        'Intensity_loss: {:.4}, SSIM_loss: {:.4}, Grad_loss: {:.4}, Fusion_loss: {:.4}, Seg_loss:{:.4} \n'.format(
                            Intensity_loss, SSIM_loss, Grad_loss, Fusion_loss, Seg_loss))
                total_it += 1
        else:
            for it, (image_ir, image_vi) in enumerate(train_loader):
                # input data
                image_ir = image_ir.cuda(opts.gpu).detach()
                image_vi = image_vi.cuda(opts.gpu).detach()
                if len(image_ir.shape) > 4:
                    image_ir = image_ir.squeeze(1)
                    image_vi = image_vi.squeeze(1)
                # update model
                model.update_FS(image_ir, image_vi, seg_model)

                # save to display file
                if not opts.no_display_img:
                    saver.write_display(total_it, model)

                if (total_it + 1) % 10 == 0:
                    Intensity_loss = model.loss_intensity
                    SSIM_loss = model.loss_ssim
                    Fusion_loss = model.loss_fus
                    Grad_loss = model.loss_grad
                    Seg_loss = model.loss_seg
                    Total_loss = model.loss_total
                    logger.info('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                        total_it, ep, it, model.FN_opt.param_groups[0]['lr'], Total_loss))
                    logger.info(
                        'Intensity_loss: {:.4}, SSIM_loss: {:.4}, Grad_loss: {:.4}, Fusion_loss: {:.4}, Seg_loss:{:.4}'.format(
                            Intensity_loss, SSIM_loss, Grad_loss, Fusion_loss, Seg_loss))

                total_it += 1

        if (ep + 1) % opts.model_save_freq == 0:
            saver.write_img(total_it, model, stage=opts.stage)
            model_save_path = saver.write_model(ep, total_it, model)
        if (ep + 1) % opts.model_save_freq == 0:
            # logger = logging.getLogger() 
            
            if dataset_name == 'MSRS':
                run_fusion(dataset_name, model, save_dir='./dataset/test/MSRS/fused')
                mIoU = evaluate(seg_model, logger)
                logger.info("|Calculate mIoU Sucessfully, The mIoU is {:.6}!".format(mIoU))
                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    saver.write_model(ep, opts.n_ep, model, best=True)

        if opts.n_ep_decay > -1:
            model.update_lr()

    return


def main_RF(opts):
    # daita loader
    print('\n--- load dataset ---')
    dataset = RegData(opts)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = SuperFusion(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):

        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in enumerate(train_loader):
            # input data
            image_ir = image_ir.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_ir_warp = image_ir_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_ir.shape) > 4:
                image_ir = image_ir.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_ir_warp = image_ir_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)
            # update model
            model.update_RF(image_ir, image_vi, image_ir_warp,
                            image_vi_warp, deformation)

            # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            if (total_it + 1) % 10 == 0:
                Reg_Img_loss = model.loss_reg_img
                Reg_Field_loss = model.loss_reg_field
                Fusion_loss = model.loss_fus
                NCC_loss = model.loss_ncc
                Total_loss = model.loss_total
                print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                    total_it, ep, it, model.DM_opt.param_groups[0]['lr'], Total_loss))
                print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, NCC_loss: {:.4}, Fusion_loss: {:.4}'.format(
                    Reg_Img_loss, Reg_Field_loss, NCC_loss, Fusion_loss))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(total_it, model)
                saver.write_model(total_it, model)
                break
        print(ep)
        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        saver.write_model(ep, opts.n_ep, model)

    return


if __name__ == '__main__':
    parser = TrainOptions()
    opts = parser.parse()
    if opts.stage == 'RF':
        main_RF(opts)
    else:
        main_FS(opts)
