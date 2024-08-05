import os
import torch
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

# tensor to PIL Image


def tensor2img(img):
    img = img[0].cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


def tensor2content(content):
    img = content[0].cpu().float().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img.astype(np.uint8)

# save a set of images


def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = Image.fromarray(img)
        img.save(os.path.join(path, name + '.png'))


def save_img_single(img, name):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(name)


def save_content(content, dir):
    for i in range(content.shape[1]):
        sub_content = tensor2content(content[:, i, :, :])
        img = Image.fromarray(sub_content)
        img.save(os.path.join(dir, '%03d.jpg' % (i + 1)))


class Saver():
    def __init__(self, opts):
        self.display_dir = os.path.join(opts.display_dir, opts.name)
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if not callable(
                getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)
            # write img
            image_dis = torchvision.utils.make_grid(
                model.image_display, nrow=model.image_display.size(0)//2)
            self.writer.add_image('Image', image_dis, total_it)

    # save result images

    def write_img(self, ep, model, stage='RF'):
        if ep % self.img_save_freq == 0:
            if stage == 'FS':
                assembled_images = model.assemble_outputs1()
            else:
                assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(
                assembled_images, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(
                assembled_images, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model, best=False):
        #assert False
              
        if best:
            mode_save_path ='%s/%s.pth' % (self.model_dir, 'Best_RegFusion')
        else:
            if ep % self.model_save_freq == 0:
                if ep > total_it*0.9:
                    model.save('{}/{}_{}.pth'.format(self.model_dir, 'RegFusion',ep), ep, total_it)
                print('--- save the model @ ep %d ---' % (ep))
                mode_save_path = '%s/%s.pth' % (self.model_dir, 'RegFusion')
            elif ep == total_it or ep == total_it-1:
                print('--- save the model @ ep %d ---' % (ep))
                mode_save_path = '%s/%s.pth' % (self.model_dir, 'RegFusion')
            elif ep == -1:
                print('--- save the model @ ep %d ---' % (ep))
                mode_save_path = '%s/%s.pth' % (self.model_dir, 'RegFusion')
        model.save(mode_save_path, ep, total_it)
            
        return mode_save_path
