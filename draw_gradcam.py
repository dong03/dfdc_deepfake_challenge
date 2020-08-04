# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re
import json
import pdb
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation

from training.zoo.classifiers import DeepFakeClassifier
from torch.backends import cudnn
from training.datasets.classifier_dataset import DeepFakeClassifierDataset, collate_function
from albumentations import Compose, PadIfNeeded
from training.transforms.albu import create_val_transforms
from training.tools.utils import AverageMeter, read_annotations, Progbar, predict_set, evaluate

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb,转换前冷色调激活，转换后暖色调激活

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.cpu().numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, output_dir):

    for key, image in image_dicts.items():
        cv2.imwrite(os.path.join(output_dir, '{}_{}'.format(key, input_image_name)), image)


def parse_args():
    parser = argparse.ArgumentParser(description='draw gradcam')

    parser.add_argument('--draw_list', type=str, default='/data/dongchengbo/VisualSearch/dfdc_dfv2_ff++_timit_withface_val.txt')
    parser.add_argument('--model_dir', type=str, default='/data/dongchengbo/code/dfdc_1st_Vdcb/weights/fix_lr0.01_decay0.8DeepFakeClassifier_tf_efficientnet_b7_ns_0_best_dice')
    parser.add_argument('--save_dir', type=str, default='/data/dongchengbo/VisualSearch/output')
    parser.add_argument('--img_size', default=380, type=int, help='resize scale')
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()
    return args


def main(argv=None):
    opt = parse_args()
    print(json.dumps(vars(opt), indent=4))

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    index = None
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    net = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
    print("loading state dict {}".format(opt.model_dir))
    checkpoint = torch.load(opt.model_dir, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
    net.eval()
    net.cuda()

    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    draw_set = DeepFakeClassifierDataset(
            annotations=read_annotations(opt.draw_list)[:150],
            mode="val",
            balance=False,
            transforms=create_val_transforms(380))

    draw_loader = DataLoader(
        dataset=draw_set,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_function
    )

    # net = torchvision.models.resnet50(pretrained=True).to(opt.device)
    # layer_name = 'layer4'
    layer_name = get_last_conv_name(net)
    print("layer name: %s" % layer_name)
    grad_cam = GradCAM(net, layer_name,(opt.img_size,opt.img_size))
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name,(opt.img_size,opt.img_size))
    gbp = GuidedBackPropagation(net)

    progbar = Progbar(len(draw_set))
    batch_time = AverageMeter()
    end = time.time()

    for inner, (labels, inputs, name) in enumerate(draw_loader):
        name = name[0]
        img = inputs[0]
        for i in range(3):
            img[i] = img[i] * std[i] + mean[i]
        img = ToPILImage()(img)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) / 255.0
        inputs = inputs.cuda()
        inputs = inputs.requires_grad_(True)
        labels = labels.item()
        # 输出图像
        image_dict = {}

        # Grad-CAM
        mask = grad_cam(inputs, index)  # cam mask
        image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
        grad_cam.remove_handlers()

        # Grad-CAM++
        mask_plus_plus = grad_cam_plus_plus(inputs, index)  # cam mask
        image_dict['campp'], image_dict['heatmappp'] = gen_cam(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()

        # GuidedBackPropagation
        inputs.grad.zero_()  # 梯度置零
        grad = gbp(inputs)
        gb = gen_gb(grad)
        image_dict['gb'] = norm_image(gb)

        # 生成Guided Grad-CAM
        cam_gb = gb * mask[..., np.newaxis]
        image_dict['cam_gb'] = norm_image(cam_gb)

        save_image(image_dict, '%d_%s'%(labels,os.path.split(name)[-1]), save_dir)
        progbar.add(1, values=[('batch_time', batch_time.val)])
        batch_time.update(time.time() - end)
        end = time.time()


if __name__ == '__main__':
    main()
