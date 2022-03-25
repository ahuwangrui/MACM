#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
from posixpath import dirname
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='models/lip/lip.pth', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='input', help="path of input image folder.")
    parser.add_argument("--mask-dir", type=str, default='mask', help="path of mask image folder.")
    parser.add_argument("--output-dir", type=str, default='output', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 255
        palette[j * 3 + 1] = 255
        palette[j * 3 + 2] = 255

        while lab:
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0

            lab >>= 3
    return palette

def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    palette = get_palette(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(args.mask_dir, img_name[:-4] + '_mask.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            # maskimg =cv2.cvtColor(np.asarray(output_img),cv2.COLOR_GRAY2BGR)
            output_img.save(parsing_result_path)
            # mask = os.path.join(args.mask_dir, img_name[:-4] + '_mask.png')
            # # mask1 = os.path.join(args.output_dir, img_name[:-4] + '_1_mask.png')
            # # mask2 = os.path.join(args.output_dir, img_name[:-4] + '_2_mask.png')
            # # mask3 = os.path.join(args.output_dir, img_name[:-4] + '_3_mask.png')
            # # mask4 = os.path.join(args.output_dir, img_name[:-4] + '_4_mask.png')
            # # mask5 = os.path.join(args.output_dir, img_name[:-4] + '_5_mask.png')
            # # # cv2.imwrite(mask, maskimg)
            # src = os.path.join(args.input_dir, img_name[:-4] + '.jpg')
            # res = os.path.join(args.output_dir, img_name[:-4] + '_mask.jpg')
            # img_mask = cv2.imread(mask, cv2.IMREAD_COLOR)
            # # img_mask1 = cv2.imread(mask1, cv2.IMREAD_COLOR)
            # # img_mask2 = cv2.imread(mask2, cv2.IMREAD_COLOR)
            # # img_mask3 = cv2.imread(mask3, cv2.IMREAD_COLOR)
            # # img_mask4 = cv2.imread(mask4, cv2.IMREAD_COLOR)
            # # img_mask5 = cv2.imread(mask5, cv2.IMREAD_COLOR)
            # # img_mask = img_mask / 6 + img_mask1 / 6 + img_mask2 / 6 + img_mask3 / 6 + img_mask4 / 6 + img_mask5 / 6
            # # img_mask = img_mask > 250
            # # # print(img_mask[0])
            # # img_mask = img_mask.astype(np.uint8) * 255
            # # # print(img_mask[0])
            # # # img_mask = img_mask+img_mask1
            # # # gray_img = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            # # # gray_img = gray_img+gray_img
            # # # img_mask = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            # # # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            # # # ret, img_mask = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # # # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
            # # # print(img_mask[0])

            # # cv2.imwrite(res, img_mask)
            # # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            # # ret, img_mask = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
            # # print(ret,img_mask[0])
            # # print(img_mask.shape)
            # # img_mask = np.concatenate((img_mask, img_mask, img_mask), axis=-1)
            # img_src = cv2.imread(src, cv2.IMREAD_COLOR)
            # # print(img_src.shape)

            # img_res = cv2.add(img_src, img_mask)
            # # img_src.copyTo(img_src, img_mask)
            # cv2.imwrite(res, img_res)
            # os.chdir(args.output_dir)
            if args.logits:
                logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)
    return

def getrgb():
    args = get_arguments()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.mask_dir):
        os.makedirs(args.mask_dir)
    dir_name = args.input_dir
    index_channel = [[0, 1, 2],
                     [0, 2, 1],
                     [1, 0, 2],
                     [1, 2, 0],
                     [2, 0, 1],
                     [2, 1, 0]]
    fullname_list, filename_list = [], []
    for root, dirs, files in os.walk(dir_name):
        for filename in files:
            # 文件名列表，包含完整路径
         if 'mask' not in filename and 'bk' not in filename:
           ori=os.path.join(root,filename)
           imgo=Image.open(ori)
           for i in range(5):
                img_3channel = imgo.split()
                img = Image.merge('RGB', (img_3channel[index_channel[i + 1][0]], img_3channel[index_channel[i + 1][1]],
                                          img_3channel[index_channel[i + 1][2]]))
                res = os.path.join(ori[:-4] + '_rgb' + str(i + 1) + '.jpg')
                img.save(res, quality=100)

def getmask():
    args = get_arguments()
    dir_name = args.input_dir
   
    fullname_list, filename_list = [], []
    # print(mask_name)
    for root, dirs, files in os.walk(dir_name):
        for filename in tqdm(files):
            # 文件名列表，包含完整路径
         if 'rgb' not in filename :
                # print(os.path.abspath(r"."))
                mask0 = os.path.join(args.mask_dir, filename[:-4] + '_mask.png')
                mask1 = os.path.join(args.mask_dir, filename[:-4] + '_rgb1_mask.png')
                mask2 = os.path.join(args.mask_dir, filename[:-4] + '_rgb2_mask.png')
                mask3 = os.path.join(args.mask_dir, filename[:-4] + '_rgb3_mask.png')
                mask4 = os.path.join(args.mask_dir, filename[:-4] + '_rgb4_mask.png')
                mask5 = os.path.join(args.mask_dir, filename[:-4] + '_rgb5_mask.png')
                # print(mask)
                # # cv2.imwrite(mask, maskimg)
                src = os.path.join(root, filename[:-4] + '.jpg')
                res = os.path.join(args.output_dir, filename[:-4] + '_mask.jpg')
                img_mask0 = cv2.imread(mask0, cv2.IMREAD_COLOR)
                img_mask1 = cv2.imread(mask1, cv2.IMREAD_COLOR)
                img_mask2 = cv2.imread(mask2, cv2.IMREAD_COLOR)
                img_mask3 = cv2.imread(mask3, cv2.IMREAD_COLOR)
                img_mask4 = cv2.imread(mask4, cv2.IMREAD_COLOR)
                img_mask5 = cv2.imread(mask5, cv2.IMREAD_COLOR)
                img_mask = img_mask0 / 6 + img_mask1 / 6 + img_mask2 / 6 + img_mask3 / 6 + img_mask4 / 6 + img_mask5 / 6
                os.remove(mask0)
                os.remove(mask1)
                os.remove(mask2)
                os.remove(mask3)
                os.remove(mask4)
                os.remove(mask5)
                img_mask = img_mask > 250
                img_mask = img_mask.astype(np.uint8) * 255
                img_src = cv2.imread(src, cv2.IMREAD_COLOR)
                img_res = cv2.add(img_src, img_mask)
                # img_src.copyTo(img_src, img_mask)
                cv2.imwrite(res, img_res)
                cv2.imwrite(mask0,img_mask)
         else:
               deldir = os.path.join(root, filename[:-4] + '.jpg')
               os.remove(deldir)


if __name__ == '__main__':
    getrgb()
    main()
    getmask()
