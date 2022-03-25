# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import datasets
import random
import math
import torch
import cv2
#############################################################################################################
# Channel_Dataset: It is used to get image pairs, which have the same sketch and different contents
# Parameters
#         ----------
#         domain_num: The number of augmented samples for each original one
# -----------------------------------------------------------------------------------------------------------


#############################################################################################################
# RandomErasing: Executing random erasing on input data
# Parameters
#         ----------
#         probability: The probability that the Random Erasing operation will be performed
#         sl: Minimum proportion of erased area against input image
#         sh: Maximum proportion of erased area against input image
#         r1: Minimum aspect ratio of erased area
#         mean: Erasing value
# -----------------------------------------------------------------------------------------------------------
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class ChannelTripletFolder(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=6, train=True):
        super(ChannelTripletFolder, self).__init__(root, transform)
        self.domain_num = domain_num
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root
    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]


    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        cam = self.cams[index]
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        pos_path = self._get_pos_sample(label, index)
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            img_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                img_sub.append(img)
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2 = self.data[index_two[0]], self.labels[index_two[0]].item()
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img


    def __len__(self):
        return len(self.imgs)


#add mask and camstyle aug data and cam=6 for market
class ChannelShuffling(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=7, train=True,cam_num=6):
        super(ChannelShuffling, self).__init__(root, transform)
        self.domain_num = domain_num
        # self.labels = np.array(self.imgs)[:, 1]
        # self.data = np.array(self.imgs)[:, 0]
        self.cam_num = cam_num
        self.data = []
        self.labels = []
        self.cam = []
        for s in self.samples:
            if 'mask' not in s[0] and 'bk' not in s[0] and 'fake' not in s[0]:
                self.data.append(s[0])
                self.labels.append(s[1])
                self.cam.append(self._get_cam_id(s[0]))
        self.data = np.array(self.data)
        self.labels =  np.array(self.labels)
        # self.cams = np.array(self.cam)
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root
        self.pos_index = np.argwhere(self.labels == 0)


    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) 


    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        # pos_index1 = pos_index1.tolist()
        # pos_index=[]
        # for s in pos_index1:
        #     # x =pos_index1[s]
        #     if 'mask' not in self.samples[s][0]:
        #         pos_index.append(s)
        # pos2=pos_index
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.data[tmp_index])
        # kk =0
        # while kk<4:
        #     for j in range(len(rand)):
        #         t = j % len(rand)
        #         tmp_index = pos_index[rand[t]]
        #         if 'mask' not in self.samples[tmp_index][0]:
        #             result_path.append(self.samples[tmp_index][0])
        #             kk=kk+1
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def _getpath(self, path):
        if 'mask' in path:

            path = path[:-9] + '.jpg'
            return path
        else:
            return path
    def __getitem__(self, index):

      # if 'mask' not in self.data[index]:
        img, label = self.data[index], self.labels[index]
        cam = self.cam[index]


        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        #获取4个正样本的图像路径
        pos_path = self._get_pos_sample(label, index)
        pos_path[0] = self._getpath(pos_path[0])
        pos_path[1] = self._getpath(pos_path[1])
        pos_path[2] = self._getpath(pos_path[2])
        pos_path[3] = self._getpath(pos_path[3])
        wr=img
        
        # wr.append(pos_path[0] )
        # wr.append(pos_path[1] )
        # wr.append(pos_path[2] )
        # wr.append(pos_path[3] )
        # 获取样本对应的mask图像路径

        mask_path = img[:-4] + '_mask' + '.jpg'
        mask0_path = pos_path[0][:-4] + '_mask' + '.jpg'
        mask1_path = pos_path[1][:-4] + '_mask' + '.jpg'
        mask2_path = pos_path[2][:-4] + '_mask' + '.jpg'
        mask3_path = pos_path[3][:-4] + '_mask' + '.jpg'

        maskp = []
        maskp.append(mask_path)
        maskp.append(mask0_path)
        maskp.append(mask1_path)
        maskp.append(mask2_path)
        maskp.append(mask3_path)

        

        #加载当前样本及其4个正样本图像
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])


        #加载mask图像
        mask=[]
        maskk = self.loader(mask_path)
        mask0 = self.loader(mask0_path)
        mask1 = self.loader(mask1_path)
        mask2 = self.loader(mask2_path)
        mask3 = self.loader(mask3_path)
        mask.append(maskk)
        mask.append(mask0)
        mask.append(mask1)
        mask.append(mask2)
        mask.append(mask3)




        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            # back =cv2.imread(maskp[i], cv2.IMREAD_COLOR)
            img_sub = []
            for j in range(self.domain_num-1):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                # ahead =cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                # img = cv2.bitwise_and(ahead, back)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # idd =str(j)
                # res = os.path.join(bk[i][:-7] +'_'+idd+ '_cat.jpg')
                # cv2.imwrite(res, img_res)

                img_sub.append(img)
            img_sub.append(mask[i])
            for k in range(self.cam_num):
                if k+1 != cam:
                    cam_path=wr[:-4] + '_fake_' + str(cam) + 'to' + str(k+ 1) + '.jpg'
                    camaug = self.loader(cam_path)
                    img_sub.append(camaug)

       
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num+self.cam_num-1, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num+self.cam_num-1):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2,cam2 = self.data[index_two[0]], self.labels[index_two[0]],self.cam[index_two[0]]
        wr2=img2
        if 'mask' in img2:
           mask_path2 =img2
           img2 = img2[:-9] + '.jpg'
        else:
            mask_path2 = img2[:-4] + '_mask' + '.jpg'
        mask2 = []
        mask2.append(mask_path2)
        bk2=[]
        bk_path2= img2[:-4] + '_bk' + '.jpg'
        bk2.append((bk_path2))
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num+self.cam_num-1).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            back2 = cv2.imread(bk2[i], cv2.IMREAD_COLOR)
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num-1):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                # ahead = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                # img = cv2.bitwise_and(ahead, back2)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            img_mask2=default_loader(mask_path2)
            img_sub.append(img_mask2)
            label_sub.append(int(label))
            label_all2[i][6] =int(label)+  self.class_num * 6
            for k in range(self.cam_num):
              if k+1 != cam2:
                cam_path=wr2[:-4] + '_fake_' + str(cam2) + 'to' + str(k+ 1) + '.jpg'
                camaug = self.loader(cam_path)
                img_sub.append(camaug)
            # for kk in range(self.cam_num - 1):
            #     label_all2[i][kk + 7] = int(label)
            #different label for every cam
            for kk in range(self.cam_num-1):
                 label_all2[i][kk+7] =int(label)+  self.class_num * (kk+7)
            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num+self.cam_num-1, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num+self.cam_num-1):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img



    def __len__(self):
        return len(self.data)

#without mask and camstyle
class ChannelTripletFolderold(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=5, train=True):
        super(ChannelTripletFolderold, self).__init__(root, transform)
        self.domain_num = domain_num
        # self.labels = np.array(self.imgs)[:, 1]
        # self.data = np.array(self.imgs)[:, 0]
        self.data = []
        self.labels = []
        for s in self.samples:
            if 'mask'not in s[0] and 'bk' not in s[0] and 'fake' not in s[0]:
                self.data.append(s[0])
                self.labels.append(s[1])
        self.data = np.array(self.data)
        self.labels =  np.array(self.labels)

        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root
        self.pos_index = np.argwhere(self.labels == 0)


    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1


    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        # pos_index1 = pos_index1.tolist()
        # pos_index=[]
        # for s in pos_index1:
        #     # x =pos_index1[s]
        #     if 'mask' not in self.samples[s][0]:
        #         pos_index.append(s)
        # pos2=pos_index
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.data[tmp_index])
        # kk =0
        # while kk<4:
        #     for j in range(len(rand)):
        #         t = j % len(rand)
        #         tmp_index = pos_index[rand[t]]
        #         if 'mask' not in self.samples[tmp_index][0]:
        #             result_path.append(self.samples[tmp_index][0])
        #             kk=kk+1
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def _getpath(self, path):
        if 'mask' in path:

            path = path[:-9] + '.jpg'
            return path
        else:
            return path
    def __getitem__(self, index):

      # if 'mask' not in self.data[index]:
        img, label = self.data[index], self.labels[index]
        cam = self.cams[index]


        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        #获取4个正样本的图像路径
        pos_path = self._get_pos_sample(label, index)
        pos_path[0] = self._getpath(pos_path[0])
        pos_path[1] = self._getpath(pos_path[1])
        pos_path[2] = self._getpath(pos_path[2])
        pos_path[3] = self._getpath(pos_path[3])
        # wr=[]
        # wr.append(img)
        # wr.append(pos_path[0] )
        # wr.append(pos_path[1] )
        # wr.append(pos_path[2] )
        # wr.append(pos_path[3] )
        # 获取样本对应的mask图像路径

        mask_path = img[:-4] + '_mask' + '.jpg'
        mask0_path = pos_path[0][:-4] + '_mask' + '.jpg'
        mask1_path = pos_path[1][:-4] + '_mask' + '.jpg'
        mask2_path = pos_path[2][:-4] + '_mask' + '.jpg'
        mask3_path = pos_path[3][:-4] + '_mask' + '.jpg'

        maskp = []
        maskp.append(mask_path)
        maskp.append(mask0_path)
        maskp.append(mask1_path)
        maskp.append(mask2_path)
        maskp.append(mask3_path)

        # 获取样本对应的背景图像路径

        bk_path = img[:-4] + '_bk' + '.jpg'
        bk0_path = pos_path[0][:-4] + '_bk' + '.jpg'
        bk1_path = pos_path[1][:-4] + '_bk' + '.jpg'
        bk2_path = pos_path[2][:-4] + '_bk' + '.jpg'
        bk3_path = pos_path[3][:-4] + '_bk' + '.jpg'
        bkp = []
        bkp.append(bk_path)
        bkp.append(bk0_path)
        bkp.append(bk1_path)
        bkp.append(bk2_path)
        bkp.append(bk3_path)
        bk = []
        bkk = self.loader(bk_path)
        bk0 = self.loader(bk0_path)
        bk1 = self.loader(bk1_path)
        bk2 = self.loader(bk2_path)
        bk3 = self.loader(bk3_path)
        bk.append(bkk)
        bk.append(bk0)
        bk.append(bk1)
        bk.append(bk2)
        bk.append(bk3)

        #加载当前样本及其4个正样本图像
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])


        #加载mask图像
        mask=[]
        maskk = self.loader(mask_path)
        mask0 = self.loader(mask0_path)
        mask1 = self.loader(mask1_path)
        mask2 = self.loader(mask2_path)
        mask3 = self.loader(mask3_path)
        mask.append(maskk)
        mask.append(mask0)
        mask.append(mask1)
        mask.append(mask2)
        mask.append(mask3)




        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            # back =cv2.imread(maskp[i], cv2.IMREAD_COLOR)
            img_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                # ahead =cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                # img = cv2.bitwise_and(ahead, back)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # idd =str(j)
                # res = os.path.join(bk[i][:-7] +'_'+idd+ '_cat.jpg')
                # cv2.imwrite(res, img_res)

                img_sub.append(img)
            # img_sub.append(mask[i])
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2 = self.data[index_two[0]], self.labels[index_two[0]]
        if 'mask' in img2:
           mask_path2 =img2
           img2 = img2[:-9] + '.jpg'
        else:
            mask_path2 = img2[:-4] + '_mask' + '.jpg'
        mask2 = []
        mask2.append(mask_path2)
        bk2=[]
        bk_path2= img2[:-4] + '_bk' + '.jpg'
        bk2.append((bk_path2))
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            # back2 = cv2.imread(bk2[i], cv2.IMREAD_COLOR)
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                # ahead = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                # img = cv2.bitwise_and(ahead, back2)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            # img_mask2=default_loader(mask_path2)
            # img_sub.append(img_mask2)
            # label_sub.append(int(label))
            # label_all2[i][self.domain_num-1] =int(label)

            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img



    def __len__(self):
        return len(self.data)


#add mask and camstyle aug data and cam=8 for duke
class ChannelShuffling1(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=7, train=True, cam_num=8):
        super(ChannelShuffling1, self).__init__(root, transform)
        self.domain_num = domain_num
        # self.labels = np.array(self.imgs)[:, 1]
        # self.data = np.array(self.imgs)[:, 0]
        self.cam_num = cam_num
        self.data = []
        self.labels = []
        self.cam = []
        for s in self.samples:
            if 'mask' not in s[0] and 'bk' not in s[0] and 'fake' not in s[0]:
                self.data.append(s[0])
                self.labels.append(s[1])
                self.cam.append(self._get_cam_id(s[0]))
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root
        self.pos_index = np.argwhere(self.labels == 0)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id)

    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        # pos_index1 = pos_index1.tolist()
        # pos_index=[]
        # for s in pos_index1:
        #     # x =pos_index1[s]
        #     if 'mask' not in self.samples[s][0]:
        #         pos_index.append(s)
        # pos2=pos_index
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.data[tmp_index])
        # kk =0
        # while kk<4:
        #     for j in range(len(rand)):
        #         t = j % len(rand)
        #         tmp_index = pos_index[rand[t]]
        #         if 'mask' not in self.samples[tmp_index][0]:
        #             result_path.append(self.samples[tmp_index][0])
        #             kk=kk+1
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def _getpath(self, path):
        if 'mask' in path:

            path = path[:-9] + '.jpg'
            return path
        else:
            return path

    def __getitem__(self, index):

        # if 'mask' not in self.data[index]:
        img, label = self.data[index], self.labels[index]
        cam = self.cam[index]

        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        # 获取4个正样本的图像路径
        pos_path = self._get_pos_sample(label, index)
        pos_path[0] = self._getpath(pos_path[0])
        pos_path[1] = self._getpath(pos_path[1])
        pos_path[2] = self._getpath(pos_path[2])
        pos_path[3] = self._getpath(pos_path[3])
        wr = img

        # wr.append(pos_path[0] )
        # wr.append(pos_path[1] )
        # wr.append(pos_path[2] )
        # wr.append(pos_path[3] )
        # 获取样本对应的mask图像路径

        mask_path = img[:-4] + '_mask' + '.jpg'
        mask0_path = pos_path[0][:-4] + '_mask' + '.jpg'
        mask1_path = pos_path[1][:-4] + '_mask' + '.jpg'
        mask2_path = pos_path[2][:-4] + '_mask' + '.jpg'
        mask3_path = pos_path[3][:-4] + '_mask' + '.jpg'

        maskp = []
        maskp.append(mask_path)
        maskp.append(mask0_path)
        maskp.append(mask1_path)
        maskp.append(mask2_path)
        maskp.append(mask3_path)

        # 获取样本对应的背景图像路径

        bk_path = img[:-4] + '_bk' + '.jpg'
        bk0_path = pos_path[0][:-4] + '_bk' + '.jpg'
        bk1_path = pos_path[1][:-4] + '_bk' + '.jpg'
        bk2_path = pos_path[2][:-4] + '_bk' + '.jpg'
        bk3_path = pos_path[3][:-4] + '_bk' + '.jpg'
        bkp = []
        bkp.append(bk_path)
        bkp.append(bk0_path)
        bkp.append(bk1_path)
        bkp.append(bk2_path)
        bkp.append(bk3_path)
        bk = []
        bkk = self.loader(bk_path)
        bk0 = self.loader(bk0_path)
        bk1 = self.loader(bk1_path)
        bk2 = self.loader(bk2_path)
        bk3 = self.loader(bk3_path)
        bk.append(bkk)
        bk.append(bk0)
        bk.append(bk1)
        bk.append(bk2)
        bk.append(bk3)

        # 加载当前样本及其4个正样本图像
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        # 加载mask图像
        mask = []
        maskk = self.loader(mask_path)
        mask0 = self.loader(mask0_path)
        mask1 = self.loader(mask1_path)
        mask2 = self.loader(mask2_path)
        mask3 = self.loader(mask3_path)
        mask.append(maskk)
        mask.append(mask0)
        mask.append(mask1)
        mask.append(mask2)
        mask.append(mask3)

        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            # back =cv2.imread(maskp[i], cv2.IMREAD_COLOR)
            img_sub = []
            for j in range(self.domain_num - 1):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                          img_3channel[index_channel[j][2]]))
                # ahead =cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                # img = cv2.bitwise_and(ahead, back)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # idd =str(j)
                # res = os.path.join(bk[i][:-7] +'_'+idd+ '_cat.jpg')
                # cv2.imwrite(res, img_res)

                img_sub.append(img)
            for k in range(self.cam_num):
                if k + 1 != cam:
                    cam_path = wr[:-4] + '_fake_' + str(cam) + 'to' + str(k + 1) + '.jpg'
                    camaug = self.loader(cam_path)
                    img_sub.append(camaug)

            img_sub.append(mask[i])
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num + self.cam_num, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2 = self.data[index_two[0]], self.labels[index_two[0]]
        if 'mask' in img2:
            mask_path2 = img2
            img2 = img2[:-9] + '.jpg'
        else:
            mask_path2 = img2[:-4] + '_mask' + '.jpg'
        mask2 = []
        mask2.append(mask_path2)
        bk2 = []
        bk_path2 = img2[:-4] + '_bk' + '.jpg'
        bk2.append((bk_path2))
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            # back2 = cv2.imread(bk2[i], cv2.IMREAD_COLOR)
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num - 1):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                          img_3channel[index_channel[j][2]]))
                # ahead = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                # img = cv2.bitwise_and(ahead, back2)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            img_mask2 = default_loader(mask_path2)
            img_sub.append(img_mask2)
            label_sub.append(int(label))
            label_all2[i][6] = int(label)

            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img

    def __len__(self):
        return len(self.data)
