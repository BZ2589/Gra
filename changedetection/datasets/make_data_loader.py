import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img

def get_binary_map(img):
    gray_image = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    threshold = 127
    binary_image = 1-(gray_image > threshold).astype(np.uint8)
    return binary_image
def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_name,dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        # print(dataset_path)
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        # print(label.shape)
        if aug:
            if self.dataset_name == "DSIFN-CD":
                label = Image.fromarray(label)
                label = label.resize((pre_img.shape[0], pre_img.shape[1]))
                label = np.array(label)
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
    
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        if self.dataset_name=='LEVIR-CD' or self.dataset_name == 'LEVIR-CD+':
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path,'B', self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        if self.dataset_name=='SYSU':
            pre_prefix = 'time1' if os.path.exists(os.path.join(self.dataset_path, 'time1')) else 'A'
            post_prefix = 'time2' if os.path.exists(os.path.join(self.dataset_path, 'time2')) else 'B'
            pre_path = os.path.join(self.dataset_path, pre_prefix, self.data_list[index])
            post_path = os.path.join(self.dataset_path, post_prefix, self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        if self.dataset_name == 'DSIFN-CD':
            filename = self.data_list[index]
            label_stem = os.path.splitext(filename)[0]
            pre_dirs = ['im1', 't1', 't11', 'A']
            post_dirs = ['im2', 't2', 't22', 'B']
            pre_path = None
            for d in pre_dirs:
                candidate = os.path.join(self.dataset_path, d, filename)
                if os.path.exists(candidate):
                    pre_path = candidate
                    break
            if pre_path is None:
                raise FileNotFoundError(f'Cannot find DSIFN-CD pre image for {filename} under {self.dataset_path}')
            post_path = None
            for d in post_dirs:
                candidate = os.path.join(self.dataset_path, d, filename)
                if os.path.exists(candidate):
                    post_path = candidate
                    break
            if post_path is None:
                raise FileNotFoundError(f'Cannot find DSIFN-CD post image for {filename} under {self.dataset_path}')
            label_path = None
            possible_label_dirs = ['mask', 'mask_png', 'mask_128/m', 'mask_128', 'mask_256/m', 'mask_256', 'mask_64/m', 'mask_64', 'mask_32/m', 'mask_32', 'label']
            possible_label_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            for label_dir in possible_label_dirs:
                for ext in possible_label_exts:
                    candidate = os.path.join(self.dataset_path, label_dir, label_stem + ext)
                    if os.path.exists(candidate):
                        label_path = candidate
                        break
                if label_path is not None:
                    break
            if label_path is None:
                raise FileNotFoundError(f'Cannot find DSIFN-CD label for {filename} under {self.dataset_path}')
        if self.dataset_name=='WHU-CD':
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path,'B', self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label / 255

        if 'train'  in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, data_name,dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.data_name = data_name
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        if self.data_name =='SECOND':
            if 'train' in self.data_pro_type:
                pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
                post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
                T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
                T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
                cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])
            else:
                pre_path = os.path.join(self.dataset_path, 'T1', self.data_list[index])
                post_path = os.path.join(self.dataset_path, 'T2', self.data_list[index])
                T1_label_path = os.path.join(self.dataset_path, 'GT_T1', self.data_list[index])
                T2_label_path = os.path.join(self.dataset_path, 'GT_T2', self.data_list[index])
                cd_label_path = os.path.join(self.dataset_path, 'GT_CD', self.data_list[index])

        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        t1_label = self.loader(T1_label_path)
        t2_label = self.loader(T2_label_path)
        cd_label = self.loader(cd_label_path)
        cd_label = cd_label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(True, pre_img, post_img, cd_label, t1_label, t2_label)
        else:
            pre_img, post_img, cd_label, t1_label, t2_label = self.__transforms(False, pre_img, post_img, cd_label, t1_label, t2_label)
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, cd_label, t1_label, t2_label, data_idx

    def __len__(self):
        return len(self.data_list)

def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU-CD' in args.dataset or 'LEVIR-CD' in args.dataset or 'DSIFN-CD' in args.dataset or 'SYSU' in args.dataset:
        dataset = ChangeDetectionDatset(args.dataset,args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=8, pin_memory=True,
                                 drop_last=False)
        return data_loader
    if 'SECOND' in args.dataset:
        dataset = SemanticChangeDetectionDatset(args.dataset,args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=8, pin_memory=True,
                                 drop_last=False)
        return data_loader
    else:
        raise NotImplementedError