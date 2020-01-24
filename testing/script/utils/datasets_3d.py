import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

def pre_proc_for_detect(img_path):
    """
    pre-processing for detection or ImageFolder(Dataset) class.
    Grayscale images are converted to rgb ones.

    :param img_path: image file path
    :type img_path: string
    :return: np.array(Image.open(img_path))
    :rtype: np.array
    """
    # Extract image
    imgpil = Image.open(img_path)
    img = np.array(imgpil)
    if len(img.shape)==2: #grayscale images such CT images
        # convert from grayscale to RGB image
        ## Note that imgpil.convert("RGB") cannot behavior as expected.
        ## It makes the image black.
        rgbimgpil = Image.new("RGB",imgpil.size)
        rgbimgpil.paste(imgpil)
        img = np.array(rgbimgpil) #img.shape must be 3 length for the following proc.
        return img
    elif len(img.shape)==3: #rgb color images
        return img
    else: #broken image cases
        print("#####################")
        print("#### FATAL ERROR ####")
        print("#{} is broken".format(img_path))
        print("#####################")
        raise NotImplementedError

def pre_proc_for_train(img_path, index, self):
    """
    pre-processing for training or ListDataset(Dataset) class.
    Grayscale images are converted to rgb ones.
    Broken images are skipped until a not broken image appears.

    :param img_path: image file path
    :type img_path: string
    :param index: index of the image gile in the batch
    :type index: int
    :param self: self
    :type self: class
    :return: np.array(Image.open(img_path))
    :rtype: np.array
    """
    imgpil = Image.open(img_path)
    img = np.array(imgpil)
    if len(img.shape)==2: #grayscale images such CT images
        # convert from grayscale to RGB image
        ## Note that imgpil.convert("RGB") cannot behavior as expected.
        ## It makes the image black.
        rgbimgpil = Image.new("RGB", imgpil.size)
        rgbimgpil.paste(imgpil)
        img = np.array(rgbimgpil) #img.shape must be 3 length for the following proc.
        return img
    elif len(img.shape)==3: #rgb color images
        return img
    else: #broken image cases
        # the batch can include broken images thus skip them
        # TODO: test for this block
        while len(img.shape)!=2 or len(img.shape)!=3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            return pre_proc_for_train(img_path,index,self)

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=512):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = pre_proc_for_detect(img_path)
        h,w,_ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        #input_img = (input_img - 127.5)/255.0
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=512):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Handles images with less than three channels
        # if two channels, then convert to rgb color type image
        """
        # original handling code
        imgpil = Image.open(img_path)
        img = np.array(imgpil)
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))
        """
        # Handled result
        img = pre_proc_for_train(img_path,index,self)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        #input_img = (input_img - 127.5)/255.0
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

## ここから下が追記箇所。

def get_nbhd_slice(img_path, i):
    """
    近傍スライスpathを獲得する

    :param img_path: キースライスの画像へのパス
    :type img_path: string
    :param i: キースライスからのスライスインデックス。負なら前の、正なら後ろの。
    :type i: int
    :return: img_pathからi番目のスライスへのパス
    :rtype: string
    """
    key_slice_index = int(((img_path.rsplit(os.sep, 1)[-1]).rsplit("_", 1)[-1]).split(".")[-2])
    this_slice_index = "-1"
    #define error message
    messe = "Error: slice_volume_size is out of range in this dataset.\n"
    vars = "key_slice_index=={}, i=={}, img_path={}".format(key_slice_index,i,img_path)
    #slice index and path control
    if key_slice_index + i <= 0:
        try:
            raise ValueError(messe+vars)
        except ValueError as e:
             print(e)
    elif key_slice_index + i <= 9:
        this_slice_index = "00" + str(key_slice_index + i)
    elif key_slice_index + i <= 99:
        this_slice_index = "0" + str(key_slice_index + i)
    else:
        this_slice_index = str(key_slice_index + i)
    this_path = os.path.join(img_path.rsplit(os.sep,1)[0],
                           str(((img_path.rsplit(os.sep,1)[-1]).rsplit("_",1)[0])) + "_" + this_slice_index + ".png")
    return this_path

class ImageTensorFolder(Dataset):
    """
    2019/01/24追記
    立体情報考慮したときの検出に使うDataset
    2019/02/05追記
    ListTensorDatasetのラベルなし版でいい
    """
    def __init__(self, list_path, img_size=512, slice_volume_size=3):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        #self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.slice_size = slice_volume_size
        assert(int(self.slice_size)%2 == 1)

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Handles images with less than three channels
        # if two channels, then convert to rgb color type image
        """
        # original handling code
        imgpil = Image.open(img_path)
        img = np.array(imgpil)
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))
        """
        # Handled result
        # TODO:previous and next images must be converted to RGB and loaded.
        # previous img_path is defined by
        # prev_img_path = os.path.join(img_path.split(os.sep)[0:-1],
        #                   str(int((img_path.split(os.sep)[-1]).split("_").split("."))-1)+".png")
        # next is s/-1/+1/ as above tail.
        # if slice volume size = 3, then the above definition may be run correctly.
        # when slice volume size >3 (such 5,7,9; channels 15,21,27), we have TODO consider generalization.
        # one generalization procedure is s/1/i/ when slice volume size = 2n+1 (n>=1)
        # for i in [1 ~ n].
        #
        # Handle the slice volume
        n = self.slice_size // 2 if self.slice_size>1 else 0
        img = pre_proc_for_train(img_path,index,self)
        h, w, _ = img.shape
        if n: #{
            prev_img_paths = [get_nbhd_slice(img_path,-i) for i in range(1,n+1)]
            #print(prev_img_paths)
            prev_imgs = [pre_proc_for_train(ip,index,self) for ip in prev_img_paths]
            next_img_paths = [get_nbhd_slice(img_path,i) for i in range(1,n+1)]
            next_imgs = [pre_proc_for_train(ip,index,self) for ip in next_img_paths]
            volume_imgs = [im for im in prev_imgs]
            volume_imgs.append(img)
            for ni in next_imgs: #{
                volume_imgs.append(ni)
            #}
            #print(volume_imgs)
            del prev_imgs
            del prev_img_paths
            del next_imgs
            del next_img_paths
            input_imgs = []
            for im in volume_imgs: #{
                dim_diff = np.abs(h - w)
                # Upper (left) and lower (right) padding
                pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
                # Determine padding
                pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
                # Normalize and add padding
                input_img = np.pad(im, pad, 'constant', constant_values=127.5) / 255.
                padded_h, padded_w, _ = input_img.shape
                # Resize
                input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
                # Channels-first
                input_img = np.transpose(input_img, (2, 0, 1))
                #input_img = (input_img - 127.5)/255.0
                # As pytorch tensor
                input_img = torch.from_numpy(input_img).float()
                input_imgs.append(input_img)
            #}
            input_img = torch.cat(input_imgs,0) #[slice_size*C,H,W] tensor
        #}
        else: #{
            h, w, _ = img.shape
            dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            # Add padding
            input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
            padded_h, padded_w, _ = input_img.shape
            # Resize and normalize
            input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
            # Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))
            #input_img = (input_img - 127.5)/255.0
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.img_files)


class ListTensorDataset(Dataset):
    """
    2019-01-22以降追記
    立体情報を利用するためのDataset
    """
    def __init__(self, list_path, img_size=512, slice_volume_size=3):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.slice_size = slice_volume_size
        assert(int(self.slice_size)%2 == 1)

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Handles images with less than three channels
        # if two channels, then convert to rgb color type image
        """
        # original handling code
        imgpil = Image.open(img_path)
        img = np.array(imgpil)
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))
        """
        # Handled result
        # TODO:previous and next images must be converted to RGB and loaded.
        # previous img_path is defined by
        # prev_img_path = os.path.join(img_path.split(os.sep)[0:-1],
        #                   str(int((img_path.split(os.sep)[-1]).split("_").split("."))-1)+".png")
        # next is s/-1/+1/ as above tail.
        # if slice volume size = 3, then the above definition may be run correctly.
        # when slice volume size >3 (such 5,7,9; channels 15,21,27), we have TODO consider generalization.
        # one generalization procedure is s/1/i/ when slice volume size = 2n+1 (n>=1)
        # for i in [1 ~ n].
        #
        # Handle the slice volume
        n = self.slice_size // 2 if self.slice_size>1 else 0
        img = pre_proc_for_train(img_path, index, self) # imageを3次元に整える
        h, w, _ = img.shape
        if n:
            prev_img_paths = [get_nbhd_slice(img_path, -i) for i in range(1, n+1)]
            #print(prev_img_paths)
            prev_imgs = [pre_proc_for_train(ip,index,self) for ip in prev_img_paths]
            next_img_paths = [get_nbhd_slice(img_path,i) for i in range(1,n+1)]
            next_imgs = [pre_proc_for_train(ip,index,self) for ip in next_img_paths]
            volume_imgs = [im for im in prev_imgs]
            volume_imgs.append(img)
            for ni in next_imgs:
                volume_imgs.append(ni)

            #print(volume_imgs)
            del prev_imgs
            del prev_img_paths
            del next_imgs
            del next_img_paths
            input_imgs = []
            for im in volume_imgs: #{
                dim_diff = np.abs(h - w)
                # Upper (left) and lower (right) padding
                pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
                # Determine padding
                pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
                # Normalize and add padding
                input_img = np.pad(im, pad, 'constant', constant_values=127.5) / 255.
                padded_h, padded_w, _ = input_img.shape
                # Resize
                input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
                # Channels-first
                input_img = np.transpose(input_img, (2, 0, 1))
                #input_img = (input_img - 127.5)/255.0
                # As pytorch tensor
                input_img = torch.from_numpy(input_img).float()
                input_imgs.append(input_img)
            #}
            input_img = torch.cat(input_imgs,0) #[slice_size*C,H,W] tensor
        #}
        else: #{
            h, w, _ = img.shape
            dim_diff = np.abs(h - w)
            # Upper (left) and lower (right) padding
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            # Determine padding
            pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
            # Add padding
            input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
            padded_h, padded_w, _ = input_img.shape
            # Resize and normalize
            input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
            # Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))
            #input_img = (input_img - 127.5)/255.0
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
