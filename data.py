# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:45:01 2020

@author: admin
"""


import numpy as np
import torch as th
from torch import nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import os
import pickle
from scipy import signal
#from sconv.functional.sconv import spherical_conv
from tqdm import tqdm
import numbers
#import cv2
#from functools import lru_cache
from random import Random

class VRVideo(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        vid, fid = self.i2v[item]
        if int(fid) - self.frame_interval <= 0:
            last = self._get_salency_map(-1)
        else:
            last = self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])

        target = self._get_salency_map(item)

        if self.train:
            return img, last, target
        else:
            return img, self.data[item], last, target

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                #target_map = transforms.ToPILImage(mode=None)(target_map)
                #upplayer = Interpolate(size=(), mode='bilinear')
                target_map = nn.functional.interpolate(target_map.unsqueeze(0), size = (self.frame_h, self.frame_w ),  mode='bilinear') #.resize((self.frame_w, self.frame_h))
                #target_map = transforms.ToTensor()(target_map)
                target_map = target_map.squeeze(0)
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return target_map #th.from_numpy(np.load(cfile)).float()
        target_map = th.zeros((1, self.frame_h, self.frame_w))
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map#.data / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self