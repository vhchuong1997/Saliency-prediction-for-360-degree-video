# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:46:43 2020

@author: admin
"""
from torch import nn
import numpy as np
import torch as th
from data2 import VRVideo
import torchvision.transforms as tf
from torch.utils import data as tdata
from torch.optim import SGD
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
#from fire import Fire
from tqdm import trange, tqdm
import visdom
import time

from spherical_unet2 import Final1
from sconv.module import SphericalConv, SphereMSE


def train(
        data_dir='C:/Users/admin/Desktop/Chuong/Saliency-detection-in-360-video-master/DATASET/360_Saliency_dataset_2018ECCV',
        bs=2, #28
        lr=1e-4,
        epochs=20,
        clear_cache=False,
        plot_server='http://localhost',
        plot_port=8097,
        save_interval=5,
        resume=True,
        start_epoch=0,
        exp_name='final',
        test_mode=False
):
    
    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

    transform = tf.Compose([
        tf.Resize((240, 480)),
        tf.ToTensor()
    ])
    dataset = VRVideo(data_dir, 240, 480, 40, frame_interval=5, cache_gt=True, transform=transform, train = True, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    if clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    model = Final1().cuda()
    optimizer = SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #pmodel = nn.DataParallel(model).cuda()
    criterion = SphereMSE(240, 480).float().cuda()# nn.BCELoss()
    if resume:
        ckpt = th.load('ckpt-' + exp_name + '-latest.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']+1
    coor1 = th.load('240x480.pt')
    coor2 = th.load('120x240.pt')
    coor3 = th.load('60x120.pt')
    coor4 = th.load('30x60.pt')
    log_file = open(exp_name +'.out', 'w+')
    cudnn.benchmark = True
    for epoch in trange(start_epoch, epochs, desc='epoch'):
        tic = time.time()
        for i, (img_batch, last_batch, target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
            model.train()
            img_var = img_batch.cuda()
            last_var = (last_batch * 10).cuda()
            t_var = (target_batch * 10).cuda()
            data_time = time.time() - tic
            tic = time.time()

            out = model(img_var, last_var, coor1, coor2, coor3, coor4)
            loss = criterion(out, t_var)
            fwd_time = time.time() - tic
            tic = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bkw_time = time.time() - tic

            msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
                epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss
            )
            viz.images(img_batch.cpu().numpy(), win='img')
            viz.images(target_batch.cpu().numpy() * 50, win='gt')
            viz.images(out.data.cpu().numpy() * 5, win='out')
            viz.text(msg, win='log')
            print(msg, file=log_file, flush=True)
            #print(msg, flush=True)

            tic = time.time()

            if (i + 1) % save_interval == 0:
                state_dict = model.state_dict()
                ckpt = dict(epoch=epoch, iter=i, state_dict=state_dict)
                #if epoch == 5:
                #    th.save(ckpt, 'ckpt-' + exp_name + ' - 6epoch -latest.pth.tar')
                #elif epoch == 6:
                #    th.save(ckpt, 'ckpt-' + exp_name + ' - 7epoch -latest.pth.tar')
                #elif epoch == 7:
                th.save(ckpt, 'ckpt-' + exp_name + ' - ' + str(epoch+1) +'epoch -latest.pth.tar')
        th.save(ckpt, 'ckpt-' + exp_name + '-latest.pth.tar')
                


if __name__ == '__main__':
    train()
