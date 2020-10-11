# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:22:05 2020

@author: admin
"""
import torch as th
import os
import numpy as np
from data import VRVideo
import torchvision.transforms as tf
from torchvision.utils import save_image
from torch.utils import data as tdata
from tqdm import tqdm
import visdom
from model import Final1
import cv2

def test(
        data_dir='./360_Saliency_dataset_2018ECCV',
        bs=1,
        plot_server='http://localhost',
        plot_port=8097,
        exp_name='test1'
): 
    outputdir = './output123'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)
    #textwindow = viz.text("Hello Pytorch3")
    transform = tf.Compose([
        tf.Resize((240, 480)),#128, 256
        tf.ToTensor()
    ])
    coor1 = th.load('./support/240x480.pt')#.cuda()
    coor2 = th.load('./support/120x240.pt')#.cuda()
    coor3 = th.load('./support/60x120.pt')#.cuda()
    coor4 = th.load('./support/30x60.pt')#.cuda()
    tar = th.load('ckpt-final - 20epoch -latest.pth.tar')
    #Loading model
    model = Final1().cuda()
    #model = nn.DataParallel(model).cuda()
    
    #state_dict = tar['state_dict']
    model.load_state_dict(tar['state_dict'])
    model.eval()

    #Prepare Dataset 128, 256
    dataset = VRVideo(data_dir, 240, 480, 80, frame_interval=5, train=False, cache_gt=True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    print(len(dataset))
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    with th.no_grad():
    #Put data through the model
        for i, (img_batch, a, last_batch, target_batch) in tqdm(enumerate(loader), desc='batch', total = len(loader)):
            
            if i%10==0:
                img_var = img_batch.cuda()
                last_var = (last_batch * 10).cuda()
                
                out = model(img_var, last_var,coor1, coor2, coor3, coor4)
                #viz.images(target_batch.cpu().numpy() * 10)
                #print(out)
                img = img_batch[0]
                img1 = out[0]
                img2 = target_batch[0]
                save_image(img.cpu(), outputdir + '/img_'+ str(i) +'.png')
                save_image((img1*30).cpu(), outputdir + '/SalMap_'+ str(i) +'.png')
                save_image((img2*300).cpu(), outputdir + '/gt_'+ str(i) +'.png')
                #th.save(img1.data, 'C:/Users/admin/Desktop/Chuong/Saliency-detection-in-360-video-master/output1_1/SalMap_'+ str(i) +'.bin')
                #th.save(img2*10, 'C:/Users/admin/Desktop/Chuong/Saliency-detection-in-360-video-master/output1_1/gt_'+str(i)+'.bin')
                
                gt = cv2.imread(outputdir + '/gt_'+ str(i) +'.png',0)
                gt = th.tensor(gt)
                Salmap = cv2.imread(outputdir + '/SalMap_'+ str(i) +'.png',0)
                Salmap = th.tensor(Salmap)
                viz.images(img_batch.cpu().numpy(), win='Frame')
                viz.images(Salmap.cpu().numpy(), win='Saliency map')
                viz.images(gt.cpu().numpy(), win='Ground truth')
            if i == 1000:
                break

if __name__ == '__main__':
    test()