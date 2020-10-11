# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:13:52 2020

@author: admin
"""
import torch
import torch.nn.functional as F
import numpy as np
import torch._utils

from torch.autograd import Variable
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

kc, kh, kw = 1, 3, 3  # kernel size
dc, dh, dw = 1, 1, 1  # stride
pad = (1,1,1,1)

def preprocess60x60(image):

    if image.shape[2] == 240:
        coor = torch.load('240x480.pt')
    elif image.shape[2] == 120:
        coor = torch.load('120x240.pt')
    elif image.shape[2] == 60:
        coor = torch.load('60x120.pt')
    elif image.shape[2] == 30:
        coor = torch.load('30x60.pt')
    
    paddedimage = F.pad(image, pad)
    patches = paddedimage.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous()

    coor_in = coor.unsqueeze(0).unsqueeze(0)
    #print(coor_in.size())
    patches[:, :, :, :,0,2,1] = image[:,:,coor_in[0, 0, :, :, 1,0],coor_in[0, 0, :,:,0,0]]
    patches[:, :, :, :,0,2,2] = image[:,:,coor_in[0, 0, :, :, 1,1],coor_in[0, 0, :,:,0,1]]
    patches[:, :, :, :,0,1,2] = image[:,:,coor_in[0, 0, :, :, 1,2],coor_in[0, 0, :,:,0,2]]
    patches[:, :, :, :,0,0,2] = image[:,:,coor_in[0, 0, :, :, 1,3],coor_in[0, 0, :,:,0,3]]
    patches[:, :, :, :,0,0,1] = image[:,:,coor_in[0, 0, :, :, 1,4],coor_in[0, 0, :,:,0,4]]
    patches[:, :, :, :,0,0,0] = image[:,:,coor_in[0, 0, :, :, 1,5],coor_in[0, 0, :,:,0,5]]
    patches[:, :, :, :,0,1,0] = image[:,:,coor_in[0, 0, :, :, 1,6],coor_in[0, 0, :,:,0,6]]
    patches[:, :, :, :,0,2,0] = image[:,:,coor_in[0, 0, :, :, 1,7],coor_in[0, 0, :,:,0,7]]
    
    # Reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(image.shape[0], output_c, output_h, output_w)
    
    return patches_orig

def preprocess(image,coor):

    tempimg = image
    image = F.pad(image, pad)
    image = image.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = image.size()
    image = image.contiguous()

    coor_in = coor.unsqueeze(0).unsqueeze(0)
    #print(patches.size())
    image[:, :, :, :,0,2,1] = tempimg[:,:,coor_in[0, 0, :, :, 1,0],coor_in[0, 0, :,:,0,0]]
    image[:, :, :, :,0,2,2] = tempimg[:,:,coor_in[0, 0, :, :, 1,1],coor_in[0, 0, :,:,0,1]]
    image[:, :, :, :,0,1,2] = tempimg[:,:,coor_in[0, 0, :, :, 1,2],coor_in[0, 0, :,:,0,2]]
    image[:, :, :, :,0,0,2] = tempimg[:,:,coor_in[0, 0, :, :, 1,3],coor_in[0, 0, :,:,0,3]]
    image[:, :, :, :,0,0,1] = tempimg[:,:,coor_in[0, 0, :, :, 1,4],coor_in[0, 0, :,:,0,4]]
    image[:, :, :, :,0,0,0] = tempimg[:,:,coor_in[0, 0, :, :, 1,5],coor_in[0, 0, :,:,0,5]]
    image[:, :, :, :,0,1,0] = tempimg[:,:,coor_in[0, 0, :, :, 1,6],coor_in[0, 0, :,:,0,6]]
    image[:, :, :, :,0,2,0] = tempimg[:,:,coor_in[0, 0, :, :, 1,7],coor_in[0, 0, :,:,0,7]]

    # Reshape back
    image = image.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    image = image.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    image = image.view(image.shape[0], output_c, output_h, output_w)
    
    return image.cuda()
#coor = torch.cuda.LongTensor(30, 30, 2, 8)
"""
if __name__ == '__main__':
    import time
    
    start = time.time()
    coor = torch.cuda.LongTensor(240, 480, 2, 8)#load('60x60.pt')
    img = Variable(torch.randn(32,21,240,480), requires_grad=True).cuda()#, requires_grad=True)
    img = preprocess60x60(img, coor)
    end = time.time()
    print('Elapsed time (process patches): {} s'.format(end - start))
    print(img.size())
    
    coor = torch.cuda.LongTensor(30, 30, 2, 8)#load('60x60.pt')
    img = torch.randn(1,3,30,30).cuda()#, requires_grad=True)
    from torch.autograd import gradcheck
    print(gradcheck(lambda x: preprocess(x).sum(), img.clone().detach().double().requires_grad_()))
    #img.backward()
    #print(img.grad)
"""