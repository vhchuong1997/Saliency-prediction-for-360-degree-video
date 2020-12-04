# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:13:52 2020

@author: admin
"""
import torch
import torch.nn.functional as F
import torch._utils

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
pad = (1, 1, 1, 1)

def preprocess(image, coor):

    tempimg = image
    image = F.pad(image, pad)
    image = image.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = image.size()
    image = image.contiguous()

    coor_in = coor.unsqueeze(0).unsqueeze(0)
    #print(patches.size())
    image[:, :, :, :, 0, 2, 1] = tempimg[:,:,coor_in[0, 0, :, :, 1, 0],coor_in[0, 0, :, :, 0, 0]]
    image[:, :, :, :, 0, 2, 2] = tempimg[:,:,coor_in[0, 0, :, :, 1, 1],coor_in[0, 0, :, :, 0, 1]]
    image[:, :, :, :, 0, 1, 2] = tempimg[:,:,coor_in[0, 0, :, :, 1, 2],coor_in[0, 0, :, :, 0, 2]]
    image[:, :, :, :, 0, 0, 2] = tempimg[:,:,coor_in[0, 0, :, :, 1, 3],coor_in[0, 0, :, :, 0, 3]]
    image[:, :, :, :, 0, 0, 1] = tempimg[:,:,coor_in[0, 0, :, :, 1, 4],coor_in[0, 0, :, :, 0, 4]]
    image[:, :, :, :, 0, 0, 0] = tempimg[:,:,coor_in[0, 0, :, :, 1, 5],coor_in[0, 0, :, :, 0, 5]]
    image[:, :, :, :, 0, 1, 0] = tempimg[:,:,coor_in[0, 0, :, :, 1, 6],coor_in[0, 0, :, :, 0, 6]]
    image[:, :, :, :, 0, 2, 0] = tempimg[:,:,coor_in[0, 0, :, :, 1, 7],coor_in[0, 0, :, :, 0, 7]]

    # Reshape back
    image = image.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    image = image.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    image = image.view(image.shape[0], output_c, output_h, output_w)

    return image.cuda()
