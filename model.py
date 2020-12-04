from torch import nn
import torch as th
from torch_support import preprocess60x60, preprocess

try:
    th._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = th._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    th._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
class Final1(nn.Module):
    def __init__(self):
        super(Final1, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 3, stride=3, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=3, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=3, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 256, 3, stride=3, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv5 = nn.Conv2d(256 + 256, 128, 3, stride=3, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv6 = nn.Conv2d(128 + 128, 64, 3, stride=3, padding=0)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(64 + 64, 1, 3, stride=3, padding=0)

    def forward(self, image, last, coor1, coor2, coor3, coor4):
        x = th.cat([image, last], dim=1)
        x = preprocess(x , coor1)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        r1 = self.relu1(c1)
        p1 = self.pool1(r1)

        c2 = self.conv2(preprocess(p1, coor2))
        c2 = self.bn2(c2)
        r2 = self.relu2(c2)
        p2 = self.pool2(r2)

        c3 = self.conv3(preprocess(p2, coor3))
        c3 = self.bn3(c3)
        r3 = self.relu3(c3)
        p3 = self.pool3(r3)

        c4 = self.conv4(preprocess(p3, coor4))
        c4 = self.bn4(c4)
        r4 = self.relu4(c4)

        r4 = self.up1(r4)
        c5 = self.conv5(preprocess(th.cat([r4, r3], dim=1),coor3))
        c5 = self.bn5(c5)
        r5 = self.relu5(c5)

        r5 = self.up2(r5)
        c6 = self.conv6(preprocess(th.cat([r5, r2], dim=1),coor2))
        c6 = self.bn6(c6)
        r6 = self.relu6(c6)

        r6 = self.up3(r6)
        c7 = self.conv7(preprocess(th.cat([r6, r1], dim=1),coor1))

        return c7
