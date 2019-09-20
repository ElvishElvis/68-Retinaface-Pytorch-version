# from dataloader import TrainDataset, collater, Resizer, PadToSquare,Color,Rotate,RandomErasing,RandomFlip,ValDataset
# import torchvision.transforms as transforms
# import cv2
# import copy
# import torch.nn.functional as F
# import torch
# from PIL import Image
# import numpy as np
# import os
# import skimage



# dataset_train = ValDataset('./widerface/train/label.txt',transform=transforms.Compose([Resizer(640),PadToSquare()]))
# list__=dataset_train[99]
# img=np.array(list__['img'])
# print(img.shape)
# # img = skimage.io.imread("/versa/elvishelvis/RetinaFace_Pytorch/\
# # CelebA/Img/img_celeba.7z/img_celeba/101299.jpg")

# box=np.array(list__['annot'])[0]


# img=cv2.circle(img,(int(box[0]),int(box[1])),radius=1,color=(0,255,0),thickness=10)
# img=cv2.circle(img,(int(box[2]),int(box[3])),radius=1,color=(255,0,0),thickness=10)
# img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),thickness=2)

# for i in range(4,140,2):
#     try:
#         if(i>=100):
#             img=cv2.circle(img,(int(box[i]),int(box[i+1])),radius=1,color=(255,255,255),thickness=2)
#         else:
#             img=cv2.circle(img,(int(box[i]),int(box[i+1])),radius=1,color=(0,0,255),thickness=2)
#         # img=cv2.circle(img,(int(box[i+2]),int(box[i+3])),radius=1,color=(255,0,0),thickness=2)
#         # img=cv2.circle(img,(int(box[i+4]),int(box[i+5])),radius=1,color=(0,255,0),thickness=2)
#         # img=cv2.circle(img,(int(box[i+6]),int(box[i+7])),radius=1,color=(255,255,0),thickness=2)
#     except:
#         break
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('sdfas33df.jpg',img)


# import torch
# from torch import nn


# # torch.log  and math.log is e based
# class AdaptiveWingLoss(nn.Module):
#     def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
#         super(AdaptiveWingLoss, self).__init__()
#         self.omega = omega
#         self.theta = theta
#         self.epsilon = epsilon
#         self.alpha = alpha

#     def forward(self, pred, target):
#         '''
#         :param pred: BxNxHxH
#         :param target: BxNxHxH
#         :return:
#         '''

#         y = target
#         y_hat = pred
#         delta_y = (y - y_hat).abs()
#         delta_y1 = delta_y[delta_y < self.theta]
#         delta_y2 = delta_y[delta_y >= self.theta]
#         y1 = y[delta_y < self.theta]
#         y2 = y[delta_y >= self.theta]
#         loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
#         A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
#             torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
#         C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
#         loss2 = A * delta_y2 - C
#         return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

# if __name__ == "__main__":
#     loss_func = AdaptiveWingLoss()
#     y = torch.rand(3,136)
#     y_hat = torch.rand(3,136)
#     print(y_hat)
#     y_hat.requires_grad_(True)
#     loss = loss_func(y_hat, y)
#     loss.backward()
#     print(loss)


import torch
import math
import torch.nn as nn
class WingLoss(nn.Module):
    def __init__(self, omega=1, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        print(delta_y.shape)
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        print(delta_y2)
        sdf
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
aaa=WingLoss()
a=torch.rand(1,136)*3
b=torch.rand(1,136)
print(aaa(a,b))
