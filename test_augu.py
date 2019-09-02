from dataloader import TrainDataset, collater, RandomFlip, Resizer,RandomErasing, Rotate, Color
import torchvision.transforms as transforms
import cv2
import copy
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import os
import skimage


dataset_train = TrainDataset(transform=transforms.Compose([Resizer(),Color()]))
list__=dataset_train[101310]
img=np.array(list__['img'])
# img = skimage.io.imread("/versa/elvishelvis/RetinaFace_Pytorch/\
# CelebA/Img/img_celeba.7z/img_celeba/101299.jpg")


box=np.array(list__['annot'])[0]

cv2.circle(img,(int(box[0]),int(box[1])),radius=1,color=(0,255,0),thickness=10)
cv2.circle(img,(int(box[2]),int(box[3])),radius=1,color=(0,255,255),thickness=10)
cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),thickness=2)
cv2.circle(img,(int(box[4]),int(box[5])),radius=1,color=(0,0,255),thickness=2)
cv2.circle(img,(int(box[6]),int(box[7])),radius=1,color=(0,255,0),thickness=2)
cv2.circle(img,(int(box[8]),int(box[9])),radius=1,color=(255,0,0),thickness=2)
cv2.circle(img,(int(box[10]),int(box[11])),radius=1,color=(0,255,255),thickness=2)
cv2.circle(img,(int(box[12]),int(box[13])),radius=1,color=(255,255,0),thickness=2)


cv2.imwrite('sdfas33df.jpg',img)
