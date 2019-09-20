from dataloader import TrainDataset, collater, Resizer, PadToSquare,Color,Rotate,RandomErasing,RandomFlip,ValDataset,RandomCroper
import torchvision.transforms as transforms
import cv2
import copy
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import os
import skimage



dataset_train = TrainDataset('./widerface/train/label.txt',transform=transforms.Compose([Rotate(),RandomErasing(),RandomCroper(),Resizer(),PadToSquare(),RandomFlip(),Color()]))
list__=dataset_train[1]
img=np.array(list__['img'])
print(list__['weight'])

# img = np.array(sample['img'])
img=Image.fromarray(img.astype('uint8'))
img=transforms.ColorJitter(brightness=0.6,contrast=0.5,saturation=0.5,hue=0.5)(img)
img=np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print(img.shape)
# img = skimage.io.imread("/versa/elvishelvis/RetinaFace_Pytorch/\
# CelebA/Img/img_celeba.7z/img_celeba/101299.jpg")

box=np.array(list__['annot'])[0]


img=cv2.circle(img,(int(box[0]),int(box[1])),radius=1,color=(0,255,0),thickness=10)
img=cv2.circle(img,(int(box[2]),int(box[3])),radius=1,color=(255,0,0),thickness=10)
img=cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),thickness=2)

for i in range(4,200,8):
    # if(i>=100):
    #     img=cv2.circle(img,(int(box[i]),int(box[i+1])),radius=1,color=(0,255,255),thickness=2)
    # else:
    try:
        img=cv2.circle(img,(int(box[i]),int(box[i+1])),radius=1,color=(0,0,255),thickness=2)
        img=cv2.circle(img,(int(box[i+2]),int(box[i+3])),radius=1,color=(255,0,0),thickness=2)
        img=cv2.circle(img,(int(box[i+4]),int(box[i+5])),radius=1,color=(0,255,0),thickness=2)
        img=cv2.circle(img,(int(box[i+6]),int(box[i+7])),radius=1,color=(255,255,0),thickness=2)
    except:
        break
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('sdfas33df.jpg',img)
