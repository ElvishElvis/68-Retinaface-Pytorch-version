# from dataloader import TrainDataset, collater, RandomFlip, Resizer,RandomErasing, Rotate, Color
# import torchvision.transforms as transforms
# import cv2
# import copy
# import torch.nn.functional as F
# import torch
# from PIL import Image
# import numpy as np
# import os 
# from scipy import misc
# # kkk=3
# # image=cv2.imread("/versa/elvisshi/landmarks/new_dataset/{}.jpg".format(kkk))
# # anno=torch.load("/versa/elvisshi/landmarks/new_dataset/{}.pth".format(kkk))
# # anno=np.array(anno)
# # for num,an in enumerate( anno):
# #     if(num==0 or num==16):
# #         cv2.circle(image,(int(an[0]),int(an[1])),radius=1,color=(255,0,0),thickness=20)
# #     elif(num==8 or num==30):
# #         cv2.circle(image,(int(an[0]),int(an[1])),radius=1,color=(255,0,255),thickness=20)
# #     else:
# #         cv2.circle(image,(int(an[0]),int(an[1])),radius=1,color=(0,255,0),thickness=5)


# # cv2.imwrite('sdfas33df.jpg',image)
# # dataset_train = TrainDataset()
# dataset_train = TrainDataset(transform=transforms.Compose([Resizer(),Rotate(),Color()]))
# list__=dataset_train[323]
# img=np.array(list__['img'])

# box=np.array(list__['annot'])[0]
# cv2.circle(img,(int(box[0]),int(box[1])),radius=1,color=(0,255,0),thickness=10)
# cv2.circle(img,(int(box[2]),int(box[3])),radius=1,color=(0,255,255),thickness=10)
# cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),thickness=2)
# for i in range(4,140,2):
#     a=i
#     b=i+1
#     if(i%4==0):
#         cv2.circle(img,(int(box[a]),int(box[b])),radius=1,color=(255,0,0),thickness=2)
#     else:
#         cv2.circle(img,(int(box[a]),int(box[b])),radius=1,color=(0,255,0),thickness=2)


# misc.imsave('sdfas33df.jpg',img)


# # import torch
# # import cv2
# # a=cv2.imread("/versa/elvisshi/landmarks/new_dataset/{}.jpg".format(5550))
# # print(a.shape)
# # a = torch.randn(4)
# # b=torch.randn(320,320)
# # c=torch.randn(320,320)

# # print(torch.stack(([b,c])).t().shape)

import os
path="/versa/elvishelvis/landmarks56/data55/"
for i in range(7201,22988):
    if(os.path.isfile(path+str(i)+".jpg") and os.path.isfile(path+str(i)+".pth")):
        pass
    else:
        print("fuck")
        print(i)


