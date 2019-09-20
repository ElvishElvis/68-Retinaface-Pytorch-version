import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import eval_widerface
import torchvision_model
import model
import os
import skimage
from dataloader import  ValDataset, Resizer, PadToSquare,ValDataset_CeleB, TrainDataset
from torchvision import datasets, models, transforms
def pad_to_square(img, pad_value):
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--image_path', type=str, default='WechatIMG10.jpeg', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model',default="/versa/elvishelvis/RetinaYang/out/68_full_model_epoch_10.pt")
    parser.add_argument('--save_path', type=str, default='./out', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    args = parser.parse_args()

    return args

def main(nummmmmm):
    args = get_args()

	# Create the model
    # if args.depth == 18:
    #     RetinaFace = model.resnet18(num_classes=2, pretrained=True)
    # elif args.depth == 34:
    #     RetinaFace = model.resnet34(num_classes=2, pretrained=True)
    # elif args.depth == 50:
    #     RetinaFace = model.resnet50(num_classes=2, pretrained=True)
    # elif args.depth == 101:
    #     RetinaFace = model.resnet101(num_classes=2, pretrained=True)
    # elif args.depth == 152:
    #     RetinaFace = model.resnet152(num_classes=2, pretrained=True)
    # else:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Create torchvision model
    
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load('/versa/elvishelvis/96RetinaYang/out/model_epoch_121.pt')
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)
    RetinaFace.to(device)
    
    import time
    start=time.time()
    dataset_val = ValDataset_CeleB('./widerface/train/label.txt',transform=transforms.Compose([Resizer(1024)]))
    # dataset_val = ValDataset('./widerface/train/label.txt')
    for qq in range(600,700):
        img=dataset_val[qq]['img']
        # boxssss=dataset_val[qq]['annot'][0]
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img=skimage.io.imread("/versa/elvishelvis/RetinaFace_Pytorch/CelebA/Img/img_celeba.7z/img_celeba/118{}.jpg".format(str(qq)))
        img = img.permute(2,0,1)
        resized_img = img.float()
        input_img = resized_img.unsqueeze(0).to(device)
        
        picked_boxes, picked_landmarks = eval_widerface.get_detections(input_img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)
        # print(picked_boxes)
        np_img = resized_img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)

        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box,landmark in zip(boxes,picked_landmarks[j]):
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)
                    # cv2.rectangle(img,(boxssss[0],boxssss[1]),(boxssss[2],boxssss[3]),(0,255,0),thickness=2)
                    # for k in range(4,200,2):
                    #     cv2.circle(img,(boxssss[k],boxssss[k+1]),radius=1,color=(0,255,0),thickness=2)
                    for i in range(0,196,2):
                        cv2.circle(img,(landmark[i],landmark[i+1]),radius=1,color=(0,0,255),thickness=2)
                    
        image_name = args.image_path.split('/')[-1]
        save_path = os.path.join(args.save_path,image_name)
        cv2.imwrite('./RetinaFace-Pytorch{}.jpg'.format(qq),cv2.resize(img,(640,640)))
    print(time.time()-start)
if __name__=='__main__':
    main(10)


