import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import torch.nn.functional as F
from skimage.util import crop
import skimage.transform
from PIL import Image
import skimage.color
import torch.nn as nn
import numpy as np
import skimage.io
import skimage
import random
import torch
import math
import os
from collections import OrderedDict
from scipy import misc
import cv2

class TrainDataset(Dataset):
    def __init__(self,txt_path=None,transform=None,flip=False):
        self.words = []
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_size = 640
        path="/versa/elvishelvis/96RetinaYang/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
        file_=OrderedDict()
        f = open(path,'r')
        lines = f.readlines()
        for num ,line in enumerate( lines):
            annotation = np.zeros((1,200))
            result=line.split(' ')
            annotation[0,0] = result[196]  
            annotation[0,1] = result[197]    
            annotation[0,2] = result[198] 
            annotation[0,3] = result[199]  
            for i in range(4,200):
                annotation[0,i] = result[i-4]
            weight=10*int(result[200])+10*int(result[201])+2*int(result[202])+2*int(result[203])+5*int(result[204])+2*int(result[205])
            file_[num]=(result[-1][:-1],annotation,weight)
        self.file=file_
        # torch.save(file_, 'anno.pth')
    def __len__(self):
        # return len(self.name_list)   
        # return 10
        return len(self.file)
        # return 200
        # return 10

    def __getitem__(self,index):
        cur=self.file[index]
        annotations = np.zeros((0, 200))
        annotations = np.append(annotations,cur[1],axis=0)
        img=skimage.io.imread("/versa/elvishelvis/96RetinaYang/WFLW/WFLW_images/"+cur[0])
        num=cur[-1]
        if(num==0):
            num=1
        sample = {'img':torch.tensor(img), 'annot':torch.tensor(annotations),'weight':num}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    

def collater(data):
    batch_size = len(data)

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    weight = np.mean([s['weight'] for s in data])

    # batch images
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    assert height==width ,'Input width must eqs height'

    input_size = width
    batched_imgs = torch.zeros(batch_size, height, width, 3)

    for i in range(batch_size):
        img = imgs[i]
        batched_imgs[i,:] = img

    # batch annotations
    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:
        if annots[0].shape[1] > 4:
            annot_padded = torch.ones((len(annots), max_num_annots, 200)) * -1
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), max_num_annots, 4)) * -1
            #print('annot~~~~~~~~~~~~~~~~~~,',annots)
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        if annots[0].shape[1] > 4:
            annot_padded = torch.ones((len(annots), 1, 200)) * -1
        else:
            annot_padded = torch.ones((len(annots), 1, 4)) * -1

    batched_imgs = batched_imgs.permute(0, 3, 1, 2)

    return {'img': batched_imgs, 'annot': annot_padded,'weight':weight}


class RandomFlip(object):
    def __call__(self, sample, input_size=640, flip_x=0.5):
        aaa=np.random.rand()
        if aaa < flip_x:
            image, annots = sample['img'], sample['annot']
            c,w,h=image.shape
            # flip image
            image = torch.flip(image,[1])

            image = image.numpy()
            annots = annots.numpy()
            # relocate bboxes
            for i in range(0,200):
                if i%2==0:
                    annots[0, i] = w - annots[0, i]
            annots[0, 0],annots[0, 2]=annots[0, 2],annots[0, 0]
            for k in range(4,36):
                if(k%2==0):
                    annots[0, k],annots[0, (72-k)]=annots[0, (72-k)],annots[0, k]
                else:
                    annots[0, k],annots[0, (74-k)]=annots[0, (74-k)],annots[0, k]
            for b in range(70,80):
                if(b%2==0):
                    annots[0, b],annots[0, (166-b)]=annots[0, (166-b)],annots[0, b]
                else:
                    annots[0, b],annots[0, (168-b)]=annots[0, (168-b)],annots[0, b]
            for a in range(80,88):
                if(a%2==0):
                    annots[0, a],annots[0, (184-a)]=annots[0, (184-a)],annots[0, a]
                else:
                    annots[0, a],annots[0, (186-a)]=annots[0, (186-a)],annots[0, a]
            
            for k in range(124,134):
                if(k%2==0):
                    annots[0, k],annots[0, (272-k)]=annots[0, (272-k)],annots[0, k]
                else:
                    annots[0, k],annots[0, (274-k)]=annots[0, (274-k)],annots[0, k]
            for b in range(134,140):
                if(b%2==0):
                    annots[0, b],annots[0, (288-b)]=annots[0, (288-b)],annots[0, b]
                else:
                    annots[0, b],annots[0, (290-b)]=annots[0, (290-b)],annots[0, b]

            annots[0, 196],annots[0, 198]=annots[0, 198],annots[0, 196]
            annots[0, 197],annots[0, 199]=annots[0, 199],annots[0, 197]

            annots[0, 114],annots[0, 122]=annots[0, 122],annots[0, 114]
            annots[0, 115],annots[0, 123]=annots[0, 123],annots[0, 115]
            annots[0, 116],annots[0, 120]=annots[0, 120],annots[0, 116]
            annots[0, 117],annots[0, 121]=annots[0, 121],annots[0, 117]

            annots[0, 156],annots[0, 168]=annots[0, 168],annots[0, 156]
            annots[0, 157],annots[0, 169]=annots[0, 169],annots[0, 157]
            annots[0, 158],annots[0, 166]=annots[0, 166],annots[0, 158]
            annots[0, 159],annots[0, 167]=annots[0, 167],annots[0, 159]
            annots[0, 160],annots[0, 164]=annots[0, 164],annots[0, 160]
            annots[0, 161],annots[0, 165]=annots[0, 165],annots[0, 161]
            
            annots[0, 170],annots[0, 178]=annots[0, 178],annots[0, 170]
            annots[0, 171],annots[0, 179]=annots[0, 179],annots[0, 171]
            annots[0, 172],annots[0, 176]=annots[0, 176],annots[0, 172]
            annots[0, 173],annots[0, 177]=annots[0, 177],annots[0, 173]

            annots[0, 180],annots[0, 188]=annots[0, 188],annots[0, 180]
            annots[0, 181],annots[0, 189]=annots[0, 189],annots[0, 181]
            annots[0, 182],annots[0, 186]=annots[0, 186],annots[0, 182]
            annots[0, 183],annots[0, 187]=annots[0, 187],annots[0, 183]

            annots[0, 190],annots[0, 194]=annots[0, 194],annots[0, 190]
            annots[0, 191],annots[0, 195]=annots[0, 195],annots[0, 191]


        

            image = torch.from_numpy(image)
            annots = torch.from_numpy(annots)

            sample = {'img': image, 'annot': annots,'weight':sample['weight']}

        return sample


class Rotate(object):
    def __init__(self,angle=[-30,30],p=0.4):
        self.angle=angle
        self.p=p
    def __call__(self,sample):
        if(np.random.rand()<self.p):
            img=np.array(sample['img'])
            h,w,a=img.shape
            annots=sample['annot']
            weight=sample['weight']
            def rotate(img, angle, resample=False, expand=False, center=None):
                rows,cols = img.shape[0:2]
                if center is None:
                    center = (cols/2, rows/2)
                M = cv2.getRotationMatrix2D(center,angle,1)
                if img.shape[2]==1:
                    return cv2.warpAffine(img,M,(cols,rows))[:,:,np.newaxis]
                else:
                    return cv2.warpAffine(img,M,(cols,rows))
            rand_num=random.randint(self.angle[0],self.angle[1])
            wei=1+abs(rand_num/3)
            img=rotate(img,rand_num)
            box=np.array(annots)[0]
            offsetx=0
            offsety=0
            for i in range(4,200):
                if(i%2==0):
                    x_=box[i]
                    y_=box[i+1]
                    x = x_ - w/2
                    y = y_ - h/2
                    angle=rand_num
                    box[i] = int(x * math.cos(math.radians(angle)) + y * math.sin(math.radians(angle)) + w/2)
                    box[i+1] = int(-x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)) + h/2)
                    if i==112:
                        offsetx=box[i]-x_
                        offsety=box[i+1]-y_
            box[0]+=offsetx
            box[2]+=offsetx
            box[1]+=offsety
            box[3]+=offsety

            return {'img': torch.tensor(img), 'annot': torch.tensor(box[np.newaxis,:]),'weight':weight+wei}
        return sample

class RandomErasing(object):
    def __init__(self,p=0.4):
        self.p=p
    def __call__(self, sample):
            if(np.random.rand()<self.p):
                image, annots = np.array(sample['img']), np.array(sample['annot'][0])
                box1,box2,box3,box4=int(annots[0]),int(annots[1]),int(annots[2]),int(annots[3])
                randx1=random.randint(box1,box3)
                randx2=random.randint(box1,box3)
                randy1=random.randint(box2,box4)
                randy2=random.randint(box2,box4)
                x1=min(randx1,randx2)
                x2=max(randx1,randx2)
                y1=min(randy1,randy2)
                y2=max(randy1,randy2)
                num1=int(random.randint(0,255))
                num2=int(random.randint(0,255))
                num3=int(random.randint(0,255))
                for y in range(y1,y2):
                    for x in range(x1,x2):
                        try:
                            image[y][x]=[num1,num2,num3]
                        except:
                            return sample
                return {'img': torch.tensor(image), 'annot': torch.tensor(annots[np.newaxis,:]),'weight':sample['weight']}
            return sample

class Color(object):
    def __call__(self,sample):
        if(np.random.rand()<0.2):
            img = np.array(sample['img'])
            img=Image.fromarray(img.astype('uint8'))
            img=transforms.ColorJitter(brightness=0.5,contrast=0.1,saturation=0.2,hue=0.2)(img)
            img=np.array(img)
            return {'img': torch.tensor(img), 'annot': sample['annot'],'weight':sample['weight']+5}
        else:
            return sample
        
            



class Resizer(object):
    def __init__(self,input_size=None):
        if input_size==None:
            self.input_size=None
        else:
            self.input_size=input_size
    def __call__(self, sample):
        image, annots = np.array(sample['img']), sample['annot']
        if self.input_size==None:
            if random.random()<0.3:
                input_size=random.randint(500,640)
            else:
                input_size=640
        else:
            input_size=self.input_size
        rows, cols, _ = image.shape 
        long_side = max(rows, cols)
        scale = input_size / long_side

        # resize image
        resized_image = skimage.transform.resize(image,(int(rows*input_size / long_side),int(cols*input_size / long_side)))
        resized_image = resized_image * 255
        
        assert (resized_image.shape[0]==input_size or resized_image.shape[1]==input_size), 'resized image size not {}'.format(input_size)

        if annots.shape[1] > 4 :
            annots = annots * scale
        else :
            annots[:,:4] = annots[:,:4] * scale

        return {'img': torch.tensor(resized_image), 'annot': annots,'weight': sample['weight']}




    


class PadToSquare(object):
    def __call__(self, sample, input_size=640):    
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape
        dim_diff = np.abs(rows - cols)
        input_size=max(rows,cols)
        # relocate bbox annotations
        if rows == input_size:
            diff = input_size - cols
            annots[:,0] = annots[:,0] + diff/2
            annots[:,2] = annots[:,2] + diff/2
        elif cols == input_size:
            diff = input_size - rows
            annots[:,1] = annots[:,1] + diff/2
            annots[:,3] = annots[:,3] + diff/2
        if annots.shape[1] > 4 :
            ldm_mask = annots[:,4] > 0
            if rows == input_size:
                diff = input_size - cols
                annots[ldm_mask,4::2] = annots[ldm_mask,4::2] + diff/2
            elif cols == input_size:
                diff = input_size - rows
                annots[ldm_mask,5::2] = annots[ldm_mask,5::2] + diff/2

        # pad image to square
        img = image
        img = img.permute(2,0,1)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if rows <= cols else (pad1, pad2, 0, 0)

        padded_img = F.pad(img, pad, "constant", value=0)
        
        # # pad to input size
        pad_=640-padded_img.shape[1]
        num1= random.randint(0,pad_)
        num2= random.randint(0,pad_)
        
        pading = (num1, pad_-num1,num2,pad_-num2)
        padded_img = F.pad(padded_img, pading, "constant", value=0)
        for i in range(0,200):
            if i%2==0:
                annots[0,i]+=num1
            else:
                annots[0,i]+=num2
        padded_img = padded_img.permute(1,2,0)

        return {'img': padded_img, 'annot': annots,'weight':sample['weight']}
def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)

class RandomCroper(object):
    def __call__(self, sample, input_size=640):
        image, annots = np.array(sample['img']), np.array(sample['annot'])
        height, width, _ = image.shape  
        wei=sample['weight']
        for _ in range(100):
        # while(True):
            if random.uniform(0, 1) <= 0.2:
                scale = 1.0
            else:
                scale = random.uniform(0.3, 1.0)
                wei+=0.05

            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                l = 0
            else:
                l = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))                

            value = matrix_iof(annots[:,:4], roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (annots[:, :2] + annots[:, 2:4]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)

            l_mask = annots[:,4] > 0
            l_mask = l_mask & mask_a

            if annots[mask_a, :4].shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            annots[mask_a, :2] = np.maximum(annots[mask_a, :2], roi[:2])
            annots[mask_a, :2] -= roi[:2]
            annots[mask_a, 2:4] = np.minimum(annots[mask_a, 2:4], roi[2:])
            annots[mask_a, 2:4] -= roi[:2]

            annots[l_mask, 4::2] -= roi[0]
            annots[l_mask, 5::2] -= roi[1]
            annots[l_mask, 4::2] = np.maximum(annots[l_mask, 4::2], np.array([0]))
            annots[l_mask, 5::2] = np.maximum(annots[l_mask, 5::2], np.array([0]))
            annots[l_mask, 4::2] = np.minimum(annots[l_mask, 4::2], roi[2] - roi[0])
            annots[l_mask, 5::2] = np.minimum(annots[l_mask, 5::2], roi[3] - roi[1])

            b_w_t = (annots[mask_a, 2] - annots[mask_a, 0] + 1) / w * input_size
            b_h_t = (annots[mask_a, 3] - annots[mask_a, 1] + 1) / h * input_size
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            annots = annots[mask_a,:][mask_b]

            if annots.shape[0] == 0:
                continue

            # return {'img': torch.from_numpy(image_t), 'annot': torch.from_numpy(annots)}
            return {'img': torch.tensor(image_t), 'annot': torch.tensor(annots),'weight':wei}

        return sample

class ValDataset(Dataset):
    def __init__(self,txt_path=None,transform=None,flip=False):
        self.words = []
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_size = 640
        # self.file=torch.load('anno.pth')
        path="/versa/elvishelvis/96RetinaYang/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
        file_=OrderedDict()
        f = open(path,'r')
        lines = f.readlines()
        for num ,line in enumerate( lines):
            annotation = np.zeros((1,200))
            result=line.split(' ')
            annotation[0,0] = result[196]  
            annotation[0,1] = result[197]    
            annotation[0,2] = result[198] 
            annotation[0,3] = result[199]  
            for i in range(4,200):
                annotation[0,i] = result[i-4]
            weight=int(result[200])+int(result[201])+int(result[202])+int(result[203])+int(result[204])+int(result[205])
            file_[num]=(result[-1][:-1],annotation,weight)
        self.file=file_
        # torch.save(file_, 'anno.pth')
    def __len__(self):
        # return len(self.name_list)   
        # return 10
        return int(len(self.file)/2)
        # return 200
        # return 10

    def __getitem__(self,index):
        cur=self.file[index]
        annotations = np.zeros((0, 200))
        annotations = np.append(annotations,cur[1],axis=0)
        img=skimage.io.imread("/versa/elvishelvis/96RetinaYang/WFLW/WFLW_images/"+cur[0])
        num=cur[-1]
        if(num==0):
            num=1
        sample = {'img':torch.tensor(img), 'annot':torch.tensor(annotations),'weight':torch.tensor(num)}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    


class ValDataset_CeleB(Dataset):
    def __init__(self,txt_path=None,transform=None,flip=False):
        self.words = []
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_size = 640
        self.name_list=[]
        self.bbox = []
        self.landmarks=[]
        path1="/versa/elvishelvis/RetinaFace_Pytorch/CelebA/Anno/list_bbox_celeba.txt"
        # for the bbox
        f = open(path1,'r')
        f.readline()
        f.readline()
        lines = f.readlines()
        for line in lines:
            self.name_list.append(line[0:10])
            count=0
            begin=11
            temp=[]
            is_first=False
            while (count<4):
                while(line[begin]==" "):
                    begin+=1
                cur=begin
                while(line[cur]!=" " and line[cur]!='\n'):
                    cur+=1
                temp.append(line[begin:cur])
                is_first=True
                begin=cur
                count+=1
            self.bbox.append(temp)

        path2="/versa/elvishelvis/RetinaFace_Pytorch/CelebA/Anno/list_landmarks_celeba.txt"
        k = open(path2,'r')
        k.readline()
        k.readline()
        lines = k.readlines()
        for line in lines:
            count=0
            begin=11
            temp=[]
            is_first=False
            while (count<10):
                while(line[begin]==" "):
                    begin+=1
                cur=begin
                while(line[cur]!=" " and line[cur]!='\n'):
                    cur+=1
                temp.append(line[begin:cur])
                is_first=True
                begin=cur
                count+=1
            self.landmarks.append(temp)

    def __len__(self):
        # return len(self.name_list)   
        return 20
        # return 30

    def __getitem__(self,index):
        img = skimage.io.imread("/versa/elvishelvis/RetinaFace_Pytorch/\
CelebA/Img/img_celeba.7z/img_celeba/"+str(self.name_list[int(index)]))
        #img = img.astype(np.float32)/255.0

        box_ = self.bbox[int(index)]
        land_=self.landmarks[int(index)]
        annotations = np.zeros((0, 14))
        if len(box_) == 0:
            return annotations
        annotation = np.zeros((1,14))
        # bbox
        annotation[0,0] = box_[0]                  # x1
        annotation[0,1] = box_[1]                  # y1
        annotation[0,2] = str(int(box_[0]) + int(box_[2]))       # x2
        annotation[0,3] = str(int(box_[1]) + int(box_[3]))       # y2

        # landmarks
        annotation[0,4] = land_[0]                  # l0_x
        annotation[0,5] = land_[1]                  # l0_y
        annotation[0,6] = land_[2]                  # l1_x
        annotation[0,7] = land_[3]                  # l1_y
        annotation[0,8] = land_[4]                 # l2_x
        annotation[0,9] = land_[5]                 # l2_y
        annotation[0,10] = land_[6]                # l3_x
        annotation[0,11] = land_[7]                # l3_y
        annotation[0,12] = land_[8]                # l4_x
        annotation[0,13] = land_[9]                # l4_y

        annotations = np.append(annotations,annotation,axis=0)
        sample = {'img':img, 'annot':torch.tensor(annotations),'weight':torch.tensor(1)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
