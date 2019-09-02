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
import cv2
from scipy import misc

class TrainDataset(Dataset):
    def __init__(self,txt_path=None,transform=None,flip=False):
        self.words = []
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_size = 320

    def __len__(self):
        # return len(self.name_list)   
        return 22986
        # return 10

    def __getitem__(self,index):
        index+=1
        img = cv2.imread("/versa/elvishelvis/landmarks56/new_dataset/{}.jpg".format(index))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.astype(np.float32)/255.0

        annotations = np.zeros((0, 4+136))
        annotation = np.zeros((1,140))
        landmark=[]
        minx=float('inf')
        miny=float('inf')
        maxx=0
        maxy=0
        path="/versa/elvishelvis/landmarks56/new_dataset/{}.pth".format(index)
        data=np.array(torch.load(path))
        for da in data:
            if(da[0]<minx):
                minx=da[0]
            if(da[0]>maxx):
                maxx=da[0]
            if(da[1]<miny):
                miny=da[1]
            if(da[1]>maxy):
                maxy=da[1]
            landmark.append(da[0])
            landmark.append(da[1])
        # bbox
        annotation[0,0] = minx                  # x1
        annotation[0,1] = miny                  # y1
        annotation[0,2] = maxx      # x2
        annotation[0,3] = maxy       # y2

        for i in range(4,140):
            annotation[0,i] = landmark[i-4]
        annotations = np.append(annotations,annotation,axis=0)
        sample = {'img':torch.tensor(img), 'annot':torch.tensor(annotations)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample



def collater(data):
    batch_size = len(data)

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

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
            annot_padded = torch.ones((len(annots), max_num_annots, 140)) * -1
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
            annot_padded = torch.ones((len(annots), 1, 140)) * -1
        else:
            annot_padded = torch.ones((len(annots), 1, 4)) * -1

    batched_imgs = batched_imgs.permute(0, 3, 1, 2)

    return {'img': batched_imgs, 'annot': annot_padded}
'''
class RandomCroper(object):
    def __call__(self, sample, input_size=320):
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape        
        
        smallest_side = min(rows, cols)
        longest_side = max(rows,cols)
        scale = random.uniform(0.3,1)
        short_size = int(smallest_side * scale)
        start_short_upscale = smallest_side - short_size
        start_long_upscale = longest_side - short_size
        crop_short = random.randint(0,start_short_upscale)
        crop_long = random.randint(0,start_long_upscale)
        crop_y = 0
        crop_x = 0
        if smallest_side == rows:
            crop_y = crop_short
            crop_x = crop_long
        else:
            crop_x = crop_short
            crop_y = crop_long
        # crop        
        cropped_img = image[crop_y:crop_y + short_size,crop_x:crop_x + short_size]
        # resize
        new_image = skimage.transform.resize(cropped_img, (input_size, input_size))

        # why normalized from 255 to 1 after skimage.transform?????????
        new_image = new_image * 255

        # relocate bbox
        annots[:,0] -= crop_x
        annots[:,1] -= crop_y
        annots[:,2] -= crop_x
        annots[:,3] -= crop_y

        # relocate landmarks56
        if annots.shape[1] > 4:
            # l_mask = annots[:,4]!=-1
            l_mask = annots[:,4] > 0
            annots[l_mask,4] -= crop_x
            annots[l_mask,5] -= crop_y
            annots[l_mask,6] -= crop_x
            annots[l_mask,7] -= crop_y
            annots[l_mask,8] -= crop_x
            annots[l_mask,9] -= crop_y
            annots[l_mask,10]  -= crop_x
            annots[l_mask,11]  -= crop_y
            annots[l_mask,12]  -= crop_x
            annots[l_mask,13]  -= crop_y

        # scale annotations
        resize_scale = input_size/short_size
        annots[:,:4] = annots[:,:4] * resize_scale
        if annots.shape[1] > 4:
            annots[l_mask,4:] = annots[l_mask,4:] * resize_scale

        # remove faces center not in image afer crop
        center_x = (annots[:,0] + annots[:,2]) / 2
        center_y = (annots[:,1] + annots[:,3]) / 2

        mask_x = (center_x[:,]>0)&(center_x[:,]<input_size)
        mask_y = (center_y[:,]>0)&(center_y[:,]<input_size)

        mask = mask_x & mask_y

        # clip bbox
        annots[:,:4] = annots[:,:4].clip(0, input_size)
        
        # clip landmarks56
        if annots.shape[1] > 4:
            annots[l_mask,4:] = annots[l_mask,4:].clip(0, input_size)
        
        annots = annots[mask]  

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots)}
'''
class RandomFlip(object):
    def __call__(self, sample, input_size=320, flip_x=1):
        aaa=np.random.rand()
        if aaa < flip_x:
            image, annots = sample['img'], sample['annot']
            c,w,h=image.shape
            # flip image
            image = torch.flip(image,[1])

            image = image.numpy()
            annots = annots.numpy()
            # relocate bboxes
            for i in range(0,140):
                if i%2==0:
                    annots[0, i] = w - annots[0, i]
            for k in range(0,8):

                annots[0, k*2],annots[0, (16-k)*2]=annots[0, (16-k)*2],annots[0, k*2]
            for b in range(17,22):
                annots[0, b*2],annots[0, (43-b)*2]=annots[0, (43-b)*2],annots[0, b*2]
            for a in range(36,42):
                annots[0, a*2],annots[0, (81-a)*2]=annots[0, (81-a)*2],annots[0, a*2]
            for c in range(31,33):
                annots[0, c*2],annots[0, (66-c)*2]=annots[0, (66-c)*2],annots[0, c*2]
            # for d in range(31,33):
            #     annots[0, d*2],annots[0, (81-d)*2]=annots[0, (81-d)*2],annots[0, d*2]

            annots[0, 48*2],annots[0, 54*2]=annots[0, 54*2],annots[0, 48*2]
            annots[0, 49*2],annots[0, 53*2]=annots[0, 53*2],annots[0, 49*2]
            annots[0, 50*2],annots[0, 52*2]=annots[0, 52*2],annots[0, 50*2]
            annots[0, 59*2],annots[0, 55*2]=annots[0, 55*2],annots[0, 59*2]
            annots[0, 58*2],annots[0, 56*2]=annots[0, 56*2],annots[0, 58*2]
            annots[0, 60*2],annots[0, 64*2]=annots[0, 64*2],annots[0, 60*2]
            annots[0, 61*2],annots[0, 63*2]=annots[0, 63*2],annots[0, 61*2]
            annots[0, 67*2],annots[0, 65*2]=annots[0, 65*2],annots[0, 67*2]

        

            image = torch.from_numpy(image)
            annots = torch.from_numpy(annots)

            sample = {'img': image, 'annot': annots}

        return sample

class Resizer(object):
    def __call__(self, sample, input_size=320):
        image, annots = np.array(sample['img']), np.array(sample['annot'][0])
        # if(len(image.shape)==2):

            # print(image.shape)
            
        cols, rows,_ = image.shape 
        scalex=input_size/rows
        scaley=input_size/cols

        # resize image
        resized_image = skimage.transform.resize(image,(320,320))
        resized_image = resized_image * 255
        for i in range(140):
            if(i%2==0):
                annots[i]*=scalex
            else:
                annots[i]*=scaley
        return {'img': torch.tensor(resized_image), 'annot': torch.tensor(annots[np.newaxis,:])}


class Rotate(object):
    def __init__(self,angle=[-45,45],p=1):
        self.angle=angle
        self.p=p
    def __call__(self,sample):
        if(np.random.rand()<self.p):
            img=np.array(sample['img'])
            h,w,a=img.shape
            annots=sample['annot']
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
            img=rotate(img,rand_num)
            box=np.array(annots)[0]
            minx=float('inf')
            miny=float('inf')
            maxx=0
            maxy=0
            for i in range(4,140):
                if(i%2==0):
                    x_=box[i]
                    y_=box[i+1]
                    x = x_ - w/2
                    y = y_ - h/2
                    angle=rand_num
                    box[i] = int(x * math.cos(math.radians(angle)) + y * math.sin(math.radians(angle)) + w/2)
                    box[i+1] = int(-x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)) + h/2)
                    if(box[i]<minx):
                        minx=box[i]
                    if(box[i]>maxx):
                        maxx=box[i]
                    if(box[i+1]<miny):
                        miny=box[i+1]
                    if(box[i+1]>maxy):
                        maxy=box[i+1]
            box[0] = minx                  # x1
            box[1] = miny                  # y1
            box[2] = maxx      # x2
            box[3] = maxy 

            return {'img': torch.tensor(img), 'annot': torch.tensor(box[np.newaxis,:])}
        return sample
'''     
class PadToSquare(object):
    def __call__(self, sample, input_size=320):    
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape
        dim_diff = np.abs(rows - cols)

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

        # pad image
        img = torch.from_numpy(image).type(torch.cuda.IntTensor)
        
        img = img.permute(2,0,1)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if rows <= cols else (pad1, pad2, 0, 0)
        pad = torch.from_numpy(np.array(pad))
        padded_img = F.pad(img, pad, "constant", value=0)
        padded_img = padded_img.permute(1,2,0)

        annots = torch.from_numpy(annots)

        return {'img': padded_img, 'annot': annots}
'''
class RandomErasing(object):
    def __init__(self,p=1):
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
                for y in range(y1,y2):
                    for x in range(x1,x2):
                        image[y][x]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
                return {'img': torch.tensor(image), 'annot': torch.tensor(annots[np.newaxis,:])}
            return sample
'''
class ValDataset(Dataset):
    def __init__(self,txt_path,transform=None,flip=False):
        self.imgs_path = []
        self.words = []
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_size = 320
            
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        bbox = []
        for line in lines:
            line = line.rstrip() 
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)        
                    labels.clear()       
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)            
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)    

    def __getitem__(self,index):
        img = skimage.io.imread(self.imgs_path[index])

        labels = self.words[index]
        annotations = np.zeros((0, 4))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1,4))
            # bbox
            annotation[0,0] = label[0]                  # x1
            annotation[0,1] = label[1]                  # y1
            annotation[0,2] = label[2]       # x2
            annotation[0,3] =  label[3]       # y2

            annotations = np.append(annotations,annotation,axis=0)
        
        sample = {'img':img, 'annot':annotations}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.imgs_path)  

    def _load_annotations(self,index):
        labels = self.words[index]
        annotations = np.zeros((0,4))

        if len(labels) == 0:
            return annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1,4))
            annotation[0,0] = label[0]                  # x1
            annotation[0,1] = label[1]                  # y1
            annotation[0,2] = label[0] + label[2]       # x2
            annotation[0,3] = label[1] + label[3]       # y2                

            annotations = np.append(annotations, annotation, axis=0)

        return annotations
'''



class Color(object):
    def __call__(self,sample):
        img = np.array(sample['img'])
        img=Image.fromarray(img.astype('uint8'))
        img=transforms.ColorJitter()(img)
        img=np.array(img)
        return {'img': torch.tensor(img), 'annot': sample['annot']}
            





