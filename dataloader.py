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
        self.img_size = 640

    def __len__(self):
        # return len(self.name_list)   
        # return 10
        # return 22995
        return 1000
        # return 10

    def __getitem__(self,index):
        img = cv2.imread("/versa/elvishelvis/landmarks56/new_dataset/{}.jpg".format(index))
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            import random
            rad=random.randint(1,22995)
            return self.__getitem__(rad)

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
        annotation[0,0] = minx  -int((maxx-minx)/10)                # x1
        annotation[0,1] = miny    -int((maxy-miny)/10)               # y1
        annotation[0,2] = maxx +int((maxx-minx)/10)   
        annotation[0,3] = maxy  +int((maxy-miny)/10)   

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


class RandomFlip(object):
    def __call__(self, sample, input_size=320, flip_x=0.4):
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
            annots[0, 0],annots[0, 2]=annots[0, 2],annots[0, 0]
            for k in range(4,20):
                if(k%2==0):
                    annots[0, k],annots[0, (40-k)]=annots[0, (40-k)],annots[0, k]
                else:
                    annots[0, k],annots[0, (42-k)]=annots[0, (42-k)],annots[0, k]
            for b in range(38,48):
                if(b%2==0):
                    annots[0, b],annots[0, (94-b)]=annots[0, (94-b)],annots[0, b]
                else:
                    annots[0, b],annots[0, (96-b)]=annots[0, (96-b)],annots[0, b]
            for a in range(76,84):
                if(a%2==0):
                    annots[0, a],annots[0, (170-a)]=annots[0, (170-a)],annots[0, a]
                else:
                    annots[0, a],annots[0, (172-a)]=annots[0, (172-a)],annots[0, a]

            annots[0, 86],annots[0, 96]=annots[0, 96],annots[0, 86]
            annots[0, 84],annots[0, 98]=annots[0, 98],annots[0, 84]

            annots[0, 66],annots[0, 74]=annots[0, 74],annots[0, 66]
            annots[0, 67],annots[0, 75]=annots[0, 75],annots[0, 67]
            annots[0, 68],annots[0, 72]=annots[0, 72],annots[0, 68]
            annots[0, 69],annots[0, 73]=annots[0, 73],annots[0, 69]

            annots[0, 100],annots[0, 112]=annots[0, 112],annots[0, 100]
            annots[0, 102],annots[0, 110]=annots[0, 110],annots[0, 102]
            annots[0, 104],annots[0, 108]=annots[0, 108],annots[0, 104]
            annots[0, 126],annots[0, 130]=annots[0, 130],annots[0, 126]
            annots[0, 138],annots[0, 134]=annots[0, 134],annots[0, 138]
            annots[0, 116],annots[0, 120]=annots[0, 120],annots[0, 116]
            annots[0, 114],annots[0, 122]=annots[0, 122],annots[0, 114]
            annots[0, 124],annots[0, 132]=annots[0, 132],annots[0, 124]

        

            image = torch.from_numpy(image)
            annots = torch.from_numpy(annots)

            sample = {'img': image, 'annot': annots}

        return sample


class Rotate(object):
    def __init__(self,angle=[-45,45],p=0.3):
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

            box[0] = minx  -int((maxx-minx)/10)                # x1
            box[1] = miny    -int((maxy-miny)/10)               # y1
            box[2] = maxx +int((maxx-minx)/10)   
            box[3] = maxy  +int((maxy-miny)/10)  

            return {'img': torch.tensor(img), 'annot': torch.tensor(box[np.newaxis,:])}
        return sample

class RandomErasing(object):
    def __init__(self,p=0.3):
        self.p=p
    def __call__(self, sample):
            if(np.random.rand()<self.p):
                image, annots = np.array(sample['img']), np.array(sample['annot'][0])
                box1,box2,box3,box4=int(annots[0]),int(annots[1]),int(annots[2]),int(annots[3])
                randx1=random.randint(box1-3,box3-3)
                randx2=random.randint(box1-3,box3-3)
                randy1=random.randint(box2-3,box4-3)
                randy2=random.randint(box2-3,box4-3)
                x1=min(randx1,randx2)
                x2=max(randx1,randx2)
                y1=min(randy1,randy2)
                y2=max(randy1,randy2)
                for y in range(y1,y2):
                    for x in range(x1,x2):
                        try:
                            image[x][y]=[0,0,0]
                        except:
                            return sample
                return {'img': torch.tensor(image), 'annot': torch.tensor(annots[np.newaxis,:])}
            return sample

class Color(object):
    def __call__(self,sample):
        img = np.array(sample['img'])
        img=Image.fromarray(img.astype('uint8'))
        img=transforms.ColorJitter()(img)
        img=np.array(img)
        return {'img': torch.tensor(img), 'annot': sample['annot']}
            



class Resizer(object):
    def __init__(self,input_size=640):
        if input_size==None:
            self.input_size=None
        else:
            self.input_size=input_size
    def __call__(self, sample):
        image, annots = np.array(sample['img']), sample['annot']
        if self.input_size==None:
            if random.random()<0.4:
                input_size=random.randint(250,640)
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
        
        return {'img': torch.tensor(resized_image), 'annot': annots}




    


class PadToSquare(object):
    def __call__(self, sample, input_size=640):    
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

        # pad image to square
        img = image
        img = img.permute(2,0,1)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if rows <= cols else (pad1, pad2, 0, 0)

        padded_img = F.pad(img, pad, "constant", value=0)
        
        # # pad to input size
        pad_=input_size-padded_img.shape[1]
        num1= random.randint(0,pad_)
        num2= random.randint(0,pad_)
        
        pading = (num1, pad_-num1,num2,pad_-num2)
        padded_img = F.pad(padded_img, pading, "constant", value=0)
        for i in range(0,140):
            if i%2==0:
                annots[0,i]+=num1
            else:
                annots[0,i]+=num2
        padded_img = padded_img.permute(1,2,0)

        return {'img': padded_img, 'annot': annots}


class ValDataset(Dataset):
    def __init__(self,txt_path=None,transform=None,flip=False):
        self.words = []
        self.transform = transform
        self.flip = flip
        self.batch_count = 0
        self.img_size = 640

    def __len__(self):
        # return len(self.name_list)   
        return 299
        # return 50
        # return 10

    def __getitem__(self,index):
        index+=1
        img = cv2.imread("/versa/elvishelvis/landmarks56/300w/{}.jpg".format(index))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.astype(np.float32)/255.0

        annotations = np.zeros((0, 4+136))
        annotation = np.zeros((1,140))
        landmark=[]
        minx=float('inf')
        miny=float('inf')
        maxx=0
        maxy=0
        label=[]
        with open("/versa/elvishelvis/landmarks56/300w/{}.pts".format(index),'r') as f:
            f.readline()
            f.readline()
            f.readline()
            while(True):
                try:
                    item=f.readline()
                    label.append([float(item[0:7]),float(item[8:15])])
                    item[2]
                except:
                    break
        # label=torch.tensor(label)
        for da in label:
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
        annotation[0,0] = minx  -int((maxx-minx)/5)                # x1
        annotation[0,1] = miny    -int((maxy-miny)/5)               # y1
        annotation[0,2] = maxx +int((maxx-minx)/5)   
        annotation[0,3] = maxy  +int((maxy-miny)/5)   
        if(len(landmark)!=136):
            return self.__getitem__(index+1)
        for i in range(4,140):
            annotation[0,i] = landmark[i-4]
        annotations = np.append(annotations,annotation,axis=0)
        sample = {'img':torch.tensor(img), 'annot':torch.tensor(annotations)}
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
        sample = {'img':img, 'annot':torch.tensor(annotations)}
        if self.transform is not None:
            sample = self.transform(sample)
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



'''
class RandomCroper(object):
    def __call__(self, sample, input_size=640):
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