import utils
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import torchvision.ops as ops
import cv2
import time
def get_detections(img_batch, model,score_threshold=0.5, iou_threshold=0.5):
    start=time.time()
    model.eval()
    model.cuda()
    img_batch.cuda()
    with torch.no_grad():
        #[1,16800,2]
        classifications, bboxes, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        
        for i in range(batch_size):
            #[16800,2]
            classification = torch.exp(classifications[i,:,:])
            bbox = bboxes[i,:,:]
            landmark = landmarks[i,:,:]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax==0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice
            
            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]
            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
        print(time.time()-start)
        return picked_boxes, picked_landmarks

def compute_overlap(a,b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    # (N, K) ndarray of overlap between boxes and query_boxes
    return torch.from_numpy(intersection / ua)    


def evaluate(val_data,retinaFace,threshold=0.5):
    recall = 0.
    precision = 0.
    landmark_loss=0
    miss=0
    #for i, data in tqdm(enumerate(val_data)):
    resssss=[]
    count=0
    for data in tqdm(iter(val_data)):
        img_batch = data['img'].cuda()
        annots = data['annot'].cuda()


        picked_boxes,picked_landmarks = get_detections(img_batch,retinaFace)
        recall_iter = 0.
        precision_iter = 0.
        for j, boxes in enumerate(picked_boxes):      
            annot_boxes = annots[j]
            annot_boxes = annot_boxes[annot_boxes[:,0]!=-1]
            annot_boxes=annot_boxes[:,:4]
            annot_land=annot_boxes[:,4:]
            if boxes is None and annot_boxes.shape[0] == 0:
                continue
            elif boxes is None and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes is not None and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.   
                continue         
            overlap = ops.boxes.box_iou(annot_boxes, boxes)
                 
            # compute recall
            max_overlap, _ = torch.max(overlap,dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num/annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap,dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives/boxes.shape[0]
        if (picked_landmarks==None):
            continue
        for i, land in enumerate(picked_landmarks):

            annot_land = annots[i]
            annot_land=annot_land[:,4:]
            # img_batch=np.array(img_batch[0].cpu()).transpose(1,2,0)
            try:
                
                land=land[0,:]
                landmark_loss=torch.mean(torch.sqrt(torch.sum((annot_land - land)**2)))
                offset=abs(int(annot_land[0][40])-int(annot_land[0][48]))
                # landmark_loss=nn.SmoothL1Loss()(annot_land,land)
                landmark_loss=int(landmark_loss/offset)
                if landmark_loss<10:
                    resssss.append(landmark_loss)
                # annot_land=np.array(annot_land[0].cpu())
                # land=np.array(land.cpu())
                # for kkk in range(0,136,2):
                #     img_batch=cv2.circle(img_batch,(annot_land[kkk],annot_land[kkk+1]),radius=1,color=(0,0,255),thickness=2)
                #     img_batch=cv2.circle(img_batch,(land[kkk],land[kkk+1]),radius=1,color=(0,255,0),thickness=2)
                # cv2.imwrite('{}.jpg'.format(count),img_batch)
                # count+=1
                # landmark_loss+=torch.mean((annot_land-land)**2).item()
            except:
                # print('miss')
                miss+=1
            
        recall += recall_iter/len(picked_boxes)
        precision += precision_iter/len(picked_boxes)
    return recall/len(val_data),precision/len(val_data), np.mean(resssss) ,miss











