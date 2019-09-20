import numpy as numpy
import torch.nn as nn
import torch
import math
# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=3, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def filt_IoU(a, b, l):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    ldm_sum = l.sum(dim=1)
    mask = ldm_sum<0
    ldm_mask = torch.ones_like(mask)
    ldm_mask[mask] = -1
    filted_IoU = IoU * ldm_mask.float()

    return IoU, filted_IoU

class LossLayer(nn.Module):
    def __init__(self):
        super(LossLayer, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self,classifications,bbox_regressions,ldm_regressions,anchors,annotations):
        batch_size = classifications.shape[0]
        classification_losses = []
        bbox_regression_losses = []
        ldm_regression_losses = []  

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights  

        #temp
        positive_indices_list = []    

        for j in range(batch_size):
            classification = classifications[j,:,:]
            bbox_regression = bbox_regressions[j,:,:]
            ldm_regression = ldm_regressions[j,:,:]

            annotation = annotations[j,:,:]
            # annotation = annotation[annotation[:,0] != -1]
            annotation = annotation[annotation[:,0] > 0]
            bbox_annotation = annotation[:,:4]
            ldm_annotation = annotation[:,4:]

            if bbox_annotation.shape[0] == 0:
                bbox_regression_losses.append(torch.tensor(0.,requires_grad=True).cuda())
                classification_losses.append(torch.tensor(0.,requires_grad=True).cuda())
                ldm_regression_losses.append(torch.tensor(0.,requires_grad=True).cuda())

                # temp
                positive_indices_list.append([])

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            #IoU, filt_iou = filt_IoU(anchors[0, :, :], bbox_annotation, ldm_annotation)

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            targets = torch.ones(classification.shape)*-1
            targets = targets.cuda()       
            
            # those whose iou<0.3 have no object
            negative_indices = torch.lt(IoU_max, 0.4)
            targets[negative_indices, :] = 0
            targets[negative_indices, 1] = 1

            # those whose iou>0.5 have object
            positive_indices = torch.ge(IoU_max, 0.7)

            #temp
            positive_indices_list.append(positive_indices)

            num_positive_anchors = positive_indices.sum()

            #keep positive and negative ratios with 1:3
            keep_negative_anchors = num_positive_anchors * 3

            bbox_assigned_annotations = bbox_annotation[IoU_argmax, :]
            ldm_assigned_annotations = ldm_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, 0] = 1

            # ignore targets with no landmarks
            # f_IoU_max ,f_IoU_argmax = torch.max(filt_iou, dim=1)
            # ldm_positive_indices = torch.ge(f_IoU_max, 0.5)
            
            ldm_sum = ldm_assigned_annotations.sum(dim=1)
            ge0_mask = ldm_sum > 0
            ldm_positive_indices = ge0_mask & positive_indices

            # OHEM
            negative_losses = classification[negative_indices,1] * -1
            sorted_losses, _ = torch.sort(negative_losses, descending=True)
            if sorted_losses.numel() > keep_negative_anchors:
                sorted_losses = sorted_losses[:keep_negative_anchors]
            positive_losses = classification[positive_indices,0] * -1
                
            focal_loss = False
            # focal loss
            if focal_loss:
                alpha = 0.25
                gamma = 2.0            
                alpha_factor = torch.ones(targets.shape).cuda() * alpha

                alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

                classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            else:
                if positive_indices.sum() > 0:
                    classification_losses.append(positive_losses.mean() + sorted_losses.mean())
                else:
                    classification_losses.append(torch.tensor(0.,requires_grad=True).cuda())


            # compute bboxes loss
            if positive_indices.sum() > 0:
                # bbox
                bbox_assigned_annotations = bbox_assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = bbox_assigned_annotations[:, 2] - bbox_assigned_annotations[:, 0]
                gt_heights = bbox_assigned_annotations[:, 3] - bbox_assigned_annotations[:, 1]
                gt_ctr_x   = bbox_assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = bbox_assigned_annotations[:, 1] + 0.5 * gt_heights

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi + 1e-14)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi + 1e-14)
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                bbox_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                bbox_targets = bbox_targets.t()

                # Rescale
                bbox_targets = bbox_targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                # smooth L1
                # box losses
                bbox_regression_loss = self.smoothl1(bbox_targets,bbox_regression[positive_indices, :])
                bbox_regression_losses.append(bbox_regression_loss)
            else:
                bbox_regression_losses.append(torch.tensor(0.,requires_grad=True).cuda())  

            # compute landmarks loss
            if ldm_positive_indices.sum() > 0 :
                ldm_assigned_annotations = ldm_assigned_annotations[ldm_positive_indices, :]

                anchor_widths_l = anchor_widths[ldm_positive_indices]
                anchor_heights_l = anchor_heights[ldm_positive_indices]
                anchor_ctr_x_l = anchor_ctr_x[ldm_positive_indices]
                anchor_ctr_y_l = anchor_ctr_y[ldm_positive_indices]
                ldm_targets=[]
                for i in range(0,196):  
                    if i %2==0:
                        candidate=(ldm_assigned_annotations[:,i] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                    else:
                        candidate=(ldm_assigned_annotations[:,i] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)
                    ldm_targets.append(candidate)
                ldm_targets=torch.stack((ldm_targets))
                ldm_targets = ldm_targets.t()

                # Rescale
                scale = torch.ones(1,196)*0.1
                ldm_targets = ldm_targets/scale.cuda()
                # increase the weight for lips
                s1 = torch.ones(1,68)
                s2 = torch.ones(1,128)*3
                s=torch.cat([s1,s2],dim=-1).cuda()
                aaaaaaa=WingLoss()
                ldm_regression_loss = WingLoss(ldm_targets*s, ldm_regression[ldm_positive_indices, :]*s)
                ldm_regression_losses.append(ldm_regression_loss)
            else:
                ldm_regression_losses.append(torch.tensor(0.,requires_grad=True).cuda())

        return torch.stack(classification_losses), torch.stack(bbox_regression_losses),torch.stack(ldm_regression_losses)
