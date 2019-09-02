
import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models.resnet as resnet
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from anchors import Anchors
from utils import RegressionTransform
import losses
from mobile import mobileV1
# from mnas import MnasNet

class ContextModule(nn.Module):
    def __init__(self,in_channels=256):
        super(ContextModule,self).__init__()
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2)
        )
        self.det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2)
        )
        self.det_concat_relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.det_conv1(x)
        x_ = self.det_context_conv1(x)
        x2 = self.det_context_conv2(x_)
        x3_ = self.det_context_conv3_1(x_)
        x3 = self.det_context_conv3_2(x3_)

        out = torch.cat((x1,x2,x3),1)
        act_out = self.det_concat_relu(out)

        return act_out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.rf_c3_lateral = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c3_det_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=32, eps=2e-05, momentum=0.9))

        self.rf_c3_det_context_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c3_det_context_conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c3_det_context_conv3_1 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c3_det_context_conv3_2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c3_det_concat_relu = nn.Sequential(
                    nn.ReLU(inplace=True))

        self.rf_c2_lateral = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(num_features=64),
                    nn.ReLU(inplace=True))

        self.rf_c3_upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'))

        self.rf_c2_aggr = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c2_det_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=32, eps=2e-05, momentum=0.9))
        
        self.rf_c2_det_context_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c2_det_context_conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c2_det_context_conv3_1 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.rf_c2_det_context_conv3_2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c2_det_concat_relu = nn.Sequential(
                    nn.ReLU(inplace=True))

        self.rf_c1_red_conv = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c2_upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'))

        self.rf_c1_aggr = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c1_det_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=32, eps=2e-05, momentum=0.9))
        
        self.rf_c1_det_context_conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))

        self.rf_c1_det_context_conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c1_det_context_conv3_1 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
                    nn.ReLU(inplace=True))
        
        self.rf_c1_det_context_conv3_2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c1_det_concat_relu = nn.Sequential(
                    nn.ReLU(inplace=True))

        

    def forward(self,x_dict):
        names = list(x_dict.keys())
        result=[]


        #the plane of shape 256,256 (x26)
        x26=x_dict[3]
        o1 = self.rf_c3_lateral(x26)
        o2 = self.rf_c3_det_conv1(o1)
        o3 = self.rf_c3_det_context_conv1(o1)
        o4 = self.rf_c3_det_context_conv2(o3)
        o5 = self.rf_c3_det_context_conv3_1(o3)
        o6 = self.rf_c3_det_context_conv3_2(o5)
        o7 = torch.cat((o2, o4, o6), 1)
        o8 = self.rf_c3_det_concat_relu(o7)


        #the plane of shape 128,128 (x22)

        x22=x_dict[2]
        p1 = self.rf_c2_lateral(x22)
        p2 = self.rf_c3_upsampling(o1)
        p2 = F.adaptive_avg_pool2d(p2, (p1.shape[2], p1.shape[3])) 
        p3 = p1 + p2
        p4 = self.rf_c2_aggr(p3)
        p5 = self.rf_c2_det_conv1(p4)
        p6 = self.rf_c2_det_context_conv1(p4)
        p7 = self.rf_c2_det_context_conv2(p6)
        p8 = self.rf_c2_det_context_conv3_1(p6)
        p9 = self.rf_c2_det_context_conv3_2(p8)
        p10 = torch.cat((p5, p7, p9), 1)
        p10 = self.rf_c2_det_concat_relu(p10)

        #the plane of shape 64,64 (x10)
        x10=x_dict[1]
        q1 = self.rf_c1_red_conv(x10)
        q2 = self.rf_c2_upsampling(p4)
        q2 = F.adaptive_avg_pool2d(q2, (q1.shape[2], q1.shape[3]))
        q3 = q1 + q2
        q4 = self.rf_c1_aggr(q3)
        q5 = self.rf_c1_det_conv1(q4)
        q6 = self.rf_c1_det_context_conv1(q4)
        q7 = self.rf_c1_det_context_conv2(q6)
        q8 = self.rf_c1_det_context_conv3_1(q6)
        q9 = self.rf_c1_det_context_conv3_2(q8)
        q10 = torch.cat((q5, q7, q9), 1)
        q10 = self.rf_c2_det_concat_relu(q10)


        result.append(q10)
        result.append(p10)
        result.append(o8)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, result)])

        return out

class ClassHead(nn.Module):
    def __init__(self,inchannels=64,num_anchors=2):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.face_rpn_cls_score_stride8 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*2, kernel_size=1, stride=1, padding=0, bias=True))
        self.face_rpn_cls_score_stride16 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*2, kernel_size=1, stride=1, padding=0, bias=True))
        self.face_rpn_cls_score_stride32 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*2, kernel_size=1, stride=1, padding=0, bias=True))
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self,x,idx):
        if(idx==0):
            out = self.face_rpn_cls_score_stride8(x)
        if(idx==1):
            out = self.face_rpn_cls_score_stride16(x)
        if(idx==2):
            out = self.face_rpn_cls_score_stride32(x)
        out = out.permute(0,2,3,1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        
        return out.contiguous().view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=64,num_anchors=2):
        super(BboxHead,self).__init__()
        self.num_anchors=num_anchors
        self.face_rpn_bbox_pred_stride8 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*4, kernel_size=1, stride=1, padding=0, bias=True))
        self.face_rpn_bbox_pred_stride16 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*4, kernel_size=1, stride=1, padding=0, bias=True))
        self.face_rpn_bbox_pred_stride32 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*4, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self,x,idx):
        if(idx==0):
            out = self.face_rpn_bbox_pred_stride8(x)
        if(idx==1):
            out = self.face_rpn_bbox_pred_stride16(x)
        if(idx==2):
            out = self.face_rpn_bbox_pred_stride32(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=64,num_anchors=2):
        super(LandmarkHead,self).__init__()
        self.num_anchors=num_anchors
        self.face_rpn_landmark_pred_stride8 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*136, kernel_size=1, stride=1, padding=0, bias=True))
        self.face_rpn_landmark_pred_stride16 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*136, kernel_size=1, stride=1, padding=0, bias=True))
        self.face_rpn_landmark_pred_stride32= nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=self.num_anchors*136, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self,x,idx):
        if(idx==0):
            out = self.face_rpn_landmark_pred_stride8(x)
        if(idx==1):
            out = self.face_rpn_landmark_pred_stride16(x)
        if(idx==2):
            out = self.face_rpn_landmark_pred_stride32(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 136)

class RetinaFace(nn.Module):
    def __init__(self,backbone,return_layers,anchor_nums=2):
        super(RetinaFace,self).__init__()
        # if backbone_name == 'resnet50':
        #     self.backbone = resnet.resnet50(pretrained)
        # self.backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
        # self.return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        assert backbone,'Backbone can not be none!'
        assert len(return_layers)>0,'There must be at least one return layers'
        # self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        self.body=mobileV1()
        # self.body=MnasNet()
        in_channels_stage2 = 64
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4
        ]
        out_channels = 64
        self.fpn = FeaturePyramidNetwork(in_channels_list,out_channels)
        # self.ClassHead = ClassHead()
        # self.BboxHead = BboxHead()
        # self.LandmarkHead = LandmarkHead()
        # self.ClassHead = ClassHead
        # self.BboxHead = BboxHead
        # self.LandmarkHead = LandmarkHead
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()        
        self.losslayer = losses.LossLayer()
        self.bbx=BboxHead().cuda()
        self.ldm=LandmarkHead().cuda()
        self.clls=ClassHead().cuda()

    # def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
    #     classhead = nn.ModuleList()
    #     for i in range(fpn_num):
    #         classhead.append(ClassHead(inchannels,anchor_num))
    #     return classhead
    
    # def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
    #     bboxhead = nn.ModuleList()
    #     for i in range(fpn_num):
    #         bboxhead.append(BboxHead(inchannels,anchor_num))
    #     return bboxhead

    # def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
    #     landmarkhead = nn.ModuleList()
    #     for i in range(fpn_num):
    #         landmarkhead.append(LandmarkHead(inchannels,anchor_num))
    #     return landmarkhead

    def forward(self,inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        out = self.body(img_batch)

        features = self.fpn(out)


        # bbox_regressions = torch.cat([self.BboxHead(feature) for feature in features.values()], dim=1)
        # ldm_regressions = torch.cat([self.LandmarkHead(feature) for feature in features.values()], dim=1)
        # classifications = torch.cat([self.ClassHead(feature) for feature in features.values()],dim=1)      

        #(80**2+40**2+20**2)*num_anchors
        bbox_regressions = torch.cat([self.bbx(feature,idx) for idx, feature in enumerate(features.values())], dim=1)
        ldm_regressions = torch.cat([self.ldm(feature,idx) for idx, feature in enumerate(features.values())], dim=1)
        classifications = torch.cat([self.clls(feature,idx) for idx, feature in enumerate(features.values())],dim=1)
        anchors = self.anchors(img_batch)
        
        if self.training:
            return self.losslayer(classifications, bbox_regressions,ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)

            return classifications, bboxes, landmarks


def create_retinaface(return_layers,backbone_name='resnet50',anchors_num=2,pretrained=True):
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
    model = RetinaFace(backbone,return_layers,anchor_nums=2)

    return model
if __name__ == "__main__":
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    retinaface = create_retinaface(return_layers)
    print(retinaface)