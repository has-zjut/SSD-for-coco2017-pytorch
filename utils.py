import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch
# transform = transforms.Compose(
#     [transforms.ToTensor(),  ###transforms.Resize()y有两个参数
#      transforms.Normalize((0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])  ##均值，标准差
transform = transforms.Compose(
    [transforms.ToTensor()  ###transforms.Resize()y有两个参数
    ])  ##均值，标准差


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def subMean(bgr, mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr
def base_transform(image, size):
    # x = cv2.resize(image, (size, size)).astype(np.float32)
    image = image.astype(np.uint8)
    image = subMean(image,(104, 117, 123))
    # image = image/255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (size, size))
    x=transform(x)
    return x


class BaseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size), boxes, labels

from compute_iou import decode
from compute_iou import nms
def predict(model,img):

    img=img[None,:,:,:]
    img=img.cuda()
    pre_local,pre_conf,priors=model(img)
    # print(pre_local.shape)
    softmax=nn.Softmax(dim=-1)
    pre_conf=softmax(pre_conf)
    num=pre_local.shape[0]###batchsize
    num_priors=priors.shape[0]
    output=torch.zeros((num,81,200,5))
    pre_conf =pre_conf.view(num, num_priors, 81).transpose(2, 1)###【batchsize, 21,8732】
    for i in range(num):###第i幅图片
        decoded_boxes = decode(pre_local[i], priors, [0.1,0.2])
        conf_scores = pre_conf[i].clone()
        for cl in range(1, 81):###20个类别不包含背景类
            c_mask = conf_scores[cl].gt(0.06)######################################
            scores = conf_scores[cl][c_mask]###[c1这个类别对应的默认盒数目]里面的值为类别自信度 保留自信度0.06的值
            if scores.shape[0] == 0:
                # 说明锚点框与这一类的GT框不匹配,简介说明,不存在这一类的目标
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)  # 得到置信度大于阈值的那些锚点框
            # boxes=boxes* torch.Tensor([w, h, w, h]).expand_as(boxes).float().cuda()
            #####[c1这个类别对应的默认盒数目,4],c1这个类别对应的默认盒的坐标
            ids = nms(boxes, scores, 0.5, 200)###保留下来的索引号 #######################################################
            output[i, cl, :len(ids)] = torch.cat((scores[ids].unsqueeze(1),####为什么要unsqueeze(1)？
                                                  boxes[ids]), 1)  # [置信度,xmin,ymin,xmax,ymax]
    return output