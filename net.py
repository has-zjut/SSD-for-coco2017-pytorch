# coding:utf-8
import torch.nn as nn
import torch
from l2norm import L2Norm
import torch.nn.functional as F
from priors_box import Priors_box

cfg_coco = {
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
class SSD(nn.Module):
    def __init__(self,vgg16,extra_layer,loc_layer,conf_layer,num_class):
        super(SSD, self).__init__()

        self.num_class=num_class
        self.backbone=nn.ModuleList(vgg16)
        self.L2norm=L2Norm(512,20)
        self.extra=nn.ModuleList(extra_layer)
        self.loc_layer=nn.ModuleList(loc_layer)
        self.conf_layer=nn.ModuleList(conf_layer)
        self.priors=Priors_box(cfg_coco).forward()
    def forward(self,x):

        feature_fordetection=[]
        loc=[]
        conf=[]
        for k in range(23):
            x=self.backbone[k](x)
        s=self.L2norm(x)
        feature_fordetection.append(s)
        for k in range(23,len(self.backbone)):
            x=self.backbone[k](x)
        feature_fordetection.append(x)

        #辅助特征层
        for k,v in enumerate(self.extra):
            x=F.relu(v(x),inplace=True)
            if k%2==1:
                feature_fordetection.append(x)

        for i,j,k in zip(feature_fordetection,
                         self.loc_layer,self.conf_layer):
            loc.append(j(i).permute((0,2,3,1)).contiguous())
            conf.append(k(i).permute((0,2,3,1)).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        output=(
            loc.view(loc.size(0),-1,4),
            conf.view(conf.size(0),-1,self.num_class),
            self.priors
        )
        return output


def vgg16(cfg,init_chanels,batchnorm):
    layers=[]
    inchanels=init_chanels
    for v in cfg:
        if v=='M':
            #layers.append([nn.MaxPool2d(kernel_size=2,stride=2)])
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        elif v=='C':
            #layers.append([nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)])
            layers +=[nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)]
        else:
            conv2d=nn.Conv2d(in_channels=inchanels,out_channels=v,kernel_size=3,padding=1)
            if batchnorm:
                #layers.append([conv2d,nn.BatchNorm2d(v),nn.ReLU()])
                layers +=[conv2d,nn.BatchNorm2d(v),nn.ReLU()]
            else:
                #layers.append([conv2d,nn.ReLU()])
                layers+=[conv2d,nn.ReLU()]
            inchanels=v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def extra_layer(cfg,inchanels,batchnorm=None):
    inchanels=inchanels
    layers=[]
    flag=False
    for k,v in enumerate(cfg):
        if inchanels !='S':
            if v=='S':
                # layers.append([nn.Conv2d(in_channels=inchanels,out_channels=cfg[k+1],
                #                          kernel_size=(1,3)[flag],stride=2,padding=1)
                #                ])
                layers+=[nn.Conv2d(in_channels=inchanels,out_channels=cfg[k+1],
                                         kernel_size=(1,3)[flag],stride=2,padding=1)
                               ]
            else:
                # layers.append([nn.Conv2d(in_channels=inchanels,out_channels=v,
                #                          kernel_size=(1,3)[flag],stride=1)])
                layers+=[nn.Conv2d(in_channels=inchanels,out_channels=v,
                                         kernel_size=(1,3)[flag],stride=1)]
            flag = not flag
        inchanels=v
    return layers
def local_conf_layer(cfg,vgg,extra,num_class=21):
    local_layer=[]
    conf_layer=[]
    vgg_for_detection=[21,-2]
    for k,v in enumerate(vgg_for_detection):
        local_layer.append(nn.Conv2d(in_channels=vgg[v].out_channels,out_channels=cfg[k]*4,kernel_size=3,padding=1)
                           )
        conf_layer.append(nn.Conv2d(in_channels=vgg[v].out_channels,out_channels=cfg[k]*num_class,kernel_size=3,padding=1))
    for k,v in enumerate(extra[1::2],2):#后面的2表示k重2开始计数
        local_layer.append(nn.Conv2d(in_channels=v.out_channels,out_channels=cfg[k]*4,kernel_size=3,padding=1))
        conf_layer.append(nn.Conv2d(in_channels=v.out_channels,out_channels=cfg[k]*num_class,kernel_size=3,padding=1))
    return local_layer,conf_layer


cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]
cfg_extra=[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256] ##s的位置是stride=2的位置对应的channel是s后面的数

cfg_detect=[4, 6, 6, 6, 4, 4]

def build_SSD():
    vgg=vgg16(cfg,3,batchnorm=False)
    extra=extra_layer(cfg_extra,1024)
    local_layer,conf_layer=local_conf_layer(cfg_detect,vgg,extra,num_class=81)
    return SSD(vgg,extra,local_layer,conf_layer,81)

if __name__=='__main__':
    a=torch.rand((1,3,300,300))
    net=build_SSD()
    print(net)
    b=net(a)
    print(b[0].shape)
    print(b[1].shape)
    print(b[2].shape)





