# coding:utf-8
from itertools import product
from math import sqrt
import torch
coco = {
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

class Priors_box(object):
    def __init__(self,cfg):
        super(Priors_box, self).__init__()
        self.feature_maps=cfg['feature_maps']
        self.image_size=cfg['min_dim']
        self.steps=cfg['steps']
        self.min_size=cfg['min_sizes']
        self.max_size=cfg['max_sizes']
        self.aspect_ratios=cfg['aspect_ratios']
        self.clip=cfg['clip']

    def forward(self):
        mean=[]
        for k,f in enumerate(self.feature_maps):
            # 建立坐标
            for i,j in product(range(f),repeat=2):
                # 尺寸映射到对应的feature mapsde size
                f_k=self.image_size/self.steps[k]
                # 确认默认盒的坐标
                center_x=(i+0.5)/f_k
                center_y=(j+0.5)/f_k
                s_k=self.min_size[k]/self.image_size
                #mean.append([center_x,center_y,s_k,s_k]) #最基础默认盒
                mean+=[center_x,center_y,s_k,s_k]
                s_k_prime = sqrt(s_k * (self.max_size[k] / self.image_size))
                #mean.append([center_x,center_y,s_k_prime,s_k_prime])
                mean+=[center_x,center_y,s_k_prime,s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [center_x, center_y, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [center_x, center_y, s_k / sqrt(ar), s_k * sqrt(ar)]

        output=torch.Tensor(mean).view(-1,4)
        if self.clip:
            output=torch.clamp(output,min=0,max=1)
        return output

if __name__=='__main__':
    coco = {
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
    priors=Priors_box(coco).forward()
    print(priors)









