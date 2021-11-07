# coding:utf-8
from read_coco import COCO
from augmentations import SSDAugmentation
import argparse
from torch.utils.data import DataLoader
from net import build_SSD
import torch
from utils import BaseTransform
from engine import train_
parser=argparse.ArgumentParser(description='SSD for COCO')
parser.add_argument('--data_dir',type=str,default='E:\coco',help='COCO root')
parser.add_argument('--split',type=str,default='train',help='types of dataset')
parser.add_argument('--use_cuda',type=bool,default=True)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--threshold',type=float,default=0.5,help='use for chose neg samples')
parser.add_argument('--variance',type=list,default=[0.1,0.2])
parser.add_argument('--num_classes',type=int,default=81)
parser.add_argument('--negpos_ratio',type=int,default=3)
parser.add_argument('--init_lr',type=float,default=0.001)
parser.add_argument('--epoch',type=int,default=140)
parser.add_argument('--step_print',type=int,default=100)
parser.add_argument('--lr_step',type=list,default=[110,130])
parser.add_argument('--save_path',type=str,default='./result/',help='path for save model')
parser.add_argument('--val_epoch',type=int,default=5,help='every x epoch do val')
parser.add_argument('--start_epoch',type=int,default=86)
cfg=parser.parse_args()

def collate_fn(batch):
    return tuple(zip(*batch))
if __name__=='__main__':

    dataset=COCO(cfg.data_dir,cfg.split, split_ratio=1.0,transforms=SSDAugmentation())
    dataloader=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=2)
    dataset_test=COCO(cfg.data_dir,'val', split_ratio=1.0,transforms=BaseTransform(300),remove_empty=False)
    net=build_SSD()

    # from collections import  OrderedDict
    # new_state_dict=OrderedDict()
    # for k,v in state_dict.items():
    #     name=k[7:]
    #     new_state_dict[name]=v
    # net.load_state_dict(new_state_dict)

    if cfg.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
    load_path = './result/ssd_coco_90epoch_85.pth'
    state_dict = torch.load(load_path)
    net.load_state_dict(state_dict)
    print(net)
    train_(net,dataloader,dataset_test,cfg)



