# coding=utf-8
import torch
from pycocotools import coco
import os
import numpy as np
import torch.utils.data as data
import cv2
import numpy as np
import json
from pycocotools.cocoeval import COCOeval
COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]
class COCO(data.Dataset):
    def __init__(self,data_dir,split, split_ratio=1.0,transforms=None,remove_empty=True):
        super(COCO, self).__init__()
        #self.num_classes = 80
        self.class_name = COCO_NAMES
        self.valid_ids = COCO_IDS
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}#字典{1：o}标签 0开始编号
        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]
        self.transfrom=transforms
        self.split = split
        self.data_dir = os.path.join(data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '%s2017' % split)
        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json')
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_%s2017.json' % split)

        print('==> initializing coco 2017 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)

        if remove_empty:
            self.images=list(self.coco.imgToAnns.keys())  #训练时，没有标注的图片删除
        else:
            self.images = self.coco.getImgIds()
        if 0 < split_ratio < 1:
            split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
            self.images = self.images[:split_size]

        self.num_samples = len(self.images)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations]) ##从0开始编号
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32) #xmin,ymin,w,h读取
        # if len(bboxes) == 0:
        #     bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
        #     labels = np.array([-1]) # 从0开始编码， 在计算loss的时候，match的是时候 gt虽然是坐标虽然是0但是它会匹配到一个默认盒，为了保留关系iou值会被设定为2 其对应的label会被+1所以设定为-1
        # #添加一步移除空盒子的操作？

        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxyd
        img=cv2.imread(img_path)#bgr np.array
        h,w,_=img.shape
        #bboxes 坐标归一化出错了
        # bboxes[::2]=bboxes[::2]/w ##xmin,xmax 归一化
        # bboxes[1::2]=bboxes[1::2]/h ##ymin, ymax,归一化
        bboxes[:,::2] = bboxes[:,::2] / w
        bboxes[:,1::2] = bboxes[:,1::2] / h
        #数据增强
        if self.transfrom is not None: #包含了resize
            img,bboxes,labels=self.transfrom(img,bboxes,labels)
            img=img[:,:,(2,1,0)]
        #batch >1时， bboxes的shape 和label的shape会不一样 修改Dataloader里面的collate_fn方法
        ##先使用SSD里面的策略吧
        return torch.from_numpy(img).permute((2,0,1)),torch.from_numpy(bboxes),torch.from_numpy(labels)
    def pull_item(self,index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])  ##从0开始编号
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)  # xmin,ymin,w,h读取
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array(
                [0])  # 从0开始编码， 在计算loss的时候，match的时候 gt坐标虽然是0但是它会匹配到一个默认盒，为了保留关系iou值会被设定为2 其对应的label会被+1所以设定为-1
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        img = cv2.imread(img_path)  # bgr np.array
        h, w, _ = img.shape
        # bboxes的归一化出错了哦。。
        # bboxes[::2] = bboxes[::2] / w  ##xmin,xmax 归一化
        # bboxes[1::2] = bboxes[1::2] / h  ##ymin, ymax,归一化
        bboxes[:,::2] = bboxes[:,::2] / w
        bboxes[:,1::2] = bboxes[:,1::2] / h
        # 数据增强
        if self.transfrom is not None:  # 包含了resize
            img, bboxes, labels = self.transfrom(img, bboxes, labels)
        # batch >1时， bboxes的shape 和label的shape会不一样 修改Dataloader里面的collate_fn方法
        ##先使用SSD里面的策略吧
        return img, bboxes, labels,h,w,img_id

    def __len__(self):
        return self.num_samples

    def run_eval(self, detections, save_dir=None):
        if save_dir is not None:
            print('jinlaile')
            result_json = os.path.join(save_dir, "results.json")
            json.dump(detections, open(result_json, "w"))
        print(type(detections))
        coco_dets = self.coco.loadRes(detections)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

def collate_fn(batch):
    return tuple(zip(*batch))
if __name__=='__main__':
    import argparse
    from torch.utils.data import DataLoader
    from augmentations import SSDAugmentation
    parser=argparse.ArgumentParser(description='read coco dataset')
    parser.add_argument('--data_dir',type=str,default='E:\coco')
    parser.add_argument('--split',type=str,default='train',help='dataset name for read')
    cfg=parser.parse_args()
    dataset=COCO(data_dir=cfg.data_dir,split=cfg.split,transforms=SSDAugmentation())
    dataloder=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=collate_fn) ##有__iter__方法
    itdata=dataloder.__iter__()
    img,bboxes,label=itdata.__next__()
    print(img)
    print(bboxes)