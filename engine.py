# coding:utf-8
from loss_ssd import multiboxloss
import torch
import time
import datetime
from utils import predict
import numpy as np
import os

def train_(net, dataloader, test_dataset, cfg):

    criterion = multiboxloss(threshold=cfg.threshold, variance=cfg.variance, num_classes=cfg.num_classes,
                             negpos_ratio=cfg.negpos_ratio)
    # 动态调整学习率
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.init_lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)
    for epoch in range(cfg.start_epoch,cfg.epoch+1):
        net.train()
        running_loss = 0
        tic = time.time()
        for i, data in enumerate(dataloader):
            img, bboxes, labels = data
            img = torch.cat(img, 0).view((-1, 3, 300, 300))
            img = img.cuda()
            predict = net(img)
            loss_l, loss_c = criterion(predict, bboxes, labels)
            loss = loss_l + loss_c
            loss = loss.cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss

            if i % cfg.step_print == 0:
                duraction = time.time() - tic
                tic = time.time()
                str1 = '{:d}/{:d}-{:d}/{:d} '.format(epoch, cfg.epoch, i, len(dataloader))
                str2 = 'loss:{} '.format(running_loss / cfg.step_print)
                str3 = '{:2f}samples/sec'.format(cfg.batch_size * cfg.step_print / duraction)
                str4='lr:{}'.format(lr_scheduler.get_lr())
                date = datetime.datetime.now()
                print(str1 + str2 + str3+str4, date)
                running_loss = 0
        if epoch %cfg.val_epoch==0:
            val(net, test_dataset)
            if not os.path.exists(cfg.save_path):
                os.mkdir(cfg.save_path)
            save_path = cfg.save_path+'ssd_coco_90epoch_'+str(epoch)+'.pth'
            torch.save(net.state_dict(), save_path)
        lr_scheduler.step(epoch)  # epoch什么时候开始调整学习率


def val(net, dataset):
    COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90]
    # 创建一个字典，根据标签索引id
    labels = [i for i in range(1, 81)]
    d = {i: j for i, j in zip(labels, COCO_IDS)}  #
    net.eval()
    torch.cuda.empty_cache()
    num_images = len(dataset)
    results = []
    with torch.no_grad():
        for i in range(num_images):
            img, bboxes, labels, h, w, img_id = dataset.pull_item(i)
            detections = predict(net, img).data
            for j in range(1, detections.shape[1]):  ##标签 1，81
                dets = detections[0, j, :]  ####第j个类别的信息   shape=[200,5]
                mask = dets[:, 0].gt(0.01).expand(5, dets.size(0)).t()  ###只保留自信度》0的框
                dets = torch.masked_select(dets, mask).view(-1, 5)  ######[,5]
                # print(j,dets.shape)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]  ##boxes的shape不一定是1
                boxes[:, 0] = boxes[:, 0] * w
                boxes[:, 2] = boxes[:, 2] * w
                boxes[:, 1] = boxes[:, 1] * h
                boxes[:, 3] = boxes[:, 3] * h ##coco数据集的检测是xmin,ymin,w,h 要转换！
                boxes[:, 2] -= boxes[:, 0]  # w
                boxes[:, 3] -= boxes[:, 1]  # h
                scores = dets[:, 0].cpu().numpy()  # type,torch.tensor
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32,  # numpy
                                                                                          copy=False)  ####[num_bboxes,5:坐标自信度】坐标不是归一化的坐标
                for box in cls_dets:
                    detection = {'image_id': int(img_id), "category_id": int(d[j]), # int(d[j])
                                 'bbox': list(map(lambda x: float("{:.2f}".format(x)), box[0:4])),
                                 'score': float('{:.2f}'.format(box[4])) ## list(np.round(box[:4], 2))?
                                 }
                    results.append(detection)
            print('start val...%d/%d'%(i+1,num_images))
        eval_result = dataset.run_eval(results, save_dir=None)
        print(eval_result)

if __name__=='__main__':
    from train import cfg
    from read_coco import COCO
    from net import build_SSD
    import torch
    from utils import BaseTransform
    dataset_test = COCO(cfg.data_dir, 'val', split_ratio=1.0, transforms=BaseTransform(300),remove_empty=False)
    net = build_SSD()

    if cfg.use_cuda:
        net = torch.nn.DataParallel(net).cuda()

    load_path='./ssd_coco_90epoch_100.pth'
    state_dict=torch.load(load_path)
    net.load_state_dict(state_dict)
    result=val(net,dataset_test)
    print((result))