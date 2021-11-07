# coding:utf-8
'''
truth_bboxes 和priors匹配，挖掘正负样本
'''
from compute_iou import jaccard
import torch
def encode(matched, priors, variances):###matched是gtbox的坐标
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form###(xmin,ymin,xmax,ymax)
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    ###matched里面是gtbox的信息，len(matched)=num_perios
    priors=priors.float()
    matched=matched.float()
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]###匹配上的gtbox与每个默认盒的中心坐标偏移 本来就是归一化的坐标了呀
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])###=(g_cxcy/priors[:, 2:])*10  中心偏差/默认盒w,h 归一化  *10，放大误差加快损失下降？
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]###gtboxw,h/默认盒w.h   匹配上的gtbox的w,h/默认盒的宽高
    g_wh = torch.log(g_wh) / variances[1]### / variances[1]=*5 放大偏差
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]##decoder对应原文公式（2）

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax  后面的1表示按行拼接(xmin,ymin,xmax,ymax)
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # 计算每个truthsboxe与所有priors的iou
    overlaps=jaccard(truths,
                     point_form(priors)) #shape=[len(truths),len(pro=iors)]
    #找到iou最大的值和对应的索引
    best_prior_iou, best_prior_idx=overlaps.max(1,keepdim=True) ##最后用于priors的索引 shape=[len(truth)]

    best_target_iou,best_target_idx=overlaps.max(0,keepdim=True) #shape=[len(priors)]
    # 匹配额最终目的是选priors 从8732中选
    best_prior_iou.squeeze_(1)#又压缩为什么要keepdim呢
    # 每个 target 对应的最好的默认盒索引
    best_prior_idx.squeeze_(1)
    best_target_iou.squeeze_(0)
    #每个默认盒匹配上的gtbox的索引
    best_target_idx.squeeze_(0)
    # gtbox匹配上的最好的默认和肯定要留下作为正样本：
    # 1：best_prior_idx为gtbox对应的最好的默认盒  该种盒子与gtbox匹配时手动设定其iou为2，确保其能被保留
    best_target_iou.index_fill_(0, best_prior_idx, 2)

    #2： 修改索引， best_prior_idx为gtbox对应的最好的默认盒 该种盒子与gtbox匹配后，确保其匹配到的对应的gtbox的索引保持不变
    for j in range(best_prior_idx.size(0)):  ###遍历所有的与每一个gtbox匹配最好的默认盒   j为第j个gtbox
        best_target_idx[best_prior_idx[j]] = j

    #best_target_idx 为每个默认盒对应的gtbox的索引
    matches = truths[best_target_idx] ##shape=[len(8732,4)] #默认了每个默认盒都匹配到了gtbox 不太合理? 有bug
    conf=labels[best_target_idx]+1 #从0编的号
    conf[best_target_iou<threshold]=0  ##背景 #8732个全训练 conf可以解决上面bug的问题，loc呢？
    #只解决conf应该就可以了，因为最终的预测先看conf ,conf大于阈值了再看坐标

    loc=encode(matches,priors,variances)
    loc_t[idx]=loc
    conf_t[idx]=conf











