# coding:utf-8
'''
loss包含2个部分： bboxes loss, class_loss
predict=([b,8732,4],[b,8732,81],)
'''
import torch.nn as nn
import torch
from match import match
import torch.nn.functional as F
def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x=x
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp((x-x_max).float()), 1, keepdim=True)) + x_max
    # return torch.log(torch.sum(torch.exp(x), 1, keepdim=True))
class multiboxloss(nn.Module):
    def __init__(self,threshold,variance,num_classes,negpos_ratio,use_gpu=True):
        super(multiboxloss, self).__init__()
        self.threshould=threshold
        self.variance=variance
        self.use_gpu=use_gpu
        self.num_classes=num_classes
        self.negpos_ratio=negpos_ratio

    def forward(self,predict,bboxes,labels):
        local,conf,priors=predict
        batch=local.size(0)
        num_priors=priors.size(0)
        local_t=torch.zeros((batch,num_priors,4)) ##匹配过程中保存匹配的box坐标
        conf_t=torch.zeros((batch,num_priors)).long()

        for idx in range(batch):
            truth_bboxes=bboxes[idx].cuda()
            truth_labels=labels[idx].cuda()
            default=priors.data.cuda() #默认盒子 shape[8732,4]
            # truth_boxes和default bbox 匹配
            match(self.threshould,truth_bboxes,default,self.variance,truth_labels,local_t,conf_t,idx)
        #local_t,conf_t都找到了
        if self.use_gpu:
            local_t=local_t.cuda()
            conf_t=conf_t.cuda()
        pos = conf_t > 0 #正样本的掩膜 shape=[batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)##正样本总数目
        pos_index=pos.unsqueeze(-1).expand_as(local_t) #
        ##计算local损失只统计正样本损失
        local_t=local_t[pos_index].view(-1,4)
        local_p=local[pos_index].view(-1,4) #预测 位置误差之和正样本有关
        loss_l = F.smooth_l1_loss(local_p, local_t, reduction='sum')
        batch_conf = conf.view(-1, self.num_classes)# 预测

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(batch, -1)#[b,num_priors]
        loss_c[pos] = 0 ###正样本置0，此时里面不是0的位置对应的就是hard example
        _, loss_idx = loss_c.sort(1, descending=True)  ###hard样本的索引
        _, idx_rank = loss_idx.sort(1)  ###按索引号大小排序
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  ###正负样数目，最小值是3倍的正样本数目,最大值是num_priors-1
        #torch.clamp默认min=-inf, max=inf
        neg = idx_rank < num_neg.expand_as(idx_rank)  ###掩码。值为TRUE 和False，<的位置为TRUE
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf)  ###正样本掩膜 conf_data.shape=[conf.size(0), -1, self.num_classes)]
        neg_idx = neg.unsqueeze(2).expand_as(conf)  ##hard example样本掩膜
        conf_p = conf[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)  ###获得正样本和hard example的预测类别自信度,
        targets_weighted = conf_t[(pos + neg).gt(0)]  ####shape=[正样本数目+hard example数目,1] ###target类别标签
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')  ##targets_weighted该样本的标签，与21个值计算交叉熵
        ####### 标签*log预测的概率,,交叉熵自行有softmax的作用,,交叉熵是表征真是样本标签和预测概率之间的差值
        ###计算了正样本与每一类别的交叉熵,包括背景类别
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()  ###匹配到gtbox的默认盒数目
        loss_l /= N  ####位置误差,不包含负样本()
        loss_c /= N  ####类别误差，(包含背景类别)
        # return loss_l, loss_c
        return loss_l, loss_c  ###应该返回这个？










