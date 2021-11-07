# -*- coding: UTF-8 -*-
import torch
def intersect(box_a, box_b):###交集
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    box_a=box_a.float()
    box_b=box_b.float()
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)###分段函数，小于0取0，之间取原来值，大于取max_xy - min_xy
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # 计算 A ∩ B
    box_a=box_a.float()
    box_b=box_b.float()
    inter = intersect(box_a, box_b)###交集
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 计算 A ∪ B
    union = area_a + area_b - inter
    return inter / union  # [A,B]###iou=交集/并集

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    priors=priors.cuda()
    loc=loc.cuda()
    # 对应上面的四个解码公式
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],###中心坐标
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)####宽高
    # 将(x_center, y_center, w, h)转换为(xmin, ymin, xmax, ymax)
    boxes[:, :2] -= boxes[:, 2:] / 2###中心坐标-宽高的一半
    boxes[:, 2:] += boxes[:, :2]###(xmin, ymin)+（w,h）= xmax, ymax
    return boxes

def nms(boxes, scores, overlap=0.5, top_k=200):
    '''
    进行nms操作
    :param boxes: 模型预测的锚点框的坐标
    :param scores: 模型预测的锚点框对应某一类的置信度
    :param overlap:nms阈值
    :param top_k:选取前top_k个预测框进行操作
    :return:预测框的index
    '''
    keep = torch.zeros(scores.shape[0])
    if boxes.numel() == 0:  # numel()返回tensor里面所有元素的个数
        return keep
    _, idx = scores.sort(0)  # 升序排序
    idx = idx[-top_k:]  # 取得最大的top_k个置信度对应的index

    keep = []  # 记录最大最终锚点框的index
    while idx.numel() > 0:
        i = idx[-1]  # 取出置信度最大的锚点框的index
        keep.append(i)
        idx = idx[:-1]
        if idx.numel() == 0:
            break
        IOU = jaccard(boxes[i].unsqueeze(0), boxes[idx])  # 计算这个锚点框与其余锚点框的iou shape=[len（boxes[i].unsqueeze(0)），len(boxes[idx])]其它盒子与idx这个盒子的iou
        mask = IOU.le(overlap).squeeze(0)###比较IOU是不是小于等于overlap，是的话=True
        idx = idx[mask]  # 排除大于阈值的锚点框
    return torch.tensor(keep)