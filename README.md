# SSD-for-coco2017-pytorch
Single Shot Multibox Detector;  There is no deed to load any pre-training weight model;  It is  trained for 140 epoch on COCO2017 for this project; map=16.6;  it doesn't get enough training; training more epoch, you will get a better result;

SSD是比较早的检测方法了，在跟踪以及尝试复现近些年的目标检测方法中，SSD不需要加载预训练模型，就可以取得和论文中相似的结果,这个项目在VOC数据集上以voc2007trainval 以及voc2012trainvoc 为训练集，在voc2007 test上进行测试，map=76.8; 一共训练了390个epoch; 卡是2060super 8g显存；batchsize=8(图像resize到300*300);  SGD优化器： lr=0.001 for 270个epoch, lr=0.0001 for 60 epoch  lr=0.00001 for 60 epoch; 这是在voc数据集上的实验  SSD+voc的代码可以搜索到很多；
本次上传的代码是在coco2017数据集上的； bathsize=8（图像resize 到300*300）;训练了140个epoch; lr=0.001,0.0001,0.00001 忘记各多少个epoch了， lr=0.001多训练一些epoch;我训练的结果是coco2017 val 上map=16.6;  要训练voc的可以自己修改readset 数据增强的方法都不变; 

希望能为做方法的一些伙伴提供一些帮助;
代码有参考别人的代码：侵权告知，即删；
