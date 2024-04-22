# DETR梳理

[DETR论文原文](https://arxiv.org/pdf/2005.12872.pdf)
detr是一个全新的目标检测架构，解决了端到端的问题，使用了Transformer架构

无论是proposal based方法、anchor based方法、non-anchor based方法，最后都会生成很多预测框，如何去除这些冗余的框就是NMS要做的事情





贡献：

1、把目标检测做成一个端到端的框架，

把之前特别依赖人的先验知识的部分删掉了（NMS部分、anchor），一旦把这两个部分拿掉之后，我们也不用费尽心思设计这种anchor，最后不会出现这么多框，不会用到NMS，也不会用到很多超参去调

detr提出：

2、新的目标函数，通过二分图匹配的方式，强制模型输出一组独一无二的预测（没有那么多冗余框，每个物体理想状态下就会生成一个框）

3、使用encoder-decoder的架构 

两个小贡献：

1、decoder还有另外一个输入learned object query，类似anchor的意思

（给定这些object query之后，detr就可以把learned object query和全局图像信息结合一起，通过不同的做注意力操作，从而让模型直接输出最后的一组预测框）

2、想法&&实效性：并行比串行更合适，并不是检测一个大物体前必须先检测一个小物体，或从左到右检测，我们希望越快越好

DETR的好处：

1、简单性：想法上简单，不需要一个特殊的library，只要硬件支持transformer或CNN，就一定支持detr

2、性能：在coco数据集上，detr和一个训练非常好的faster RCNN基线网络取得了差不多的效果，模型内存和速度也和faster RCNN差不多

3、想法好，解决了目标检测领域很多痛点，写作好

4、别的任务：全景分割任务上detr效果很好，detr能够非常简单拓展到其他任务上





https://image.iokko.cn/file/470817a4055f74637eccd.png