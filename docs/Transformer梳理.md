# Transformer梳理
[论文原文](https://arxiv.org/pdf/1706.03762.pdf)
Transformer 开创了继 MLP 、CNN和 RN 之后的第四大类模型。
常见的时间序列任务采用的模型通常都是RNN系列，然而RNN系列模型的顺序计算方式带来了两个问题

1. 某个时间状态$h_t$，依赖于上一时间步状态$h_{t-1}$，导致模型**不能通过并行计算来加速**
2. RNN系列的魔改模型比如GRU, LSTM，虽然**引入了门机制**(gate)，但是对**长时间依赖的问题缓解能力有限**，不能彻底解决

因此我们设计了一个全新的结构Transformer，通过Attention注意力机制，来对时间序列更好的建模。同时我们不需要像RNN那样顺序计算，从而能让模型更能充分发挥并行计算性能。




<img src="https://image.iokko.cn/file/be6facab1fd11d6690ec3.png" alt="Image" style="float: left;">
<div class="clear"></div>

## 1.Positional Encoding

​	attention无时序信息，只是计算加权和，与时序无关 

​	一句话顺序打乱，attention计算后输出值都一样

​	位置编码为了解决词向量之间位置顺序问题	

​	你打我与我打你 通过位置编码计算出来结果是不一样的，通过位置编码计算后，attention的输入不一样了，所以输出也不一样了。

## 2.Transformer 使用encoder-decoder架构

### 2.1 encoder
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
- Q 代表 Query 矩阵
- K 代表 Key 矩阵
- V 代表 Value 矩阵
- dk 是一个缩放因子
### 2.1 decoder