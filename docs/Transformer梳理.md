# Transformer梳理
[论文原文](https://arxiv.org/pdf/1706.03762.pdf)
Transformer 开创了继 MLP 、CNN和 RN 之后的第四大类模型。
常见的时间序列任务采用的模型通常都是RNN系列，然而RNN系列模型的顺序计算方式带来了两个问题

1. 某个时间状态$h_t$，依赖于上一时间步状态$h_{t-1}$，导致模型**不能通过并行计算来加速**
2. RNN系列的魔改模型比如GRU, LSTM，虽然**引入了门机制**(gate)，但是对**长时间依赖的问题缓解能力有限**，不能彻底解决

因此我们设计了一个全新的结构Transformer，通过Attention注意力机制，来对时间序列更好的建模。同时我们不需要像RNN那样顺序计算，从而能让模型更能充分发挥并行计算性能。




<img src="https://image.iokko.cn/file/be6facab1fd11d6690ec3.png" alt="Image" style="float: left;">
<div class="clear"></div>



## Transformer 使用encoder-decoder架构

## 



## ScaledDotProductAttention

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

- Q 代表 Query 矩阵
- K 代表 Key 矩阵
- V 代表 Value 矩阵
- dk 是一个缩放因子

## MultiHeadAttention





## LayerNorm

- 这里主要的问题是在算均值和方差上面，对于batchnorm来说，会对上图中切出来的阶梯形的部分进行求解（只有这部分是有效值，其他地方因为是补零，所以其实没有太多作用），如果样本长度变化比较大的时候，每次做小批量的时候，算出来的均值和方差的抖动相对来说是比较大的
- 另外，在做预测的时候要把全局的均值和方差记录下来，这个全局的均值和方差如果碰到一个新的预测样本，如果碰到一个特别长的，因为在训练的时候没有见过这种长度的，那么在之前计算的均值和方差可能就不那么好用了。
- 相反，对于layernorm相对来说没有太多的问题，因为他是按照每个样本来进行均值和方差的计算，同时也不需要存下一个全局的均值和方差（不管样本的长短，均值和方差的计算都是以样本为单位的），这样的话相对来讲更稳定一些

## FeedForward Networks



## Encoder



## Decoder



## Positional Encoding

​	attention无时序信息，只是计算加权和，与时序无关 

​	一句话顺序打乱，attention计算后输出值都一样

​	位置编码为了解决词向量之间位置顺序问题	

​	你打我与我打你 通过位置编码计算出来结果是不一样的，通过位置编码计算后，attention的输入不一样了，所以输出也不一样了。
$$
PE(pos,2i) = sin(\frac{pos}{1000^{\frac{2i}{d_{model}}}}) \\
PE(pos,2i+1) = cos(\frac{pos}{1000^{\frac{2i}{d_{model}}}})
$$


```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # dropout是原文的0.1，max_len原文没找到
        ''' 位置编码
        max_len是假设的一个句子最多包含5000个token'''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_len * d_model 的矩阵，即5000 * 512
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # pos：[max_len,1],即[5000,1]
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 一个句子要做一次pe，一个batch中会有多个句子，所以增加一维用来和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0)  # [5000,512] -> [1,5000,512]
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # return: [batch_size, seq_len, d_model], 和输入的形状相同
```

## Embedding