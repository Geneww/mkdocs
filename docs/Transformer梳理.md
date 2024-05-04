# Transformer梳理

[论文原文](https://arxiv.org/pdf/1706.03762.pdf)  
Transformer 开创了继 MLP 、CNN和 RN 之后的第四大类模型。  
常见的时间序列任务采用的模型通常都是RNN系列，然而RNN系列模型的顺序计算方式带来了两个问题

1.  某个时间状态$h_t$，依赖于上一时间步状态$h_{t-1}$，导致模型**不能通过并行计算来加速**
2.  RNN系列的魔改模型比如GRU, LSTM，虽然**引入了门机制**(gate)，但是对**长时间依赖的问题缓解能力有限**，不能彻底解决

因此我们设计了一个全新的结构Transformer，通过Attention注意力机制，来对时间序列更好的建模。同时我们不需要像RNN那样顺序计算，从而能让模型更能充分发挥并行计算性能。

![Image](https://image.iokko.cn/file/be6facab1fd11d6690ec3.png)

## Transformer 使用encoder-decoder架构

## ScaledDotProductAttention

![Image](https://image.iokko.cn/file/49dcc4e21ce8761949feb.png)

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

- Q 代表 Query 矩阵
- K 代表 Key 矩阵
- V 代表 Value 矩阵
- dk 是一个缩放因子

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def get_attention_mask(seq):
        """seq: [batch_size, tgt_len]"""
        # batch_size个 tgt_len * tgt_len的mask矩阵
        # attn_shape = [seq.size(0), seq.size(1), seq.size(2)]
        attn_shape = seq.shape
        # np.triu 是生成一个 upper triangular matrix 上三角矩阵，k是相对于主对角线的偏移量
        # k=1意为不包含主对角线（从主对角线向上偏移1开始）
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 因为只有0、1所以用byte节省内存
        return subsequence_mask == 1  # return: [batch_size, n_head, tgt_len, tgt_len]

    def forward(self, q, k, v, mask=False):
        # scores = q @ k / np.sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(q.shape[-1])  # [batch, n_head, len_q, len_k]
        if mask:
            # 生成mask掩码
            attention_mask = self.get_attention_mask(scores)
            scores.masked_fill_(attention_mask, -1e9)
        return torch.matmul(self.dropout(self.softmax(scores)), v)
```

## MultiHeadAttention

<img src="https://image.iokko.cn/file/8c9ff442e9facb71dafdb.png" alt="Image" >

```python

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden=512, n_head=8, dropout=0.1, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_hidden = num_hidden
        self.num_head = n_head
        self.q_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.k_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.v_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.concat_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, queries, keys, values, mask=False):
        # 分多个头  [batch, seq_len, hidden_size] -> [batch, seq_len, head_size, d_k] -> [batch, head_size, seq_len, d_k]
        d_k = self.num_hidden // self.num_head
        q = self.q_mat(queries).view(queries.size(0), -1, self.num_head, d_k).transpose(1, 2)
        k = self.k_mat(keys).view(keys.size(0), -1, self.num_head, d_k).transpose(1, 2)
        v = self.v_mat(values).view(values.size(0), -1, self.num_head, d_k).transpose(1, 2)
        out = self.attention(q, k, v, mask)
        out = out.view(out.shape[0], -1, self.num_hidden)
        out = self.concat_mat(out)
        return out  # output: [batch_size, len_q, hidden_size]

```


* * *

## LayerNorm

- 这里主要的问题是在算均值和方差上面，对于batchnorm来说，会对上图中切出来的阶梯形的部分进行求解（只有这部分是有效值，其他地方因为是补零，所以其实没有太多作用），如果样本长度变化比较大的时候，每次做小批量的时候，算出来的均值和方差的抖动相对来说是比较大的
- 另外，在做预测的时候要把全局的均值和方差记录下来，这个全局的均值和方差如果碰到一个新的预测样本，如果碰到一个特别长的，因为在训练的时候没有见过这种长度的，那么在之前计算的均值和方差可能就不那么好用了。
- 相反，对于layernorm相对来说没有太多的问题，因为他是按照每个样本来进行均值和方差的计算，同时也不需要存下一个全局的均值和方差（不管样本的长短，均值和方差的计算都是以样本为单位的），这样的话相对来讲更稳定一些
```python
class AddNorm(nn.Module):
    """
    使用残差和归一化
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, x_attention_out):
        return self.layer_norm(self.dropout(x_attention_out) + x)
```

* * *

## FeedForward Networks

这个FFN模块比较简单，就是一个MLP，本质上全是两层全连接层加一个激活函数，这里使用的是relu

$$
X = Dense_1(X) \\
X = Relu(X) \\ 
Out = Dense_2(X)
$$

```python
class FeedForward(nn.Module):
    """
    前馈网络mlp
    """

    def __init__(self, d_model, dim_feedforward, activation):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        return self.net(x)
```

* * *

## Encoder
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation=nn.ReLU,
                 layer_norm_eps: float = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.add_norm_2 = AddNorm(d_model, dropout, layer_norm_eps)

    def forward(self, x):
        x_attention_out = self.attention(x, x, x)
        x_norm_out = self.add_norm_1(x, x_attention_out)
        feed_out = self.feed_forward(x_norm_out)
        x = self.add_norm_2(x_norm_out, feed_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x):
        for net in self.layers:
            x = net(x)
        return x
```
* * *
## Decoder
```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation=nn.ReLU,
                 layer_norm_eps: float = 1e-5):
        super(TransformerDecoderLayer, self).__init__()
        self.mask_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout, layer_norm_eps)
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.add_norm_3 = AddNorm(d_model, dropout, layer_norm_eps)

    def forward(self, x, encoder_out=None):
        out = self.mask_attention(x, x, x, True)
        norm_out = self.add_norm_1(x, out)
        if encoder_out is not None:
            out = self.attention(encoder_out, encoder_out, norm_out)
        else:
            out = self.attention(norm_out, norm_out, norm_out)
        norm_out = self.add_norm_2(norm_out, out)
        out = self.feed_forward(norm_out)
        norm_out = self.add_norm_3(norm_out, out)
        return norm_out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, x, encoder_out=None):
        for net in self.layers:
            x = net(x, encoder_out)
        return x

```
* * *
## Positional Encoding

​ attention无时序信息，只是计算加权和，与时序无关

​ 一句话顺序打乱，attention计算后输出值都一样

​ 位置编码为了解决词向量之间位置顺序问题

​ 你打我与我打你 通过位置编码计算出来结果是不一样的，通过位置编码计算后，attention的输入不一样了，所以输出也不一样了。

$$
PE(pos,2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE(pos,2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

```python
class PositionEncoder(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_token: int = 1000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_token * d_model 的矩阵，即1000 * 512
        # 1000是一个句子中最多的token数，512是一个token用多长的向量来表示，1000*512这个矩阵用于表示一个句子的信息
        self.pe = torch.zeros(1, max_token, d_model)
        pos = torch.arange(0, max_token, dtype=torch.float).unsqueeze(1)  # pos：[max_len,1],即[1000,1]
        # 先把括号内的分式求出来,pos是[1000,1],分母是[256],通过广播机制相乘后是[1000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        self.pe[:, :, 0::2] = torch.sin(div_term)
        self.pe[:, :, 1::2] = torch.cos(div_term)
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('PositionEncoding', self.pe)

    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 1000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # return: [batch_size, seq_len, d_model], 和输入的形状相同
```

## Embedding

- embedding module 的前向过程其实是一个索引（查表）的过程
    - 表的形式是一个 matrix（embedding.weight, learnable parameters）
        - matrix.shape: (v, h)
            - v：vocabulary size
            - h：hidden dimension
    - 具体索引的过程，是通过 one hot + 矩阵乘法的形式实现的；
    - input.shape: (b, s)
        - b：batch size
        - s：seq len
    - embedding(input)
        - (b, s) ==> (b, s, h)
        - (b, s) 和 (v, h) ==>? (b, s, h)
            - (b, s) 经过 one hot => (b, s, v)
            - (b, s, v) @ (v, h) ==> (b, s, h)