# VIT梳理

[论文原文](https://arxiv.org/pdf/2010.11929.pdf)  
        Vision Transformer不仅是在视觉领域挖了一个大坑，在多模态领域其实也挖了一个大坑，因为它打破了CV和NLP在模型上的壁垒。视觉领域，我们如果想用Transformer的话，首先第一个要解决的问题就是如何把一个2d的图片变成一个1d的序列，或者变成一个集合，最直观的方法就是，我们把每个像素点当成这边的元素，然后直接把2d的图片拉直，把它扔到Transformer里，然后做自己跟自己学习就好，但是想法很美好，实现起来复杂度太高。一般来说，在视觉里面我们训练分类任务的时候，这个图片的输入大小大概是224*224，这么大，但如果把图片里的每个元素点都直接当成这里的元素来看待，那其实这个序列长度就不是512，序列长度就是224*224=50176，因为我们一共有这么多像素点，一个有50000个像素点，所以这个就是序列的长度，那这个序列长度50000相当于是bert序列长度512*100倍，所以这复杂度是相当可怕的，然后这还只是分类任务，他的图片就224*224，那对于检测或者分割这种任务呢，现在很多模型的输入都已经变成600*600或者800*800或者更大，那这个计算复杂度便更是高不可攀了。  
        Vision Transformer通过把一个图片打成了很多patch合并patch解决计算量。

[![](https://github.com/lucidrains/vit-pytorch/raw/main/images/vit.gif)](https://github.com/lucidrains/vit-pytorch/blob/main/images/vit.gif)

## 1.PatchEmbedding

Vision Transformer把一个图片打成了很多patch，每一个patch就是16\*16

假如说这个图片的大小是224*224，那么我们刚开始算这个sequence length是224*224=50176，现在如果我们换成patch，相当于一个patch是一个元素，那有效的长度是多少呢？现在这个宽度就变成了224/16=14，高度也是14，这样的话，最后的序列长度就变成了14\*14=196，也就是现在这张图片就只有196个元素，196对于普通的Transformer来说，还是可以接受的序列长度

然后我们把这里的每一个patch，当做一个元素，通过一个这个全连接层就会得到一个linear embedding，而这些就会当做输入传给Transformer，这个时候，我们可以很容易的看出来，一个图片就已经变成了一个一个的这样的图片块了，然后我们可以把这些图片块，当成是像NLP里的那些单词，这一个句子有多少词，那就相当于是一个图片，现在有多少个patch，这就是题目的意义，一个图片等价于很多16\*16的单词

输入图片大小为224\*224\*3(用X表示)，每个patch的大小为16\*16，所以patch个数为(224/16)^2=196，每一个图像块的维度就是16\*16\*3=768

Linear Projection线性投射层就是全连接层(用E表示)，该维度是768\*768，前面的768是不变的，是算出来的16\*16\*3=768，后面的768是可以变化的

X\*E得到的还是196\*768(196个token，每个token维度都是768)，之后在拼接上1个cls token(1\*768)，再添加位置信息(是sum而不是concat)，所以最后输入到Transformer Encoder的还是197\*768

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """使用卷积代替切片"""

    def __init__(self, images_size=(224, 224, 3), embedding_dim=768, patch_size=16, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        height, weight, in_channels = images_size
        assert height % patch_size == 0 and weight % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patcher = (height // patch_size) * (weight // patch_size)
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.cls_token = nn.Parameter(torch.rand(size=(1, 1, embedding_dim)), requires_grad=True)
        # patch * patch + 1 (cls token)
        self.position_embedding = nn.Parameter(torch.rand(size=(1, num_patcher + 1, embedding_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: [n, c, h, w] tensor
        :return: [batch, 197, 768] tensor
        """
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1, 1, 768] -> [batch, 1, 768]
        x = self.patcher(x)  # [batch, 768, 14, 14]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2,
                                                       1)  # [batch, 768, 14, 14] -> [batch, 768, 196] -> [batch, 196, 768]

        x = torch.cat([x, cls_token], dim=1)  # [64, 197, 768]
        x += self.position_embedding
        x = self.dropout(x)
        return x
```

## 2.TransformerEncoder

这里只使用的Transformer的Encoder模块

```python
import copy
import torch
import torch.nn as nn

import numpy as np

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

## 3.MLP

```python
class MLP(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_classes: int = 10):
        super(MLP, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        return self.linear(self.layer_norm(x))
```

## 4.VIT

```python
class Vit(nn.Module):
    def __init__(self, images_size: tuple = (224, 224, 3), num_classes: int = 10, embedding_dim: int = 768,
                 patch_size: int = 16, dropout: float = 0.1, n_head: int = 12, num_encoders: int = 6):
        super(Vit, self).__init__()
        assert embedding_dim % n_head == 0, "embedding_dim or n_head param error"
        self.patch_embedding = PatchEmbedding(images_size, embedding_dim, patch_size, dropout)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, n_head=n_head, dropout=dropout, dim_feedforward=embedding_dim*4)
        self.encoder_layers = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoders)
        self.mlp = MLP(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x):
        """
        :param x: [n, c, h, w] tensor
        :return: [n, sql_len, num_classes]
        """
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)  # [batch, 197, 768] -> [768/12=64 8head 197x64] -> [batch, 197, 768]
        x = self.mlp(x)  # [batch, 197, num_classes]
        return x
```