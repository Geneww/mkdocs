# Swin Transformer梳理

[Swin Transformer论文原文](https://arxiv.org/pdf/2103.14030)  
Swin Transformer是ICCV 2021的最佳论文，它之所以有这么大的影响力，是因为在ViT之后，swin transformer凭借在一系列视觉任务上的出色表现，进一步证明了transformer是可以在视觉领域广泛应用的。此外，作者团队也发布了一系列基于swin transformer的工作，比如自监督版本的MoBY、视频领域的video-swin-transformer、应用swin思想的MLP版本Swin MLP以及半监督目标检测Soft Teacher、基于掩码自监督学习的SimMIM等，再次证明了swin transformer应用的广泛性和有效性。
![Swin Transformer  patch merging](https://image.iokko.cn/file/c149b7852def6561d8f81.png)  
目前Transformer应用到图像领域主要有两大挑战：

- 视觉实体变化大，在不同场景下视觉Transformer性能未必很好
- 图像分辨率高，像素点多，Transformer基于全局自注意力的计算导致计算量较大

针对上述两个问题，我们提出了一种**包含滑窗操作，具有层级设计**的Swin Transformer。

其中滑窗操作包括**不重叠的local window，和重叠的cross-window**。将注意力计算限制在一个窗口中，**一方面能引入CNN卷积操作的局部性，另一方面能节省计算量**。


## Swin Transformer的模型架构

![模型架构1](https://image.iokko.cn/file/718eccac5f6a858015590.png)  
![模型架构2](https://image.iokko.cn/file/5db63bdc94ec9cefd8d97.png)

根据模型总览图看一下模型的前向过程：

## Patch Partition

假设说有一张224x224x3（ImageNet 标准尺寸）的输入图片  
第一步就是像 ViT 那样把图片打成 patch，在 Swin Transformer 这篇论文里，它的 patch size 是4x4，而不是像 ViT 一样16x16，所以说它经过 patch partition 打成 patch 之后，得到图片的尺寸是56x56x48，56就是224/4，因为 patch size 是4，向量的维度48，因为4x4x3，3 是图片的 RGB 通道  
打完了 patch ，接下来就要做 Linear Embedding，也就是说要把向量的维度变成一个预先设置好的值，就是 Transformer 能够接受的值，在 Swin Transformer 的论文里把这个超参数设为 c，对于 Swin tiny 网络来说，也就是上图中画的网络总览图，它的 c 是96，所以经历完 Linear Embedding 之后，输入的尺寸就变成了56x56x96，前面的56x56就会拉直变成3136，变成了序列长度，后面的96就变成了每一个token向量的维度，其实 Patch Partition 和 Linear Embedding 就相当于是 ViT 里的Patch Projection 操作，而在代码里也是用一次卷积操作就完成了。
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224, 224, 3), patch_size=4, embedding_dim=96):
        super().__init__()
        self.img_size = img_size
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.projection = nn.Conv2d(in_channels=img_size[2], out_channels=embedding_dim, kernel_size=patch_size,
                                    stride=patch_size)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        n, c, h, w = x.shape
        assert h == self.img_size[0] and w == self.img_size[1], f"Input size ({h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.projection(x)  # [batch, 96, 56, 56]
        x = x.flatten(2).transpose(2, 1)  # [batch, 96, 3136] -> [batch, 3136, 96]
        x = self.layer_norm(x)  # [batch, 3136, 96]
        return x
```

## Swin Transformer block
 ![Swin Transformer Block](https://image.iokko.cn/file/8176da67e62b4c62cf80b.png) 	 ![Transformer Encoder](https://image.iokko.cn/file/d418c837218ad2b82a4b6.png)     
 上图右侧为 ViT 的Transformer结构，对比Swin Transformer只是将Multi-HeadAttention结构替换为了W-MSA和SW-MSA。所以Swin Transformer的结构是成对出现,stage1-4为[2, 2, 6, 2]。
首先序列长度是3136，对于 ViT 来说，用 patch size 16x16，它的序列长度就只有196，是相对短很多的，这里的3136就太长了，是目前来说Transformer不能接受的序列长度，所以 Swin Transformer 就引入了基于窗口的自注意力计算，每个窗口按照默认来说，都只有7x7个 patch，所以说序列长度就只有49就相当小了，这样就解决了计算复杂度的问题  
所以也就是说， stage1中的swin transformer block 是基于窗口计算自注意力的，现在暂时先把 transformer block当成是一个黑盒，只关注输入和输出的维度，对于 Transformer 来说，如果不对它做更多约束的话，Transformer输入的序列长度是多少，输出的序列长度也是多少，它的输入输出的尺寸是不变的，所以说在 stage1 中经过两层Swin Transformer block 之后，输出还是56x56x96  
到这其实 Swin Transformer的第一个阶段就走完了，也就是先过一个 Patch Projection 层，然后再过一些 Swin Transformer block，接下来如果想要有多尺寸的特征信息，就要构建一个层级式的 transformer，也就是说需要一个像卷积神经网络里一样，有一个类似于池化的操作
```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, in_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, dropout=0., drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        mlp_hidden_dim = int(in_dim * mlp_ratio)

        self.layer_norm_1 = nn.LayerNorm(in_dim)
        # W-MSA
        self.window_attention = WindowAttention(num_hidden=in_dim, window_size=window_size, n_head=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(in_dim)
        # MLP
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, in_dim),
            nn.Dropout(dropout)
        )
        #
        self.layer_norm_3 = nn.LayerNorm(in_dim)
        # SW-WSA
        self.shift_window_attention = WindowAttention(num_hidden=in_dim, window_size=window_size, n_head=num_heads)
        self.layer_norm_4 = nn.LayerNorm(in_dim)
        # MLP2
        self.mlp_2 = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, in_dim),
            nn.Dropout(dropout)
        )

        # attention mask
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    @staticmethod
    def window_partition(x, window_size):
        n, h, w, c = x.shape  # [batch, 56, 56, 96]
        x = x.view(n, h // window_size, window_size, w // window_size, window_size, c)  # [batch, 8, 7, 8, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)  # [64*batch, 7, 7, 96]
        return x

    @staticmethod
    def window_reverse(windows, window_size, h, w):
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def forward(self, x):
        """x: batch, sql_len, channels"""
        h, w = self.input_resolution  # [56, 56]
        n, l, c = x.shape  # [batch, 3136, 96]
        assert l == h * w, "input feature has wrong size"

        shortcut = x
        # 1.layer norm 1
        x = self.layer_norm_1(x)  # [batch, 3136, 96]
        x = x.view(n, h, w, c)  # [batch, 56, 56, 96]

        # 2 W-MSA
        # 2.1partition windows
        x_windows = self.window_partition(x, self.window_size)  # [64*batch, 7, 7, 96]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [64*batch, 49, 96]
        # 2.2attention
        attention_windows = self.window_attention(x_windows)  # [64*batch, 49, 96]
        # 2.3merge windows
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, c)  # [64*batch, 7, 7, 96]
        # 2.4reverse window
        x = self.window_reverse(attention_windows, self.window_size, h, w)  # [batch, 56, 56, 96]
        x = x.view(n, h * w, c)  # [batch, 3136, 96]
        # 3 shortcut 1
        x_ = shortcut + self.drop_path(x)  # [batch, 3136, 96]
        # 4 layer norm
        x = self.layer_norm_2(x_)
        # 5 MLP
        x = self.mlp_1(x)
        # 6 shortcut 2
        x = x_ + self.drop_path(x)  # [batch, 3136, 96]
        # 7 layer norm
        x = self.layer_norm_3(x)
        x_ = x
        # 8 SW-MSA
        x = x.view(n, h, w, c)
        # 8.1cyclic shift windows
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # 8.2partition windows
        shifted_x_windows = self.window_partition(shifted_x, self.window_size)  # [64*batch, 7, 7, 96]
        shifted_x_windows = shifted_x_windows.view(-1, self.window_size * self.window_size, c)  # [64*batch, 49, 96]
        # 8.3attention
        s_attention_windows = self.shift_window_attention(shifted_x_windows, self.attn_mask)  # [64*batch, 49, 96]
        # 8.4merge windows
        s_attention_windows = s_attention_windows.view(-1, self.window_size, self.window_size,
                                                       c)  # [64*batch, 7, 7, 96]
        # 8.5reverse window
        shifted_x = self.window_reverse(s_attention_windows, self.window_size, h, w)  # [batch, 56, 56, 96]
        # 8.6reverse cyclic shift windows
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(n, h * w, c)  # [batch, 3136, 96]
        # 9 shortcut 3
        x = x_ + self.drop_path(x)
        x_ = x
        # 10 layer norm
        x = self.layer_norm_3(x)
        # 11 mlp 2
        x = self.mlp_2(x)
        # 12 shortcut 4
        x = x_ + self.drop_path(x)
        return x

```
## Patch Merging操作


Swin Transformer提出来了一个类似于池化的操作叫做 patch merging，就是把相邻的小 patch 合成一个大 patch，这样合并出来的这一个大patch其实就能看到之前四个小patch看到的内容，它的感受野就增大了，同时也能抓住多尺寸的特征。

Patch Merging 这样就可以起到下采样一个特征图的效果了

这里因为是想下采样两倍，所以说在选点的时候是每隔一个点选一个  
![patch merging](https://image.iokko.cn/file/a113b22231c612976c623.png)

其实在这里的1、2、3、4并不是矩阵里有的值，而是给它的一个序号，同样序号位置上的 patch 就会被 merge 到一起，这个序号只是为了帮助理解  
经过隔一个点采一个样之后，原来的这个张量就变成了四个张量，也就是说所有的1都在一起了，2在一起，3在一起，4在一起，如果原张量的维度是 h \* w \* c ，当然这里 c 没有画出来，经过这次采样之后就得到了4个张量，每个张量的大小是 h/2、w/2，它的尺寸都缩小了一倍  
现在把这四个张量在 c 的维度上拼接起来，张量的大小就变成了 h/2 \* w/2 \* 4c，相当于用空间上的维度换了更多的通道数  
通过这个操作，就把原来一个大的张量变小了，就像卷积神经网络里的池化操作一样，为了跟卷积神经网络那边保持一致（不论是 VGGNet 还是 ResNet，一般在池化操作降维之后，通道数都会翻倍，从128变成256，从256再变成512），所以这里也只想让他翻倍，而不是变成4倍，所以紧接着又再做了一次操作，就是在 c 的维度上用一个1乘1的卷积，把通道数降下来变成2c，通过这个操作就能把原来一个大小为 h*w*c 的张量变成 h/2 \* w/2 \*2c 的一个张量，也就是说空间大小减半，但是通道数乘2，这样就跟卷积神经网络完全对等起来了  
整个这个过程就是 Patch Merging，经历过这次Patch Merging操作之后，输出的大小就从56x56x96变成了28x28x192，经过stage2中的 Transformer block，尺寸是不变的，所以出来之后还是28x28x192

这样第二阶段也就完成了，第三和第四阶段都是同理，都是先进来做一次Patch Merging，然后再通过一些 Swin Transformer block，所以维度就进一步降成了14x14x384以及7x7x768

这里其实会发现，特征图的维度真的跟卷积神经网络好像，因为如果回想残差网络的多尺寸的特征，就是经过每个残差阶段之后的特征图大小也是56x56、28x28、14x14，最后是7x7

而且为了和卷积神经网络保持一致，Swin Transformer这篇论文并没有像 ViT 一样使用 CLS token，ViT 是给刚开始的输入序列又加了一个 CLS token，所以这个长度就从196变成了197，最后拿 CLS token 的特征直接去做分类，但 Swin Transformer 没有用这个 token，它是像卷积神经网络一样，在得到最后的特征图之后用global average polling，就是全局池化的操作，直接把7x7就取平均拉直变成1了。作者这个图里并没有画，因为 Swin Transformer的本意并不是只做分类，它还会去做检测和分割，所以说它只画了骨干网络的部分，没有去画最后的分类头或者检测头，但是如果是做分类的话，最后就变成了1x768，然后又变成了1x1000。如果是做ImageNet的话，这样就完成了整个一个分类网络的前向过程。

所以看完整个前向过程之后，就会发现 Swin Transformer 有四个 stage，还有类似于池化的 patch merging 操作，自注意力还是在小窗口之内做的，以及最后还用的是 global average polling，所以说 Swin Transformer 这篇论文真的是把卷积神经网络和 Transformer 这两系列的工作完美的结合到了一起，也可以说它是披着Transformer皮的卷积神经网络
```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, in_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.liner = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.layer_norm = norm_layer(in_dim * 4)

    def forward(self, x):
        h, w = self.input_resolution
        n, sql_len, c = x.shape  # [batch, 3136, 96]
        assert sql_len == h * w, "input feature has wrong size"
        assert h % 2 == 0 and w % 2 == 0, f"x size ({h}*{w}) are not even."

        x = x.view(n, h, w, c)  # [batch, 56, 56, 96]

        x0 = x[:, 0::2, 0::2, :]  # [batch, 28, 28, 96]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [batch, 28, 28, 384]

        x = x.view(n, -1, 4 * c)
        x = self.layer_norm(x)
        x = self.liner(x)
        return x
```

## BasicLayer
```python
class BasicLayer(nn.Module):
    def __init__(self, embed_dim, input_resolution, depth, num_head, window_size, norm_layer, is_last_layer=False):
        super().__init__()
        self.is_last_layer = is_last_layer
        # swin Transformer block 56->28->14->7
        self.stbs = nn.ModuleList([SwinTransformerBlock(in_dim=embed_dim, input_resolution=input_resolution,
                                                        num_heads=num_head, window_size=window_size,
                                                        shift_size=window_size // 2) for _ in range(depth)])

        # patch merging
        self.pm = PatchMerging(input_resolution=input_resolution, in_dim=embed_dim, norm_layer=norm_layer)

    def forward(self, x):
        for stb in self.stbs:
            x = stb(x)
        if not self.is_last_layer:
            x = self.pm(x)
        return x
```

## W-MSA和SW-MSA窗口自注意力：
```python
class WindowAttention(nn.Module):
    def __init__(self, num_hidden, window_size, n_head, bias=False, dropout=0.1):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_head = n_head
        self.window_size = window_size

        self.q_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.k_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.v_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.concat_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_head))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        n, sql_len, c = x.shape
        # 分多个头  [batch, seq_len, hidden_size] -> [batch, seq_len, head_size, d_k] -> [batch, head_size, seq_len, d_k]
        d_k = self.num_hidden // self.num_head
        q = self.q_mat(x).view(x.size(0), -1, self.num_head, d_k).transpose(1, 2)
        k = self.k_mat(x).view(x.size(0), -1, self.num_head, d_k).transpose(1, 2)
        v = self.v_mat(x).view(x.size(0), -1, self.num_head, d_k).transpose(1, 2)
        score = q @ k.transpose(-2, -1)

        # relative position
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # [49, 49, n_head]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [n_head, 49, 49]
        score = score + relative_position_bias.unsqueeze(0)
        # mask
        if mask is not None:
            score = score.view(n // mask.shape[0], mask.shape[0], self.num_head, sql_len, sql_len)
            score = score + mask.unsqueeze(1).unsqueeze(0)
            score = score.view(-1, self.num_head, sql_len, sql_len)  # [batch, n_head, 49, 49]

        score = self.softmax(score)

        score = self.attention_dropout(score)
        attention = score @ v  # [batch, head, sql_len ,d_k]
        x = attention.transpose(1, 2).reshape(n, sql_len, c)
        x = self.concat_mat(x)
        x = self.dropout(x)
        return x
```

![window partition](https://image.iokko.cn/file/63735fd37572e84db3096.png)
基于窗口（移动窗口）的自注意力

全局自注意力：会导致平方倍的复杂度，（对于视觉的下游任务，尤其是密集型的任务，或者遇到非常大尺寸的图片，全局计算自注意力的复杂度就非常贵）

最小计算单元是patch，每一个窗口里有M x M个patch（论文里M=7），所有的自注意力计算都是在小窗口里完成的（序列长度永远=7x7=49），原来大的整体特征图会有多少窗口？8 x 8=64

我们会在64个窗口里分别计算自注意力

（1）标准的多头自注意力的计算复杂度，h=w=56

（2）基于窗口的自注意力，M=7（一个窗口某条边上有多少patch）

两公式差别，56 x 56 和7 x 7相差巨大，窗口自注意力很好的解决了计算量问题

新问题：窗口和窗口之间没有通信了，这样达不到全局建模了，会限制模型的能力，我们希望窗口和窗口之间通信起来

作者提出移动窗口的方式  
![window partition](https://image.iokko.cn/file/f0dba966c8471b84a9d67.png)

上图是一个基础版本的移动窗口，就是把左边的窗口模式变成了右边的窗口方式


虽然这种方式已经能够达到窗口和窗口之间的互相通信了，但是会发现一个问题，就是原来计算的时候，特征图上只有四个窗口，但是做完移动窗口操作之后得到了9个窗口，窗口的数量增加了，而且每个窗口里的元素大小不一，比如说中间的窗口还是4*4，有16个 patch，但是别的窗口有的有4个 patch，有的有8个 patch，都不一样了，如果想做快速运算，就是把这些窗口全都压成一个 patch直接去算自注意力，就做不到了，因为窗口的大小不一样

有一个简单粗暴的解决方式就是把这些小窗口周围再 pad 上0 ，把它照样pad成和中间窗口一样大的窗口，这样就有9个完全一样大的窗口，这样就还能把它们压成一个batch，就会快很多

但是这样的话，无形之中计算复杂度就提升了，因为原来如果算基于窗口的自注意力只用算4个窗口，但是现在需要去算9个窗口，复杂度一下提升了两倍多，所以还是相当可观的

那怎么能让第二次移位完的窗口数量还是保持4个，而且每个窗口里的patch数量也还保持一致呢？作者提出了一个非常巧妙的掩码方式，如下图所示

上图是说，当通过普通的移动窗口方式，得到9个窗口之后，现在不在这9个窗口上算自注意力，先再做一次循环移位（ cyclic shift ）  
经过这次循环移位之后，原来的窗口（虚线）就变成了现在窗口（实线）的样子，那如果在大的特征图上再把它分成四宫格的话，我在就又得到了四个窗口，意思就是说移位之前的窗口数也是4个，移完位之后再做一次循环移位得到窗口数还是4个，这样窗口的数量就固定了，也就说计算复杂度就固定了

但是新的问题就来了，虽然对于移位后左上角的窗口（也就是移位前最中间的窗口）来说，里面的元素都是互相紧挨着的，他们之间可以互相两两做自注意力，但是对于剩下几个窗口来说，它们里面的元素是从别的很远的地方搬过来的，所以他们之间，按道理来说是不应该去做自注意力，也就是说他们之间不应该有什么太大的联系

解决这个问题就需要一个很常规的操作，也就是掩码操作，这在Transformer过去的工作里是层出不穷，很多工作里都有各式各样的掩码操作

在 Swin Transformer这篇论文里，作者也巧妙的设计了几种掩码的方式，从而能让一个窗口之中不同的区域之间也能用一次前向过程，就能把自注意力算出来，但是互相之间都不干扰，也就是后面的 masked Multi-head Self Attention（MSA）

算完了多头自注意力之后，还有最后一步就是需要把循环位移再还原回去，也就是说需要把A、B、C再还原到原来的位置上去，原因是还需要保持原来图片的相对位置大概是不变的，整体图片的语义信息也是不变的，如果不把循环位移还原的话，那相当于在做Transformer的操作之中，一直在把图片往右下角移，不停的往右下角移，这样图片的语义信息很有可能就被破坏掉了

```python
import torch

import matplotlib.pyplot as plt


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


window_size = 7
shift_size = 3
H, W = 14, 14
img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
h_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None))
cnt = 0
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1

mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
mask_windows = mask_windows.view(-1, window_size * window_size)

attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

plt.matshow(img_mask[0, :, :, 0].numpy())
plt.matshow(attn_mask[0].numpy())
plt.matshow(attn_mask[1].numpy())
plt.matshow(attn_mask[2].numpy())
plt.matshow(attn_mask[3].numpy())

plt.show()
```

![window attention](https://image.iokko.cn/file/9874d90660d8183ff776b.png)


所以说整体而言，上图介绍了一种高效的、批次的计算方式，比如说本来移动窗口之后得到了9个窗口，而且窗口之间的patch数量每个都不一样，为了达到高效性，为了能够进行批次处理，先进行一次循环位移，把9个窗口变成4个窗口，然后用巧妙的掩码方式让每个窗口之间能够合理地计算自注意力，最后再把算好的自注意力还原，就完成了基于移动窗口的自注意力计算

作者通过这种巧妙的循环位移的方式和巧妙设计的掩码模板，从而实现了只需要一次前向过程，就能把所有需要的自注意力值都算出来，而且只需要计算4个窗口，也就是说窗口的数量没有增加，计算复杂度也没有增加，非常高效的完成了这个任务

## SwinTransformer
```python
class SwinTransformer(nn.Module):
    def __init__(self, img_size=(224, 224, 3), patch_size=4, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # patch partition
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, embedding_dim=embed_dim)
        patches_resolution = self.patch_embedding.patches_resolution

        # swin transformer layers
        self.swin_layers = nn.ModuleList([
            BasicLayer(embed_dim=int(embed_dim * 2 ** i),
                       input_resolution=(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)),
                       depth=depths[i],
                       num_head=num_heads[i], window_size=window_size, norm_layer=norm_layer,
                       is_last_layer=False if (i < self.num_layers - 1) else True)
            for i in range(self.num_layers)
        ])
        # print(self.swin_layers)
        # layer norm
        self.layer_norm = nn.LayerNorm(self.num_features)
        # average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # linear
        self.cls_head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # patch embedding
        x = self.patch_embedding(x)  # [batch, c, h, w] -> [batch, 3136, 96]
        # swin transformer layers
        for layer in self.swin_layers:
            x = layer(x)  # [batch, 3136, 96] -> [batch, 784, 192] -> [batch, 196, 384] -> [batch, 49, 768]
            print(x.shape)
        x = self.layer_norm(x)  # [batch, 49, 768]
        x = x.transpose(1, 2)  # [batch, 768, 49]
        # average pooling
        x = self.avg_pool(x)  # [batch, 768, 1]
        x = torch.flatten(x, 1)  # [batch, 768]
        # classify
        x = self.cls_head(x)  # [batch, class_num]
        return x
```