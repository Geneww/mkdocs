# MAE梳理
[Masked Autoencoders Are Scalable Vision Learners论文原文](https://arxiv.org/pdf/2111.06377)  
MAE：

BERT 的一个 CV 的版本
基于 ViT ，BERT化
把整个训练 拓展到没有标号的数据上面
通过完型填空来获取图片的一个理解
不是第一个将 BERT 拓展到 CV 上
MAE 很有可能 未来影响最大
BERT 加速了 Transformer 架构 在 NLP 的应用
MAE 加速 Transformer 在 CV 上的应用

![MAE 的模型架构](https://image.iokko.cn/file/c54471fc0d1511352f995.png)

## MAE 的模型架构

预训练流程：input --> patches --> masked --> unmasked patches in encoder --> unmasked + masked 按位置排列 进 decoder --> decoder 重构 masked patches 的像素

patches + masked：一张红色鸟图片进来，切成 patches，masked 块 (3/4) 是 灰色的。
unmasked patches，encoder：没有 masked (1 / 4) 的块 进入 encoder (ViT)，得到每一块的特征（蓝色）。
encoder 的输出 和 masked tokens 按照在图片中的原始位置排列成一长条向量 （包含位置信息）。
长条向量 进入 decoder，解码器尝试重构缺失的像素信息，还原原始图片


## MAE代码

```python
import torch
import torch.nn as nn

# local files
from transformer.transformer import TransformerEncoderLayer, TransformerEncoder


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


class MLP(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super(MLP, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x):
        return self.linear(self.layer_norm(x))


class MaskAutoEncoder(nn.Module):
    def __init__(self, img_size=(224, 224, 3), patch_size=16,
                 embedding_dim=1024, encoder_num=24, num_heads=16, masking_ratio=0.75,
                 decoder_embedding_dim=512, decoder_number=8, decoder_num_heads=16):
        super(MaskAutoEncoder, self).__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        # mae patcher
        height, weight, in_channels = img_size
        assert height % patch_size == 0 and weight % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.num_patch = (height // patch_size) * (weight // patch_size)

        # mae encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patch + 1, embedding_dim), requires_grad=False)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, n_head=num_heads)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=encoder_num)
        self.encoder_mlp = MLP(embedding_dim=embedding_dim)

        # mae decoder
        self.decoder_embedding = nn.Linear(embedding_dim, decoder_embedding_dim, bias=True)
        self.mask_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, self.num_patch + 1, decoder_embedding_dim),
                                                  requires_grad=False)
        decoder_layer = TransformerEncoderLayer(d_model=decoder_embedding_dim, n_head=decoder_num_heads)
        self.decoder = TransformerEncoder(decoder_layer, num_layers=decoder_number)
        self.decoder_mlp = MLP(embedding_dim=decoder_embedding_dim)

        self.decoder_pred = nn.Linear(decoder_embedding_dim, patch_size ** 2 * in_channels,
                                      bias=True)  # decoder to patch

    def random_mask(self, x, mask_ratio):
        batch, length, dim = x.shape  # batch, length, dim
        len_keep = int(length * (1 - mask_ratio))

        noise = torch.rand(batch, length)  # [batch, length]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch, length])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def patch_image(self, images):
        """ images: n c h w"""
        p = self.patch_size
        assert images.shape[2] == images.shape[3] and images.shape[2] % p == 0
        batch = images.shape[0]
        c = images.shape[1]
        h = w = images.shape[2] // p
        x = images.reshape(shape=(batch, c, h, p, w, p))  # [batch, 3, 224, 224] -> [batch, 3, 14, 16, 14, 16]
        x = torch.einsum('nchpwq->nhwpqc', x)  # [batch, 14, 14, 16, 16, 3]
        x = x.reshape((batch, h * w, p ** 2 * c))  # [batch, 196, 768]
        return x

    def cal_loss(self, images, pred, mask):
        """mse loss"""
        target = self.patch_image(images)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x):
        images = x
        # patch image
        x = self.patcher(x)  # [batch, 1024, 14, 14]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2,
                                                       1)  # [batch, 1024, 14, 14] -> [batch, 1024, 196] -> [batch, 196, 1024]
        # add pos embedding
        x = x + self.pos_embedding[:, :-1, :]
        # mask
        x, mask, ids_restore = self.random_mask(x, self.masking_ratio)  # [batch, 49, 1024]
        # # 统计张量中值为 0 的元素个数
        # num_zeros = torch.sum(mask == 0).item()
        # # 计算值为 0 的元素所占比例
        # zero_ratio = num_zeros / mask.numel()
        # print("Value 0 appears in the tensor with a ratio of:", zero_ratio)

        # add cls token
        cls_token = self.cls_token + self.pos_embedding[:, -1:, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)  # [batch, 1, 1024]
        x = torch.cat((cls_token, x), dim=1)  # [batch, 49+1, 1024]
        x = self.encoder(x)
        x = self.encoder_mlp(x)  # encode out [batch, sql_len, 1024]

        # decoder
        x = self.decoder_embedding(x)  # [batch, sql_len, 512]
        mask_tokens = self.mask_cls_token.repeat(x.shape[0], int(self.num_patch * self.masking_ratio),
                                                 1)  # [batch, 196*0.75, 512]
        x_ = torch.cat([x[:, :-1, :], mask_tokens], dim=1)  # no cls token [batch, 50-1+147, 512]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # un shuffle
        x = torch.cat([x[:, -1:, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embedding
        # encoder out + mask
        x = self.decoder(x)
        x = self.decoder_mlp(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, :-1, :]
        loss = self.cal_loss(images, x, mask)
        return loss, x, mask
```