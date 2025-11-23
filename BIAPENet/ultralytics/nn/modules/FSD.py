import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import cv2
import numpy as np
import torch.fft as fft

import torch_dct as dct


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MSCAttn(nn.Module):
    def __init__(self, channel, dct_h=20, dct_w=20, reduction=16, freq_sel_method='top16'):
        super(MSCAttn, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.channel = channel

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mapper_x, mapper_y = self.get_freq_indices(freq_sel_method)
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_split = len(mapper_x)
        self.mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        self.mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.num_freq = len(mapper_x)

    def forward(self, x):
        # print(x.shape)
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        n, c, h, w = x.shape
        # self.register_buffer('weight', self.get_ms_filter(h, w, self.mapper_x, self.mapper_y, self.channel))
        self.weight = self.get_ms_filter(h, w, self.mapper_x, self.mapper_y, self.channel)
        self.weight = self.weight.to(x.device)
        x_1 = x * self.weight
        result = torch.sum(x_1, dim=[2, 3])
        y = self.fc(result).view(n, c, 1, 1)  # [4, 512, 1, 1 ]
        return x * y.expand_as(x)  # [4 ,512, 20, 20]

    def get_freq_indices(self, method):
        assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                          'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                          'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
        num_freq = int(method[3:])
        if 'top' in method:
            all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
            all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
            mapper_x = all_top_indices_x[:num_freq]
            mapper_y = all_top_indices_y[:num_freq]
        elif 'low' in method:
            all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
            all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
            mapper_x = all_low_indices_x[:num_freq]
            mapper_y = all_low_indices_y[:num_freq]
        elif 'bot' in method:
            all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
            all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
            mapper_x = all_bot_indices_x[:num_freq]
            mapper_y = all_bot_indices_y[:num_freq]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y

    def build_filter(self, pos, freq, POS):
        # result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        result = torch.cos(torch.tensor(math.pi) * freq * (pos + 0.5) / POS) / torch.sqrt(torch.tensor(POS))
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_ms_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter


class CoT(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1),
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w  保持维度不变
        # print(type(k1))
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w
        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k(D),h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        # print((att.mean(2, keepdim=True)).shape)
        att = att.mean(2, keepdim=True).view(bs, c, -1)  # bs,c,h*w
        # print((F.softmax(att, dim=-1)))
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2


class FreDomain(nn.Module):
    def __init__(self, channel, dct_h=20, dct_w=20, freq_sel_method='top16'):
        super(FreDomain, self).__init__()

        self.freqAttn = CoT(channel)   # dct变换-> cot attention
        self.mscattn = MSCAttn(channel, dct_h=dct_h, dct_w=dct_w, freq_sel_method=freq_sel_method)
        self.conv = Conv(channel, channel, 1)

    def forward(self, x):
        x_dct = self.get_dct(x)
        x_dct_attn = self.freqAttn(x_dct)
        x_dct_attn_idct = self.get_idct(x_dct_attn)
        # freq_ms = self.mscattn(x)
        # result = torch.cat([x_dct_attn_idct, freq_ms], dim=1)
        return self.conv(x_dct_attn_idct)

    def get_dct(self, x):
        # 将Tensor转换为NumPy数组
        # numpy_data = x.detach().cpu().numpy().astype(np.float32)
        # # 对数组进行DCT变换
        # dct_data = np.zeros_like(numpy_data)
        # for i in range(numpy_data.shape[0]):
        #     for j in range(numpy_data.shape[1]):
        #         dct_data[i, j, :, :] = cv2.dct(numpy_data[i, j, :, :].astype(np.float32))
        # return torch.from_numpy(dct_data)
        # b, c, h, w = x.shape
        # x_complex = torch.view_as_complex(torch.stack((x, torch.zeros_like(x)), dim=-1).to(torch.float))
        # # 在最后两个维度上应用FFT进行二维DCT变换
        # dct_x_complex = fft.fftn(x_complex, dim=(-2, -1))
        # # 获取实部部分作为DCT变换后的结果
        # dct_x = dct_x_complex.real
        # return dct_x[:, :c, :, :]
        #
        dct_x = dct.dct_2d(x, norm="ortho")

        # print("dct_x shape", dct_x.shape)
        return dct_x

    def get_idct(self, x):
        # # 对DCT变换后的数组进行逆变换
        # dct_data = x.detach().cpu().numpy().astype(np.float32)
        # idct_data = np.zeros_like(dct_data)
        # for i in range(dct_data.shape[0]):
        #     for j in range(dct_data.shape[1]):
        #         idct_data[i, j, :, :] = cv2.idct(dct_data[i, j, :, :].astype(np.float32) )
        # return torch.from_numpy(idct_data)
        # 将DCT结果转换为复数形式，虚部部分设置为0，并指定数据类型为ComplexFloat
        # b, c, h, w = x.shape
        # dct_result_complex = torch.view_as_complex(
        #     torch.stack((x, torch.zeros_like(x)), dim=-1).to(torch.float))
        # # 在最后两个维度上应用FFT进行二维DCT逆变换
        # idct_result_complex = fft.ifftn(dct_result_complex, dim=(-2, -1))
        # # 获取实部部分作为DCT逆变换后的结果
        # idct_result = idct_result_complex.real
        # return idct_result[:, :c, :, :]

        idct_x = dct.idct_2d(x, norm="ortho")
        # print("idct_x shape", idct_x.shape)
        return idct_x


class ASPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    def __init__(self, c1, c2, k=3, dilation=2):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.
        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super(ASPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.k = k
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2, dilation=dilation)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        x1 = F.pad(x, (self.k // 2, self.k // 2, self.k // 2, self.k // 2), mode='constant', value=0)
        y1 = self.m(x1)
        x2 = F.pad(y1, (self.k // 2, self.k // 2, self.k // 2, self.k // 2), mode='constant', value=0)
        y2 = self.m(x2)
        x3 = F.pad(y2, (self.k // 2, self.k // 2, self.k // 2, self.k // 2), mode='constant', value=0)
        return self.cv2(torch.cat((x, y1, y2, self.m(x3)), 1))


class FSDAttention(nn.Module):
    def __init__(self, channel, dct_h=20, dct_w=20, freq_sel_method='top16'):
        super(FSDAttention, self).__init__()

        self.freq_domain = FreDomain(channel, dct_h=dct_h, dct_w=dct_w, freq_sel_method=freq_sel_method)
        self.space_domain = ASPPF(channel, channel)
        self.conv = Conv(channel * 2, channel, 1)

    def forward(self, x):
        y1 = self.freq_domain(x)
        y2 = self.space_domain(x)
        result = torch.cat([y1, y2], 1)
        return self.conv(result)


if __name__ == '__main__':

    # 创建一个随机张量模拟VisDrone数据集，形状为 [4, 512, 20, 20]
    x = torch.randn([4, 512, 20, 20])

    # 对 x 进行二维DCT变换
    # dct_result = dct2(x)[:, :512, :, :]

    # print(dct_result.shape)
    model = FSDAttention(512)
    print(model(x).shape)




