import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import PCAEncoder
import sys
from torch.nn.parameter import Parameter
from torch.nn import Softmax
from torch import nn
import math
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class UnetDownsample(nn.Module):
    def __init__(self, n_feat):
        super(UnetDownsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UnetUpsample(nn.Module):
    def __init__(self, n_feat):
        super(UnetUpsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=default_conv,
                 res_scale=1):
        super(NonLocalSparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
                                                                                             -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]

        # add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  # [N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):

        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        # number of hash buckets/hash bits
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
        hash_codes = hash_codes.detach()

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % L)  # now range from (0->H*W)
        x_embed_sorted = batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size, C))  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret


class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CRB_Layer(nn.Module):
    def __init__(self, nf1):
        super(CRB_Layer, self).__init__()

        self.norm1 = LayerNorm2d(nf1)

        # nonlocal attention
        self.nla = NonLocalSparseAttention(channels=nf1)

        # local
        conv_local = [
            nn.Conv2d(nf1, nf1, 3, 1, 1, groups=nf1),
            nn.GELU(),
            nn.Conv2d(nf1, nf1, 3, 1, 1, groups=nf1)
        ]

        self.conv_local = nn.Sequential(*conv_local)

        self.conv1_last = nn.Conv2d(2 * nf1, nf1, 1, 1)

        # channel attention
        self.ca = CALayer(nf1)

        self.norm2 = LayerNorm2d(nf1)

        self.gdfn = FeedForward(nf1, 2.66, bias=False)

    def forward(self, x):
        f1 = x

        x = self.norm1(x)

        # pixel attention
        out1 = self.nla(x)

        # local
        out2 = self.conv_local(x)

        out = [out1, out2]
        out = torch.cat(out, 1)

        # the fusion of channel
        out = self.conv1_last(out)

        # channel attention
        out = self.ca(out)

        out_temp = out + f1

        out = self.norm2(out_temp)

        f1 = self.gdfn(out) + out_temp
        return f1

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

class Restorer(nn.Module):
    def __init__(
        self, in_nc=3, out_nc=3, nf=64, nb=40, scale=4, input_para=10, min=0.0, max=1.0
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.rate3 = torch.nn.Parameter(torch.Tensor(1))
        self.rate4 = torch.nn.Parameter(torch.Tensor(1))
        self.rate5 = torch.nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

        self.head = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)

        # encoder block 1
        encoder_block1 = [CRB_Layer(nf) for _ in range(6)]
        self.encoder_block1 = nn.Sequential(*encoder_block1)

        self.conv3_noise1 = nn.Conv2d(1, 64, 3, 1, 1)

        # downsample 1
        self.down1 = UnetDownsample(nf)

        # encoder block 2
        encoder_block2 = [CRB_Layer(nf * 2) for _ in range(8)]
        self.encoder_block2 = nn.Sequential(*encoder_block2)

        self.conv3_noise2 = nn.Conv2d(1, 128, 3, 1, 1)

        # downsample 2
        self.down2 = UnetDownsample(nf * 2)

        # latent block
        latent_block = [CRB_Layer(nf * 4) for _ in range(10)]
        self.latent_block = nn.Sequential(*latent_block)

        self.conv3_noise3 = nn.Conv2d(1, 256, 3, 1, 1)

        # upsample 1
        self.up1 = UnetUpsample(nf * 4)

        self.conv3_noise4 = nn.Conv2d(1, 128, 3, 1, 1)

        # conv1x1_1
        self.conv1x1_1 = nn.Conv2d(nf * 2 * 2, nf * 2, 1, 1)

        # decoder block 1
        decoder_block1 = [CRB_Layer(nf * 2) for _ in range(6)]
        self.decoder_block1 = nn.Sequential(*decoder_block1)

        # upsample 2
        self.up2 = UnetUpsample(nf * 2)

        self.conv3_noise5 = nn.Conv2d(1, 128, 3, 1, 1)

        # decoder block 2
        decoder_block2 = [CRB_Layer(nf * 2) for _ in range(8)]
        self.decoder_block2 = nn.Sequential(*decoder_block2)

        # refine block
        refine_block = [CRB_Layer(nf * 2) for _ in range(8)]
        self.refine_block = nn.Sequential(*refine_block)

        self.fusion = nn.Conv2d(nf * 2, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        init_rate_half(self.rate3)
        init_rate_half(self.rate4)
        init_rate_half(self.rate5)

    def forward(self, input, rate1=None, rate2=None, rate3=None, rate4=None, rate5=None):
        B, C, H, W = input.size()  # I_LR batch

        if self.training is True:
            rate1 = self.rate1
            rate2 = self.rate2
            rate3 = self.rate3
            rate4 = self.rate4
            rate5 = self.rate5
        if self.training is False and (rate1==None or rate2==None or rate3==None):
            rate1 = self.rate1
            rate2 = self.rate2
            rate3 = self.rate3
            rate4 = self.rate4
            rate5 = self.rate5
        # bicubic
        upsample = transforms.Resize((input.shape[2] * 4, input.shape[3] * 4), interpolation=transforms.InterpolationMode.BICUBIC)
        lr_bic = upsample(input)
        lr_bic = torch.clamp(lr_bic, min=0, max=1)

        f = self.head(input)

        # encoder
        f_en1 = self.encoder_block1(f)
        
        rand_noise1 = torch.randn(B, 1, f_en1.shape[2], f_en1.shape[3]).cuda()
        rand_noise1 = self.conv3_noise1(rand_noise1)

        f_en1 = f_en1 * self.rate1 * rand_noise1
        
        f_en1_down = self.down1(f_en1)

        f_en2 = self.encoder_block2(f_en1_down)

        rand_noise2 = torch.randn(B, 1, f_en2.shape[2], f_en2.shape[3]).cuda()
        rand_noise2 = self.conv3_noise2(rand_noise2)

        f_en2 = f_en2 * self.rate2 * rand_noise2
        
        f_en2_down = self.down2(f_en2)

        # latent
        f_latent = self.latent_block(f_en2_down)

        rand_noise3 = torch.randn(B, 1, f_latent.shape[2], f_latent.shape[3]).cuda()
        rand_noise3 = self.conv3_noise3(rand_noise3)

        f_latent = f_latent * self.rate3 * rand_noise3

        # decoder
        f_latent_up = self.up1(f_latent)
        f_conv1 = self.conv1x1_1(torch.cat([f_en2, f_latent_up], dim=1))
        f_de1 = self.decoder_block1(f_conv1)

        rand_noise4 = torch.randn(B, 1, f_de1.shape[2], f_de1.shape[3]).cuda()
        rand_noise4 = self.conv3_noise4(rand_noise4)

        f_de1 = f_de1 * self.rate4 * rand_noise4
        
        f_de1_up = self.up2(f_de1)
        f_cat2 = torch.cat([f_en1, f_de1_up], dim=1)
        f_de2 = self.decoder_block2(f_cat2)

        rand_noise5 = torch.randn(B, 1, f_de2.shape[2], f_de2.shape[3]).cuda()
        rand_noise5 = self.conv3_noise5(rand_noise5)

        f_de2 = f_de2 * self.rate5 * rand_noise5

        # refine
        f_re = self.refine_block(f_de2)

        f = self.fusion(f_re)
        out = self.upscale(f) + lr_bic

        return out  # torch.clamp(out, min=self.min, max=self.max)


class NLCUnet(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=40,
        upscale=4,
        input_para=10,
        kernel_size=21,
        loop=8,
        pca_matrix_path=None,
        rate1=None,
        rate2=None,
        rate3=None,
        rate4=None,
        rate5=None
    ):
        super(NLCUnet, self).__init__()

        self.ksize = kernel_size
        self.loop = loop
        self.scale = upscale

        self.rate1=rate1
        self.rate2=rate2
        self.rate3=rate3
        self.rate4=rate4
        self.rate5=rate5

        self.Restorer = Restorer(nf=nf, nb=nb, scale=self.scale, input_para=input_para)

    def forward(self, lr):
        srs = []

        B, C, H, W = lr.shape

        if self.training is False:
            sr = self.Restorer(lr, rate1=self.rate1, rate2=self.rate2, rate3=self.rate3, rate4=self.rate4, rate5=self.rate5)
        if self.training is True:
            sr = self.Restorer(lr, rate1=None, rate2=None, rate3=None, rate4=None, rate5=None)

        srs.append(sr)
        return srs
