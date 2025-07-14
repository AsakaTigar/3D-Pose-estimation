import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from functools import partial

class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze操作的定义
        self.fc = nn.Sequential(  # Excitation操作的定义
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()  # 得到H和W的维度，在这两个维度上进行全局池化
        y = self.avg_pool(x).view(b, c)  # Squeeze操作的实现
        y = self.fc(y).view(b, c, 1, 1)  # Excitation操作的实现
        # 将y扩展到x相同大小的维度后进行赋权
        return x * y.expand_as(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Spatial_Forward(nn.Module):
    def __init__(self, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)

    def forward(self, x): # [b,t,p,c]
        b, f, p, c = x.shape  # b is batch size, f is number of frames, p is number of joints

        x = rearrange(x, 'b f p c  -> (b f) p  c', ) # [(b f) p c]

        x = self.Spatial_patch_to_embedding(x) # [(b f) p c]
        x = x + self.Spatial_pos_embed # [(b f) p c]
        x = self.pos_drop(x) # [(b f) p c]

        for blk in self.Spatial_blocks:
            x = blk(x) # [(b f) p c]

        x = self.Spatial_norm(x) # [(b f) p c]
        x = rearrange(x, '(b f) p c -> b f p c', f=f) # [b,t,p,c]
        return x


class FC_Relu(nn.Module):
    def __init__(self, in_features, out_features, num_frame, dropout=0.5):
        super().__init__()
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.num_frame = num_frame

    def forward(self, x):
        b, t, p, c = x.shape
        x = rearrange(x, 'b f p c  -> (b f) p  c', ) # [(b f) p c]

        x_0 = x[:, 0, :]
        y = self.dropout(self.relu(self.batch_norm(self.fc(x_0))))
        y = y.unsqueeze(1)
        for i in range(1, p):
            x_i = x[:, 1, :]
            x_i = self.dropout(self.relu(self.batch_norm(self.fc(x_i))))
            x_i = x_i.unsqueeze(1)
            y = torch.cat([y, x_i], dim=1)
        y = y.view(-1, self.num_frame, y.shape[-2], y.shape[-1])
        return y


class Temporal_Forward(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=32, embed_dim_ratio=32, depth=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.frame = num_frame
        self.num_joints = num_joints
        self.in_chans = in_chans
        embed_dim = embed_dim_ratio * num_joints
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.Temporal_norm1 = norm_layer(embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = x.view(-1, self.frame, self.num_joints * self.in_chans)
        b = x.shape[0]
        x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm1(x)
        x = self.weighted_mean(x)
        x = x.view(b, 1, self.num_joints, -1)
        return x


class Temporal_Mix(nn.Module):
    def __init__(self, in_frame, out_frame):
        super().__init__()
        self.weighted_mean = torch.nn.Conv1d(in_channels=in_frame, out_channels=out_frame, kernel_size=1)

    def forward(self, x):
        b, t, p, c = x.shape
        x = rearrange(x, 'b f p c  -> b f (p c)', )
        x = self.weighted_mean(x)
        x = rearrange(x, 'b f (p c)  -> b f p c', p=p)
        return x


class Pose3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.p17_SA1 = Spatial_Forward(num_joints=17, in_chans=2, embed_dim_ratio=16, depth=2,
                                        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        self.p17_SA2 = Spatial_Forward(num_joints=17, in_chans=16, embed_dim_ratio=16, depth=2,
                                       num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        self.p17_TM = Temporal_Mix(in_frame=args.number_of_frames, out_frame=args.number_of_frames//3)
        self.p11_SA = Spatial_Forward(num_joints=11, in_chans=16, embed_dim_ratio=32, depth=2,
                                        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)
        self.p11_TM = Temporal_Mix(in_frame=args.number_of_frames//3, out_frame=args.number_of_frames//9)
        self.p6_SA = Spatial_Forward(num_joints=6, in_chans=32, embed_dim_ratio=48, depth=2,
                                          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

        self.p6_CA = ChannelAttention(channel=args.number_of_frames//9)
        self.p6_TA = Temporal_Forward(num_frame=args.number_of_frames//9, num_joints=6, in_chans=48,
                                         embed_dim_ratio=48, depth=2,
                                         num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        self.p11_CA = ChannelAttention(channel=args.number_of_frames//3)
        self.p11_TA = Temporal_Forward(num_frame=args.number_of_frames//3, num_joints=11, in_chans=32,
                                           embed_dim_ratio=32, depth=4,
                                           num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        self.p17_CA = ChannelAttention(channel=args.number_of_frames)
        self.p17_TA = Temporal_Forward(num_frame=args.number_of_frames, num_joints=17, in_chans=16,
                                           embed_dim_ratio=16, depth=6,
                                           num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

        self.p6_RS = nn.Sequential(
            nn.LayerNorm(6*48),
            nn.Linear(6*48, 6*32),
        )
        self.p11_RS = nn.Sequential(
            nn.LayerNorm(11*32),
            nn.Linear(11*32, 11*16),
        )
        self.hidden = nn.Linear(2, 1)

        self.part0con = nn.Conv2d(3, 2, kernel_size=1)
        self.part1con = nn.Conv2d(3, 2, kernel_size=1)
        self.part2con = nn.Conv2d(3, 2, kernel_size=1)
        self.part3con = nn.Conv2d(2, 1, kernel_size=1)
        self.part4con = nn.Conv2d(3, 2, kernel_size=1)
        self.part5con = nn.Conv2d(3, 2, kernel_size=1)

        self.part0conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part1conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part2conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part3conT = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part4conT = nn.ConvTranspose2d(2, 3, kernel_size=1)
        self.part5conT = nn.ConvTranspose2d(2, 3, kernel_size=1)

        self.part0con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part1con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part2con2 = nn.Conv2d(2, 1, kernel_size=1)
        # self.part3con2 = nn.Conv2d(1, 1, kernel_size=1)
        self.part4con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part5con2 = nn.Conv2d(2, 1, kernel_size=1)

        self.part0conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part1conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part2conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        # self.part3conT2 = nn.ConvTranspose2d(1, 1, kernel_size=1)
        self.part4conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part5conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)

        self.part0conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part1conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part2conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part3conT3 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part4conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part5conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)

        self.head1 = nn.Sequential(
                nn.LayerNorm(17 * 16),
                nn.Linear(17 * 16, 17 * 3),
        )
        self.head2 = nn.Sequential(
            nn.LayerNorm(17 * 16),
            nn.Linear(17 * 16, 17 * 3),
        )
        self.head3 = nn.Sequential(
            nn.LayerNorm(17 * 32),
            nn.Linear(17 * 32, 17 * 3),
        )
        self.head4 = nn.Sequential(
            nn.LayerNorm(17 * 48),
            nn.Linear(17 * 48, 17 * 3),
        )


    def do17to11(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 1:4, :, :]
        part1 = x[:, 4:7, :, :]
        part2 = torch.stack((x[:, 0, :, :], x[:, 7, :, :], x[:, 8, :, :]), dim=1)
        part3 = x[:, 9:11, :, :]
        part4 = x[:, 11:14, :, :]
        part5 = x[:, 14:, :, :]
        part0 = self.part0con(part0)
        part1 = self.part1con(part1)
        part2 = self.part2con(part2)
        part3 = self.part3con(part3)
        part4 = self.part4con(part4)
        part5 = self.part5con(part5)
        x = torch.cat((part0, part1, part2, part3, part4, part5), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def do11to6(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 0:2, :, :]
        part1 = x[:, 2:4, :, :]
        part2 = x[:, 4:6, :, :]
        part3 = x[:, 6:7, :, :]
        part4 = x[:, 7:9, :, :]
        part5 = x[:, 9:, :, :]
        part0 = self.part0con2(part0)
        part1 = self.part1con2(part1)
        part2 = self.part2con2(part2)
        # part3 = self.part3con2(part3)
        part4 = self.part4con2(part4)
        part5 = self.part5con2(part5)
        x = torch.cat((part0, part1, part2, part3, part4, part5), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def do6to11(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 0:1, :, :]
        part1 = x[:, 1:2, :, :]
        part2 = x[:, 2:3, :, :]
        part3 = x[:, 3:4, :, :]
        part4 = x[:, 4:5, :, :]
        part5 = x[:, 5:, :, :]
        part0 = self.part0conT2(part0)
        part1 = self.part1conT2(part1)
        part2 = self.part2conT2(part2)
        # part3 = self.part3conT2(part3)
        part4 = self.part4conT2(part4)
        part5 = self.part5conT2(part5)
        x = torch.cat((part0, part1, part2, part3, part4, part5), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def do11to17(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 0:2, :, :]
        part1 = x[:, 2:4, :, :]
        part2 = x[:, 4:6, :, :]
        part3 = x[:, 6:7, :, :]
        part4 = x[:, 7:9, :, :]
        part5 = x[:, 9:, :, :]
        part0 = self.part0conT(part0)
        part1 = self.part1conT(part1)
        part2 = self.part2conT(part2)
        part3 = self.part3conT(part3)
        part4 = self.part4conT(part4)
        part5 = self.part5conT(part5)
        x = torch.cat((part0, part1, part2, part3, part4, part5), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def do6to17(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 0:1, :, :]
        part1 = x[:, 1:2, :, :]
        part2 = x[:, 2:3, :, :]
        part3 = x[:, 3:4, :, :]
        part4 = x[:, 4:5, :, :]
        part5 = x[:, 5:, :, :]
        part0 = self.part0conT3(part0)
        part1 = self.part1conT3(part1)
        part2 = self.part2conT3(part2)
        part3 = self.part3conT3(part3)
        part4 = self.part4conT3(part4)
        part5 = self.part5conT3(part5)
        x = torch.cat((part0, part1, part2, part3, part4, part5), 1)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        B, _, _, _ = x.shape

        p17 = self.p17_SA1(x)  # [B,T,17,2] -> [B,T,17,16]
        p17 = self.p17_SA2(p17)  # [B,T,17,16] -> [B,T,17,16]

        _p17 = self.p17_TM(p17) # [B,T,17,16] -> [B,T/3,17,16]
        p11 = self.do17to11(_p17)  # [B,T/3,17,16] -> [B,T/3,11,16]
        p11 = self.p11_SA(p11)  # [B,T/3,11,16] -> [B,T/3,11,32]

        _p11 = self.p11_TM(p11) # [B,T/3,11,32] -> [B,T/9,11,32]
        p6 = self.do11to6(_p11)  # [B,T/9,11,32] -> [B,T/9,6,32]
        p6 = self.p6_SA(p6) # [B,T/9,6,32] -> [B,T/9,6,48]

        p6 = self.p6_CA(p6)  # [B,T/9,6,48] -> [B,T/9,6,48]
        p6 = self.p6_TA(p6)  # [B,T/9,6,48] -> [B,1,6,48]
        p6_loss = self.do6to17(p6) # [B,1,17,48]
        p6_loss = self.head4(p6_loss.view(B, 1, -1)).view(B, 1, 17, 3)  # [B,1,17,48] -> [B,1,17,3]

        p11 = self.p11_CA(p11)  # [B,T/3,11,32] -> [B,T/3,11,32]
        p11 = self.p11_TA(p11)  # [B,T/3,11,32] -> [B,1,11,32]
        p11_loss = self.do11to17(p11) # [B,1,17,32]
        p11_loss = self.head3(p11_loss.view(B, 1, -1)).view(B, 1, 17, 3)  # [B,1,17,32] -> [B,1,17,3]

        p17 = self.p17_CA(p17)  # [B,T,17,16] -> [B,T,17,16]
        p17 = self.p17_TA(p17)  # [B,T,17,16] -> [B,1,17,16]
        p17_loss = self.head2(p17.view(B, 1, -1)).view(B, 1, 17, 3)  # [B,1,17,16] -> [B,1,17,3]

        p6 = self.p6_RS(p6.view(B, 1, -1)).view(B, 1, 6, 32) # [B,1,6,48] -> [B,1,6,32]
        p6 = self.do6to11(p6) # [B,1,6,32] -> [B,1,11,32]
        p11 = torch.stack((p6, p11)) # [2,B,1,11,32]
        p11 = self.hidden(p11.permute(1, 2, 3, 4, 0)) # [1,B,1,11,32]
        a, b, c, d, _ = p11.shape
        p11 = p11.view(a, b, c, d) # [B,1,11,32]

        p11 = self.p11_RS(p11.view(B, 1, -1)).view(B, 1, 11, 16) # [B,1,11,32] -> [B,1,11,16]
        p11 = self.do11to17(p11) # [B,1,11,16] -> [B,1,17,16]
        p17 = torch.stack((p11, p17)) # [2,B,1,17,16]
        p17 = self.hidden(p17.permute(1, 2, 3, 4, 0)) # [1,B,1,17,16]
        a, b, c, d, _ = p17.shape
        p17 = p17.view(a, b, c, d) # [B,1,17,16]

        y = self.head1(p17.view(B, 1, -1)) # [B,1,17,16] -> [B,1,51]
        y = y.view(B, 1, 17, 3)

        return y, p17_loss, p11_loss, p6_loss
