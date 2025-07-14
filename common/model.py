import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from functools import partial

class CrossAttention(nn.Module):
    def __init__(self, num_joints, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_joints = num_joints
        self.embed_dim = embed_dim

    def forward(self, query, key, value):
        # Reshape the input to [seq_len, batch_size, embed_dim]
        query = query.permute(2, 0, 1, 3).flatten(2, 3)  # [17, b, 16]
        key = key.permute(2, 0, 1, 3).flatten(2, 3)      # [17, b, 16]
        value = value.permute(2, 0, 1, 3).flatten(2, 3)  # [17, b, 16]
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)
        # Reshape back to [b, 1, 17, 16]
        attn_output = attn_output.view(self.num_joints, -1, 1, self.embed_dim).permute(1, 2, 0, 3)
        return attn_output

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, num_joints, embed_dim, num_heads):
        super(MultiScaleFeatureFusion, self).__init__()
        self.cross_attn1 = CrossAttention(num_joints, embed_dim, num_heads)
        self.cross_attn2 = CrossAttention(num_joints, embed_dim, num_heads)
        self.cross_attn3 = CrossAttention(num_joints, embed_dim, num_heads)
        self.hidden = nn.Linear(3, 1)

    def forward(self, feat1, feat2, feat3):
        # Apply cross-attention between features
        fused_feat1 = self.cross_attn1(feat1, feat2, feat3)
        fused_feat2 = self.cross_attn2(feat2, feat1, feat3)
        fused_feat3 = self.cross_attn3(feat3, feat1, feat2)

        fused_features = torch.stack((fused_feat1, fused_feat2, fused_feat3))
        fused_features = self.hidden(fused_features.permute(1, 2, 3, 4, 0)) # [1,B,1,11,32]
        a, b, c, d, _ = fused_features.shape
        fused_features = fused_features.view(a, b, c, d)
        return fused_features
    
class BlockAttention(nn.Module):
    def __init__(self, num_blocks, top_k=3):
        super(BlockAttention, self).__init__()
        self.num_blocks = num_blocks
        self.top_k = top_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_blocks, num_blocks, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(num_blocks, num_blocks, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        _, top_k_indices = torch.topk(y, k=self.top_k, dim=1)
        top_k_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        topk_x = torch.gather(x, 1, top_k_indices)
        return topk_x


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

class Spatial_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., skeleton_adjacency=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.skeleton_adjacency = skeleton_adjacency
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Linear(dim, 1)  # 门控线性层

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Ensure skeleton_adjacency has the correct shape
        if self.skeleton_adjacency is not None:
            skeleton_adjacency = self.skeleton_adjacency.to(attn.device).float().unsqueeze(0).unsqueeze(0)
            gate = self.sigmoid(self.gate(x))  # 计算门控值 [B, P, 1]
            gate = gate.permute(0, 2, 1).unsqueeze(2)  # [B, 1, 1, P]
            attn = attn * (1 - gate) + skeleton_adjacency * gate  # 混合注意力

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Temporal_Attention(nn.Module):
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
    

class Spatial_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skeleton_adjacency=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Spatial_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, skeleton_adjacency=skeleton_adjacency)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class Temporal_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Temporal_Attention(
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
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, skeleton_adjacency=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Spatial_Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skeleton_adjacency=skeleton_adjacency)
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

class Temporal_Forward(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=32, embed_dim_ratio=32, depth=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, block_size=3, top_k=6):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.frame = num_frame
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.num_heads = num_heads

        self.block_size = block_size
        self.num_blocks = self.frame // self.block_size
        self.top_k = top_k
        self.block_attention = BlockAttention(num_blocks=self.num_blocks, top_k=self.top_k)

        self.weighted_mean1 = torch.nn.Conv1d(in_channels=self.block_size, out_channels=1, kernel_size=1)

        embed_dim = embed_dim_ratio * num_joints
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Temporal_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.temporal_norm = norm_layer(embed_dim)

        self.weighted_mean2 = torch.nn.Conv1d(in_channels=self.top_k, out_channels=1, kernel_size=1)
        self.time_shift = nn.ZeroPad2d((0,0,1,0))

        self.embed_dim = embed_dim


    def forward(self, x):
        x = x.view(-1, self.frame, self.num_joints * self.in_chans)
        x = x + self.temporal_pos_embed
        x = torch.cat([self.time_shift(x)[:,:self.frame,:self.embed_dim//2], x[:,:self.frame,self.embed_dim//2:]], dim=2)

        B, T, C = x.shape
        xblocks = x.reshape(B, self.num_blocks, self.block_size, C)
        # xblocks = xblocks.view(-1, self.block_size, C)
        # xblocks = self.weighted_mean1(xblocks)
        xblocks = xblocks.mean(dim=2)  # [B, num_blocks, C]
        xblocks = xblocks.view(B, self.num_blocks, 1, C)
        # xblocks = xblocks.view(b, self.num_blocks, 1 * c)

        topk_x = self.block_attention(xblocks)
        topk_x = topk_x.view(B, self.top_k, -1)
        topk_x = self.pos_drop(topk_x)
        for blk in self.blocks:
            topk_x = blk(topk_x)
        topk_x = self.temporal_norm(topk_x)
        topk_x = self.weighted_mean2(topk_x)
        topk_x = topk_x.view(B, 1, self.num_joints, -1)
        return topk_x


class Pose3D(nn.Module):
    def __init__(self, args):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # 定义骨骼连接矩阵
        skeleton_adjacency_17 = torch.zeros((17, 17), dtype=torch.int).to(device)

        # 根据连接关系设置矩阵值
        # 0 (骨盆) 连接到 1 (右髋) 和 4 (左髋)
        skeleton_adjacency_17[0, 1] = 1
        skeleton_adjacency_17[1, 0] = 1
        skeleton_adjacency_17[0, 4] = 1
        skeleton_adjacency_17[4, 0] = 1

        # 1 (右髋) 连接到 2 (右膝)
        skeleton_adjacency_17[1, 2] = 1
        skeleton_adjacency_17[2, 1] = 1

        # 2 (右膝) 连接到 3 (右脚踝)
        skeleton_adjacency_17[2, 3] = 1
        skeleton_adjacency_17[3, 2] = 1

        # 4 (左髋) 连接到 5 (左膝)
        skeleton_adjacency_17[4, 5] = 1
        skeleton_adjacency_17[5, 4] = 1

        # 5 (左膝) 连接到 6 (左脚踝)
        skeleton_adjacency_17[5, 6] = 1
        skeleton_adjacency_17[6, 5] = 1

        # 0 (骨盆) 连接到 7 (脊柱)
        skeleton_adjacency_17[0, 7] = 1
        skeleton_adjacency_17[7, 0] = 1

        # 7 (脊柱) 连接到 8 (胸部)
        skeleton_adjacency_17[7, 8] = 1
        skeleton_adjacency_17[8, 7] = 1

        # 8 (胸部) 连接到 11 (左肩) 和 14 (右肩)
        skeleton_adjacency_17[8, 11] = 1
        skeleton_adjacency_17[11, 8] = 1
        skeleton_adjacency_17[8, 14] = 1
        skeleton_adjacency_17[14, 8] = 1

        # 11 (左肩) 连接到 12 (左肘)
        skeleton_adjacency_17[11, 12] = 1
        skeleton_adjacency_17[12, 11] = 1

        # 12 (左肘) 连接到 13 (左腕)
        skeleton_adjacency_17[12, 13] = 1
        skeleton_adjacency_17[13, 12] = 1

        # 14 (右肩) 连接到 15 (右肘)
        skeleton_adjacency_17[14, 15] = 1
        skeleton_adjacency_17[15, 14] = 1

        # 15 (右肘) 连接到 16 (右腕)
        skeleton_adjacency_17[15, 16] = 1
        skeleton_adjacency_17[16, 15] = 1

        # 8 (胸部) 连接到 9 (颈部)
        skeleton_adjacency_17[8, 9] = 1
        skeleton_adjacency_17[9, 8] = 1

        # 9 (颈部) 连接到 10 (头部)
        skeleton_adjacency_17[9, 10] = 1
        skeleton_adjacency_17[10, 9] = 1

        self.skeleton_adjacency_17 = skeleton_adjacency_17

        skeleton_adjacency_11 = torch.zeros((11, 11), dtype=torch.int).to(device)

        # Connections based on the provided parent-child relationships
        # 0 connected to 1
        skeleton_adjacency_11[0, 1] = 1
        skeleton_adjacency_11[1, 0] = 1

        # 2 connected to 3
        skeleton_adjacency_11[2, 3] = 1
        skeleton_adjacency_11[3, 2] = 1

        # 4 connected to 5
        skeleton_adjacency_11[4, 5] = 1
        skeleton_adjacency_11[5, 4] = 1

        # 4 connected to 0
        skeleton_adjacency_11[4, 0] = 1
        skeleton_adjacency_11[0, 4] = 1

        # 4 connected to 2
        skeleton_adjacency_11[4, 2] = 1
        skeleton_adjacency_11[2, 4] = 1

        # 5 connected to 6
        skeleton_adjacency_11[5, 6] = 1
        skeleton_adjacency_11[6, 5] = 1

        # 5 connected to 7
        skeleton_adjacency_11[5, 7] = 1
        skeleton_adjacency_11[7, 5] = 1

        # 7 connected to 8
        skeleton_adjacency_11[7, 8] = 1
        skeleton_adjacency_11[8, 7] = 1

        # 5 connected to 9
        skeleton_adjacency_11[5, 9] = 1
        skeleton_adjacency_11[9, 5] = 1

        # 9 connected to 10
        skeleton_adjacency_11[9, 10] = 1
        skeleton_adjacency_11[10, 9] = 1

        self.skeleton_adjacency_11 = skeleton_adjacency_11

        # Define the skeleton adjacency matrix for 6 keypoints
        skeleton_adjacency_6 = torch.zeros((6, 6), dtype=torch.int).to(device)

        # Connections based on the provided parent-child relationships
        # 0 connected to 2
        skeleton_adjacency_6[0, 2] = 1
        skeleton_adjacency_6[2, 0] = 1

        # 1 connected to 2
        skeleton_adjacency_6[1, 2] = 1
        skeleton_adjacency_6[2, 1] = 1

        # 2 connected to 3
        skeleton_adjacency_6[2, 3] = 1
        skeleton_adjacency_6[3, 2] = 1

        # 2 connected to 4
        skeleton_adjacency_6[2, 4] = 1
        skeleton_adjacency_6[4, 2] = 1

        # 2 connected to 5
        skeleton_adjacency_6[2, 5] = 1
        skeleton_adjacency_6[5, 2] = 1

        self.skeleton_adjacency_6 = skeleton_adjacency_6

        self.forward1_1 = Spatial_Forward(num_joints=17, in_chans=2, embed_dim_ratio=16, depth=2,
                                        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1, skeleton_adjacency=self.skeleton_adjacency_17)
        # self.forward2_1 = FCN(in_chans=2, embed_dim_ratio=16)
        self.forward2_1 = nn.Linear(2, 16)
        self.forward2_2 = Spatial_Forward(num_joints=11, in_chans=16, embed_dim_ratio=24, depth=2,
                                        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1, skeleton_adjacency=self.skeleton_adjacency_11)
        # self.forward3_1 = FCN(in_chans=2, embed_dim_ratio=24)
        self.forward3_1 = nn.Linear(2, 24)
        self.forward3_2 = Spatial_Forward(num_joints=6, in_chans=24, embed_dim_ratio=32, depth=2,
                                          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1, skeleton_adjacency=self.skeleton_adjacency_6)
        # self.forward4_1 = ChannelAttention(channel=args.number_of_frames)
        self.forward4_2 = Temporal_Forward(num_frame=args.number_of_frames, num_joints=6, in_chans=32,
                                         embed_dim_ratio=32, depth=4,
                                         num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        # self.forward4_2_2 = Temporal_Forward(num_frame=args.number_of_frames, num_joints=6, in_chans=32,
        #                                  embed_dim_ratio=32, depth=1,
        #                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        # self.forward4_3 = Spatial_Forward(num_joints=17, in_chans=32, embed_dim_ratio=16, depth=2,
        #                                   num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)
        # self.forward5_1 = ChannelAttention(channel=args.number_of_frames)
        self.forward5_2 = Temporal_Forward(num_frame=args.number_of_frames, num_joints=11, in_chans=24,
                                         embed_dim_ratio=24, depth=4,
                                         num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        # self.forward5_2_2 = Temporal_Forward(num_frame=args.number_of_frames, num_joints=11, in_chans=24,
        #                                  embed_dim_ratio=24, depth=2,
        #                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        # self.forward5_3 = Spatial_Forward(num_joints=17, in_chans=24, embed_dim_ratio=16, depth=2,
        #                                   num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)
        # self.forward6_1 = ChannelAttention(channel=args.number_of_frames)
        self.forward6_2 = Temporal_Forward(num_frame=args.number_of_frames, num_joints=17, in_chans=16,
                                         embed_dim_ratio=16, depth=4,
                                         num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
        # self.forward6_2_2 = Temporal_Forward(num_frame=args.number_of_frames, num_joints=17, in_chans=16,
        #                                  embed_dim_ratio=16, depth=2,
        #                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

        # self.Mix1 = Mix(num_frame=args.number_of_frames, num_joints=6, in_chans=32, embed_dim_ratio=32)
        # self.Mix2 = Mix(num_frame=args.number_of_frames, num_joints=11, in_chans=24, embed_dim_ratio=24)
        # self.Mix3 = Mix(num_frame=args.number_of_frames, num_joints=17, in_chans=16, embed_dim_ratio=16)
        
        # Initialize fusion module
        self.fusion_module1 = MultiScaleFeatureFusion(num_joints=17, embed_dim=16, num_heads=4)
        self.fusion_module2 = MultiScaleFeatureFusion(num_joints=11, embed_dim=24, num_heads=4)
        # self.fusion_module3 = MultiScaleFeatureFusion(num_joints=6, embed_dim=32, num_heads=4)

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

        self.part0conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part1conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part2conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part3conT3 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part4conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)
        self.part5conT3 = nn.ConvTranspose2d(1, 3, kernel_size=1)

        self.part0con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part1con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part2con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part3con2 = nn.Conv2d(1, 1, kernel_size=1)
        self.part4con2 = nn.Conv2d(2, 1, kernel_size=1)
        self.part5con2 = nn.Conv2d(2, 1, kernel_size=1)

        self.part0conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part1conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part2conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part3conT2 = nn.ConvTranspose2d(1, 1, kernel_size=1)
        self.part4conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)
        self.part5conT2 = nn.ConvTranspose2d(1, 2, kernel_size=1)

        self.pre_part0con3 = nn.Conv2d(3, 1, kernel_size=1)
        self.pre_part1con3 = nn.Conv2d(3, 1, kernel_size=1)
        self.pre_part2con3 = nn.Conv2d(3, 1, kernel_size=1)
        self.pre_part3con3 = nn.Conv2d(2, 1, kernel_size=1)
        self.pre_part4con3 = nn.Conv2d(3, 1, kernel_size=1)
        self.pre_part5con3 = nn.Conv2d(3, 1, kernel_size=1)

        self.head0 = nn.Sequential(
            nn.LayerNorm(17 * 16),
            nn.Linear(17 * 16, 17 * 3),
        )
        self.head1 = nn.Sequential(
            nn.LayerNorm(17 * 32),
            nn.Linear(17 * 32, 17 * 16),
        )
        self.head2 = nn.Sequential(
            nn.LayerNorm(11 * 32),
            nn.Linear(11 * 32, 11 * 24),
        )
        self.head3 = nn.Sequential(
            nn.LayerNorm(17 * 24),
            nn.Linear(17 * 24, 17 * 16),
        )
        self.head4 = nn.Sequential(
            nn.LayerNorm(17 * 32),
            nn.Linear(17 * 32, 17 * 3),
        )
        self.head5 = nn.Sequential(
            nn.LayerNorm(11 * 16),
            nn.Linear(11 * 16, 11 * 24),
        )
        self.head6 = nn.Sequential(
            nn.LayerNorm(17 * 24),
            nn.Linear(17 * 24, 17 * 3),
        )
        # self.head7 = nn.Sequential(
        #     nn.LayerNorm(17 * 16),
        #     nn.Linear(17 * 16, 17 * 3),
        # )

        self.p6_RS = nn.Sequential(
            nn.LayerNorm(6*32),
            nn.Linear(6*32, 6*24),
        )
        self.p11_RS = nn.Sequential(
            nn.LayerNorm(11*24),
            nn.Linear(11*24, 11*16),
        )

        self.hidden1 = nn.Linear(2, 1)
        self.hidden2 = nn.Linear(2, 1)

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
    
    def do17to6(self, x):
        x = x.permute(0, 2, 1, 3)
        part0 = x[:, 1:4, :, :]
        part1 = x[:, 4:7, :, :]
        part2 = torch.stack((x[:, 0, :, :], x[:, 7, :, :], x[:, 8, :, :]), dim=1)
        part3 = x[:, 9:11, :, :]
        part4 = x[:, 11:14, :, :]
        part5 = x[:, 14:, :, :]
        part0 = self.pre_part0con3(part0)
        part1 = self.pre_part1con3(part1)
        part2 = self.pre_part2con3(part2)
        part3 = self.pre_part3con3(part3)
        part4 = self.pre_part4con3(part4)
        part5 = self.pre_part5con3(part5)
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
        part3 = self.part3con2(part3)
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
        part3 = self.part3conT2(part3)
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
        x1 = self.forward1_1(x)  # [16,9,17,2] -> [16,9,17,16]

        x2 = self.do17to11(x1)  # [16,9,17,16] -> [16,9,11,16]
        pre_x2 = self.do17to11(x) # [16,9,17,2] -> [16,9,11,2]
        pre_x2 = self.forward2_1(pre_x2)  # [16,9,11,2] -> [16,9,11,16]
        x2 = torch.stack((pre_x2, x2)) # [2,16,9,11,16]
        x2 = self.hidden1(x2.permute(1, 2, 3, 4, 0)) # [1,16,9,11,16]
        a, b, c, d, _ = x2.shape
        x2 = x2.view(a, b, c, d) # [16,9,11,16]
        x2 = self.forward2_2(x2)  # [16,9,11,16] -> [16,9,11,24]

        x3 = self.do11to6(x2)  # [16,9,11,24] -> [16,9,6,24]
        pre_x3 = self.do17to6(x) # [16,9,17,2] -> [16,9,6,2]
        pre_x3 = self.forward3_1(pre_x3)  # [16,9,6,2] -> [16,9,6,24]
        x3 = torch.stack((pre_x3, x3)) # [2,16,9,6,24]
        x3 = self.hidden1(x3.permute(1, 2, 3, 4, 0)) # [1,16,9,6,24]
        a, b, c, d, _ = x3.shape
        x3 = x3.view(a, b, c, d) # [16,9,6,24]
        x3 = self.forward3_2(x3) # [16,9,6,24] -> [16,9,6,32]

        b, t, p, c = x.shape
        # x3 = self.forward4_1(x3)  # [16,9,6,32] -> [16,9,6,32]
        x3 = self.forward4_2(x3)  # [16,9,6,32] -> [16,1,6,32]
        x3_17 = self.do6to17(x3) # [16,1,17,32]
        x3_loss = self.head4(x3_17.view(b, 1, -1)).view(b, 1, 17, 3)  # [B,1,17,32] -> [B,1,17,3]
        x3_17 = self.head1(x3_17.view(b, 1, -1)).view(b, 1, 17, 16)  # [B,1,17,32] -> [B,1,17,16]
        x3_11 = self.do6to11(x3) # [16,1,11,32]
        x3_11 = self.head2(x3_11.view(b, 1, -1)).view(b, 1, 11, 24)  # [B,1,11,32] -> [B,1,11,24]

        # x2 = self.forward5_1(x2)  # [16,9,11,24] -> [16,9,11,24]
        x2 = self.forward5_2(x2) # [16,9,11,24] -> [16,1,11,24]
        x2_17 = self.do11to17(x2) # [16,1,11,24] -> [16,1,17,24]
        x2_loss = self.head6(x2_17.view(b, 1, -1)).view(b, 1, 17, 3)  # [B,1,17,16] -> [B,1,17,3]
        x2_17 = self.head3(x2_17.view(b, 1, -1)).view(b, 1, 17, 16)  # [16,1,17,24] -> [16,1,17,16]
        # x2_6 = self.do11to6(x2) # [16,1,11,24] -> [16,1,6,24]
        # x2_6 = self.head4(x2_6.view(b, 1, -1)).view(b, 1, 6, 32)  # [16,1,6,24] -> [16,1,6,32]

        # x1 = self.forward6_1(x1)  # [16,9,17,16] -> [16,9,17,16]
        x1 = self.forward6_2(x1)  # [16,9,17,16] -> [16,1,17,16]
        x1_11 = self.do17to11(x1) # [16,9,17,16] -> [16,1,11,16]
        x1_11 = self.head5(x1_11.view(b, 1, -1)).view(b, 1, 11, 24)  # [16,1,11,16] -> [16,1,11,24]
        # x1_6 = self.do17to6(x1) # [16,9,17,16] -> [16,9,6,16]
        # x1_6 = self.head6(x1_6.view(b, 1, -1)).view(b, 1, 6, 32)  # [16,1,6,16] -> [16,1,6,32]
 
        # Perform fusion
        x1 = self.fusion_module1(x1, x2_17, x3_17) # [16,1,17,16]
        x2 = self.fusion_module2(x1_11, x2, x3_11) # [16,1,11,24]
        # x3 = self.fusion_module3(x1_6, x2_6, x3) # [16,1,6,32]

        # x1_loss = self.head7(x1.view(b, 1, -1)).view(b, 1, 17, 3)  # [B,1,17,16] -> [B,1,17,3]
        

        b, t, p, c = x.shape
        x3 = self.p6_RS(x3.view(b, 1, -1)).view(b, 1, 6, 24) # [B,1,6,32] -> [B,1,6,24]
        x3 = self.do6to11(x3) # [B,1,6,24] -> [B,1,11,24]
        x2 = torch.stack((x3, x2)) # [2,B,1,11,24]
        x2 = self.hidden2(x2.permute(1, 2, 3, 4, 0)) # [1,B,1,11,24]
        a, b, c, d, _ = x2.shape
        x2 = x2.view(a, b, c, d) # [B,1,11,24]

        b, t, p, c = x.shape
        x2 = self.p11_RS(x2.view(b, 1, -1)).view(b, 1, 11, 16) # [B,1,11,24] -> [B,1,11,16]
        x2 = self.do11to17(x2) # [B,1,11,16] -> [B,1,17,16]
        x1 = torch.stack((x2, x1)) # [2,B,1,17,16]
        x1 = self.hidden2(x1.permute(1, 2, 3, 4, 0)) # [1,B,1,17,16]
        a, b, c, d, _ = x1.shape
        x1 = x1.view(a, b, c, d) # [B,1,17,16]
        
        b, t, p, c = x.shape
        y = self.head0(x1.reshape(b, 1, -1)) # [16,1,17,16] -> [16,1,51]
        y = y.view(b, 1, 17, 3)

        
        return y, x2_loss, x3_loss
