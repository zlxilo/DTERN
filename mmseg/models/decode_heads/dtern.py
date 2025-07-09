
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .transformer_module import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from mmseg.models.utils import SelfAttentionBlockWithTime,CAM
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from einops.layers.torch import Rearrange

def get_sim_mask(query,key,threshold=0.95):
    '''
        input:
            query: [B, 1,C, h, w]
            key: [B, t,C, h, w]
        output:
            sim: [B, t, h, w, h, w]
            mask: [B, t, h, w, h, w]
    '''
    b,_,_,h,w = query.shape
    query = query.flatten(3).transpose(-1, -2)  # [B,1, h*w, C]
    key = key.flatten(3).transpose(-1, -2)  # [B, t,h*w, C]
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    sim = query @ key.transpose(-1, -2)  # [B,t, h*w, h*w]
    mask_kq = (sim >= threshold).sum(dim=-1) > 0  # [B, t, h*w],
    mask_qk = (sim >= threshold).sum(dim=-2) > 0  # [B, t, h*w], 
    mask_kq = mask_kq.view(b,-1,h,w) # [B,t, h, w]
    mask_qk = mask_qk.view(b,-1,h,w) # [B,t, h, w]
    return sim,mask_qk,mask_kq


def partition(x, patch_size): 
    """
    Args:
        x: (B, H, W, C)
        patch_size (int): patch size

    Returns:
        patches: (num_patches*B, patch_size, patch_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, patch_size, patch_size, C)
    return patches


def reverse(patches, patch_size, H, W): 
    """
    Args:
        patches: (num_patches*B, patch_size, patch_size, C)
        patch_size (int): Patch size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(patches.shape[0] / (H * W / patch_size / patch_size))
    x = patches.view(B, H // patch_size, W // patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,need_dw=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.need_dw = need_dw
        if need_dw:
            self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.need_dw:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchProjection(nn.Module):
    """ Patch Projection Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Cluster_Block(nn.Module):
    def __init__(self, dim,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,backbone='b1',cityscape=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)              
        dw_kernel_size = [11,9,7,5]
        no_local = False
        no_global = False
        if backbone == 'b0' or cityscape:
            dw_kernel_size= [13,11,9,7]
        self.cam = CAM(dim, (15, 10), nn.BatchNorm2d, 11,dw_kernel_size=dw_kernel_size,no_local=no_local,no_global=no_global)
        self.Clustering = nn.Sequential(
                nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
                Rearrange("b c h w -> b c (h w)")
        )


    def forward(self,x,H,W):
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        out = self.cam(x)
        return self.Clustering(out)

class Cluster_layer(nn.Module):
    def __init__(self, dim, num_heads,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,t=3,inner_center = True,backbone='b1',cityscape=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        self.inner_center = inner_center

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.clustering = Cluster_Block(dim,num_clusters,mlp_ratio=mlp_ratio,backbone=backbone,cityscape=cityscape) # 

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, act_layer=act_layer, drop=drop)
        self.prompt1 = torch.nn.parameter.Parameter(torch.randn(num_clusters, requires_grad=True)) 
        self.top_down_transform1 = torch.nn.parameter.Parameter(torch.eye(num_clusters), requires_grad=True)
        self.prompt2 = torch.nn.parameter.Parameter(torch.randn(num_clusters, requires_grad=True)) 
        self.top_down_transform2 = torch.nn.parameter.Parameter(torch.eye(num_clusters), requires_grad=True)

    def forward(self,x,H,W,z=None,mem=None,t=3):
        '''
            x:[b,n,c]
            mem:[bt,n,c]
        '''
        res = x
        assigned_results = None
        if mem is not None:
            x = torch.cat([mem.view(-1,t,mem.shape[-2],mem.shape[-1]),x.unsqueeze(1)],dim=1) #[b,t,n,c]
            x = x.flatten(0,1) #[bt,n,c]
            t = t+1
        else:
            t = 1
    
        # CAM
        cluster_x_z = self.clustering(x,H,W) # [bt,num_clusters,n]
        cluster_x_z = rearrange(cluster_x_z,"(b t) c n -> b t c n",b=res.shape[0])

        cluster_x_z = cluster_x_z.permute(0,2,1,3).contiguous().flatten(2) # [b,num_clusters,tn]
        
        #SAM
        if z is not None:
            z = rearrange(z,'b c n -> b n c') #(b,tn,num_clusters)
            cluster_x_z = rearrange(cluster_x_z,'b c n -> b n c') #(b,tn,num_clusters)
            # select:
            cos_sim = F.normalize(z, dim=-1) @ F.normalize(self.prompt1[None, ..., None], dim=1)  # B, N, 1 
            mask = cos_sim.clamp(0, 1)  
            z = z * mask    
            z = z @ self.top_down_transform1 
            cos_sim = F.normalize(cluster_x_z, dim=-1) @ F.normalize(self.prompt2[None, ..., None], dim=1)  # B, N, 1 
            mask = cos_sim.clamp(0, 1)
            cluster_x_z = cluster_x_z * mask

            cluster_x_z = cluster_x_z @ self.top_down_transform2

            cluster_x_z = (cluster_x_z + z)/2.
            cluster_x_z = rearrange(cluster_x_z,"b n c -> b c n")

        if self.inner_center:
            center = rearrange(cluster_x_z,"b c (t n) -> b t c n",t = t)
            assigned_results = center.clone()
            center = center.softmax(dim=-1) 
            center_x = rearrange(x,"(b t) n c -> b t n c",t=t)
            center = center @ center_x 
            cos_sim = torch.sigmoid( 
                (self.sim_beta + self.sim_alpha * F.cosine_similarity(center[:,-1].unsqueeze(1),center[:,:-1],dim=-1))
            ) #[b,t,num_cluster]
            C_in = center[:,-1] + (cos_sim.unsqueeze(-1) * center[:,:-1]).sum(dim=1) 
        
        # FEM
        C_in = self.norm1(C_in)
        src = rearrange(res,"b n c -> n b c")
        mem = rearrange(C_in, "b n c -> n b c")
        out,_ = F.multi_head_attention_forward(
            query=src,
            key=mem,
            value=mem,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q.weight,
            k_proj_weight=self.k.weight,
            v_proj_weight=self.v.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_drop,
            out_proj_weight=self.proj.weight,
            out_proj_bias=self.proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=not self.training
        )
        out = rearrange(out,"n b c -> b n c")
        out = self.proj_drop(out)
        
        out = res + self.norm1(out)
        out = out + self.norm1(self.mlp(out,H,W))
        return out,cluster_x_z,assigned_results
    

class GTEM(nn.Module):
    def __init__(self, dim, num_heads=8, num_classes=150, qkv_bias=True, qk_scale=None, T=3):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.FCC = nn.Parameter(torch.zeros(1, T, num_classes, dim)) 
        
        self.VCC = nn.Parameter(torch.zeros(1, num_classes, dim)) 
        
        self.cross_attn = SelfAttentionBlockWithTime(
            key_in_channels=dim,
            query_in_channels=dim,
            channels=dim,
            out_channels=dim,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=None,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        
        self.T = T
        trunc_normal_(self.FCC, std=.02)
        trunc_normal_(self.VCC, std=.02)
        self.apply(self._init_weights) 
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
             
    def forward(self, x):
        b, t, c, h, w = x.size()
        res = x
        assert t == self.T, "Input tensor has wrong time"
        x = x.flatten(3).transpose(-1, -2) #[b,t,n,c]
        B, T, N, C = x.shape
        FCC = self.FCC.expand(B, -1, -1, -1) # [B, T, num_classes, dim]
        VCC = self.VCC.unsqueeze(1).expand(B, T, -1, -1) 
        
        # B, T, N, C = x.shape
        q = self.q(x).reshape(B, T, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.k(FCC).unsqueeze(1).reshape(B, T, self.num_classes, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) #[B, T, num_heads, num_classes, dim]
        
        k = k * self.scale
        attn = (k @ q.transpose(-2, -1)).squeeze(2).transpose(-2, -1) 
        
        out_cls_mid = attn.permute(0, 1, 3, 2).reshape(b, t, -1, h, w)
        x_cls = out_cls_mid.flatten(3).transpose(-1, -2)   # [b,t,n,c]
        x_cls = x_cls.softmax(dim=-1)
        cls = x_cls @ VCC
        cls = cls.permute(0, 1, 3, 2).reshape(b, t, c, h, w) 
        x2 = self.cross_attn(res, cls)
        out = res + x2 

        return out_cls_mid,out

class DTERN(nn.Module):
    def __init__(self,dim=[64, 128, 320, 512], num_layers=1, t=3, time_decoder_layer=3,embedding_dim=256,num_classes = 124,
                 cross_method='CAT',ratio_fusio = False,num_clusters=150,
                 backbone='b1',cityscape=False):
        super().__init__()
        dim = dim[::-1]
        self.dim=dim
        self.pre_isa_blocks = nn.ModuleList()
        self.cpa_blocks = nn.ModuleList()
        self.post_isa_blocks = nn.ModuleList()
        self.tmp_blocks = nn.ModuleList()
        self.conv_t_out = nn.ModuleList()
        self.embedding_dim = embedding_dim
        num_heads = [2,4,8,16]
        self.num_layers = num_layers
        self.patch_size = 15
        self.convs = nn.ModuleList()
        self.sub_convs = nn.ModuleList()
        self.cross_method = cross_method
        self.t = 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.ratio_fusio = False
        if not ratio_fusio:
            for idx in range(4):
                self.convs.append(
                        ConvModule(
                        in_channels=dim[idx],
                        out_channels=embedding_dim,
                        kernel_size=1,
                        stride=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True),
                        act_cfg=dict(type='ReLU'))
                    )        
        
        self.fusion_conv = ConvModule(
                            in_channels=embedding_dim*4,
                            out_channels=embedding_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.cluster_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.cluster_blocks.append(
                Cluster_layer(
                    dim=embedding_dim,
                    num_heads=num_heads[i],
                    num_clusters=num_clusters,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,t=3,backbone=backbone,cityscape=cityscape)
            )

        self.gtem = GTEM(dim=embedding_dim, num_heads=1, num_classes=num_classes, T=self.t)
        
    def forward(self, query_frame, supp_frame,t = None,img_metas=None):
        query_frame=query_frame[::-1] 
        supp_frame=supp_frame[::-1]
        out_supp_frame = []
        out_memory_frames = []
        T_pre = query_frame[0].shape[1]
        T_tg = supp_frame[0].shape[1]

        tg_size = supp_frame[-2].shape[-2:]
        if not self.ratio_fusio:
            k_ratio = -2
            for idx in range(len(supp_frame)):
                x = supp_frame[idx].flatten(0,1)
                memory = query_frame[idx].flatten(0,1)
                conv = self.convs[idx]
                out_supp_frame.append(
                    resize(               # 
                        input=conv(x),
                        size=supp_frame[k_ratio].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
                out_memory_frames.append(
                    resize(
                        input=conv(memory),
                        size=query_frame[k_ratio].shape[-2:], # 1/8
                        mode='bilinear',
                        align_corners=False))
                
        out_supp_frame = self.fusion_conv(torch.cat(out_supp_frame,dim=1)) #[BT,C,H,W] 
        out_memory_frames = self.fusion_conv(torch.cat(out_memory_frames,dim=1)) #[BT,C,H,W]

        out_supp_frame = resize(input=out_supp_frame,size=tg_size,mode='bilinear',align_corners=False)
        out_memory_frames = resize(input=out_memory_frames,size=tg_size,mode='bilinear',align_corners=False)
        memory = out_memory_frames.view(-1,T_pre,out_memory_frames.shape[-3],out_memory_frames.shape[-2],out_memory_frames.shape[-1]) #[B,T,C,H,W]
        src = out_supp_frame.view(-1,T_tg,out_supp_frame.shape[-3],out_supp_frame.shape[-2],out_supp_frame.shape[-1]) #[B,T,C,H,W]

        B,_,C,H,W = memory.shape
        z = None
        src = rearrange(src,'b t c h w -> (b t) c (h w)')
        memory = rearrange(memory,'b t c h w -> (b t) c (h w)')
        src = src.permute(0,2,1)
        memory = memory.permute(0,2,1)
        mem_out=None
        for idx in range(self.num_layers):
            if idx == 0:
                x,z,assigned_results = self.cluster_blocks[idx](src, H=H, W=W, mem = memory)
            elif idx == 1:
                x,z,_ = self.cluster_blocks[idx](x, H=H, W=W, z=z, mem = memory) 
            else:
                x,_,_ = self.cluster_blocks[idx](x, H=H, W=W) 
        
        if z is not None:
            z = rearrange(z,'b c (t h w) -> b t c h w',t=T_pre+T_tg,h=H,w=W)

        x = rearrange(x, '(b t) (h w) c -> b t c h w', b=B, t=T_tg,h=H,w=W)
        out_cls_mid,out_new = self.gtem(x)
    
        out_new=(torch.chunk(out_new, T_tg, dim=1))
        out_new=[ii.squeeze(1) for ii in out_new] 
        return out_new,out_cls_mid.squeeze(1),z,mem_out,assigned_results #[b,c,h,w]
