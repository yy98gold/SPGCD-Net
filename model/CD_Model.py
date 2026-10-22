import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SemanticTransitionModule(nn.Module):
    """
    Semantic Difference-guided Cross Attention (SDCA-STM)

    Input:
        sem_a : [B,C,H,W]
        sem_b : [B,C,H,W]

    Output:
        transition : [B,C,H,W]
    """

    def __init__(self,
                 dim,
                 num_heads=8):
        super().__init__()

        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # -----------------------------
        # Local Semantic Enhancement
        # -----------------------------
        self.local_a = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.local_b = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # -----------------------------
        # QKV
        # -----------------------------
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)

        # -----------------------------
        # Difference Gate
        # -----------------------------
        self.diff_gate = nn.Sequential(

            nn.Conv2d(dim, dim, 3, padding=1),

            nn.BatchNorm2d(dim),

            nn.GELU(),

            nn.Conv2d(dim, dim, 1),

            nn.Sigmoid()

        )

        # -----------------------------
        # Transition Refinement
        # -----------------------------
        self.refine = nn.Sequential(

            nn.Conv2d(dim, dim, 3, padding=1),

            nn.BatchNorm2d(dim),

            nn.GELU(),

            nn.Conv2d(dim, dim, 3, padding=1),

            nn.BatchNorm2d(dim),

            nn.GELU()

        )

    def reshape(self, x):

        B, C, H, W = x.shape

        x = x.view(
            B,
            self.head,
            self.head_dim,
            H * W
        )

        x = x.permute(
            0,
            1,
            3,
            2
        )

        return x

    def restore(self, x, H, W):

        B = x.shape[0]

        x = x.permute(
            0,
            1,
            3,
            2
        ).contiguous()

        x = x.view(
            B,
            self.dim,
            H,
            W
        )

        return x

    def forward(self,
                sem_a,
                sem_b):

        B, C, H, W = sem_a.shape

        # ======================================
        # Local Enhancement
        # ======================================
        feat_a = self.local_a(sem_a)

        feat_b = self.local_b(sem_b)

        # ======================================
        # Cross Attention
        # ======================================
        q = self.reshape(
            self.q_proj(feat_a)
        )

        k = self.reshape(
            self.k_proj(feat_b)
        )

        v = self.reshape(
            self.v_proj(feat_b)
        )

        attn = torch.matmul(
            q,
            k.transpose(-1, -2)
        )

        attn = attn * self.scale

        attn = attn.softmax(dim=-1)

        cross = torch.matmul(
            attn,
            v
        )

        cross = self.restore(
            cross,
            H,
            W
        )

        # ======================================
        # Difference Gate
        # ======================================
        diff = torch.abs(
            feat_a -
            feat_b
        )

        gate = self.diff_gate(
            diff
        )

        transition = cross * gate

        # residual
        transition = transition + diff

        # ======================================
        # Refinement
        # ======================================
        transition = self.refine(
            transition
        )

        return transition

class FeatureGuidedSPGBlock(nn.Module):

    def __init__(self,
                 feat_dim,
                 sem_dim):

        super().__init__()

        self.fuse = nn.Sequential(

            nn.Conv2d(
                feat_dim + sem_dim,
                sem_dim,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(sem_dim),

            nn.GELU(),

            nn.Conv2d(
                sem_dim,
                sem_dim,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(sem_dim),

            nn.GELU()
        )

        self.attention = nn.Sequential(

            nn.Conv2d(
                sem_dim,
                sem_dim,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self,
                semantic,
                feature):

        semantic = F.interpolate(

            semantic,

            size=feature.shape[-2:],

            mode="bilinear",

            align_corners=False
        )

        x = torch.cat(

            [

                semantic,

                feature

            ],

            dim=1
        )

        x = self.fuse(x)

        weight = self.attention(x)

        x = x * weight

        semantic = semantic + x

        return semantic

class FeatureGuidedSPG(nn.Module):

    def __init__(self,
                 dim=256):

        super().__init__()

        ####################################

        self.stage3 = FeatureGuidedSPGBlock(

            feat_dim=256,

            sem_dim=256

        )

        ####################################

        self.stage2 = FeatureGuidedSPGBlock(

            feat_dim=128,

            sem_dim=256

        )

        ####################################

        self.stage1 = FeatureGuidedSPGBlock(

            feat_dim=64,

            sem_dim=256

        )

    def forward(

            self,

            transition,

            f1,

            f2,

            f3):

        ################################

        transition3 = self.stage3(

            transition,

            f3

        )

        ################################

        transition2 = self.stage2(

            transition3,

            f2

        )

        ################################

        transition1 = self.stage1(

            transition2,

            f1

        )

        return (

            transition1,

            transition2,

            transition3
        )


# =========================================================
# 1. Cross-Temporal Interaction 
# =========================================================
class ETI(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim*3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1),
            nn.GELU(),
            nn.Conv2d(dim*4, dim, 1)
        )

    def forward(self, a, b):
        B, C, H, W = a.shape
        a_in = a
        b_in = b
        a = self.norm(a)
        b = self.norm(b)
       
        q_a, k_a, v_a = torch.chunk(self.qkv(a), 3, dim=1)
        q_b, k_b, v_b = torch.chunk(self.qkv(b), 3, dim=1)

        def reshape(x):
            return rearrange(x, "B (h c) H W -> B h (H W) c", h=self.head)
        qa, ka, va = reshape(q_a), reshape(k_a), reshape(v_a)
        qb, kb, vb = reshape(q_b), reshape(k_b), reshape(v_b)

        attn_ab = (qa @ kb.transpose(-1,-2)) * self.scale
        attn_ab = attn_ab.softmax(dim=-1)
        feat_a_cross = attn_ab @ vb
        feat_a_cross = rearrange(feat_a_cross, "B h (H W) c -> B (h c) H W", H=H, W=W)

        attn_ba = (qb @ ka.transpose(-1,-2)) * self.scale
        attn_ba = attn_ba.softmax(dim=-1)
        feat_b_cross = attn_ba @ va
        feat_b_cross = rearrange(feat_b_cross, "B h (H W) c -> B (h c) H W", H=H, W=W)

        out_a = self.proj(feat_a_cross) + a_in
        out_a = self.ffn(self.norm(out_a)) + out_a

        out_b = self.proj(feat_b_cross) + b_in
        out_b = self.ffn(self.norm(out_b)) + out_b

        return out_a, out_b


# =========================================================
# 2. Change Bottleneck 
# =========================================================
class ChangeBottleneck(nn.Module):

    def __init__(self, dim):
        super().__init__()

        # abs
        # sub
        # mul
        # transition

        self.diff_fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, a, b, transition):

        diff_abs = torch.abs(a - b)
        diff_sub = a - b
        diff_mul = a * b
        diff = torch.cat([
            diff_abs,
            diff_sub,
            diff_mul,
            transition
        ], dim=1)

        diff = self.diff_fuse(diff)

        gate = self.gate(
            torch.cat([ a, b, transition ], dim=1 ) )

        out = diff * gate
        return self.refine(out)


class SkipFusion(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # Channel Attention
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.sem_proj = nn.Conv2d(256, dim, 1)

        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, fa, fb, transition):

        diff = torch.abs(fa - fb)

        transition = F.interpolate(
            transition,
            size=diff.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        transition = self.sem_proj(transition)        # shape

        # diff = diff * torch.sigmoid(transition)

        gate = torch.sigmoid(transition)         # gate,Residual Attention

        diff = diff * (1 + gate)

        x = torch.cat([
            fa,
            fb,
            diff
        ], dim=1)

        x = self.fuse(x)

        w = self.attn(x)

        x = x * w

        x = self.out(x)

        return x + diff



class Decoder(nn.Module):

    def __init__(self, dim=256, out_channels=1):
        super().__init__()

        # Skip Fusion
        self.skip1 = SkipFusion(64)
        self.skip2 = SkipFusion(128)
        self.skip3 = SkipFusion(256)

        # =====================
        # 16 -> 32
        # =====================
        # self.up1 = nn.ConvTranspose2d( dim,256, kernel_size=2, stride=2 )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, 256, 3, padding=1)
        )

        self.conv1 = nn.Sequential( nn.Conv2d( 512, 256,3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256,256,3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # =====================
        # 32 -> 64
        # =====================
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        # =====================
        # 64 -> 128
        # =====================
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # =====================
        # 128 -> 256
        # =====================
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(
            self,
            x,

            f1_a,
            f2_a,
            f3_a,

            f1_b,
            f2_b,
            f3_b,

            transition1,
            transition2,
            transition3 ):
        # ----------------------------------
        # Learnable Skip Fusion
        # ----------------------------------
        skip1 = self.skip1(
            f1_a,
            f1_b,
            transition1
        )

        skip2 = self.skip2(
            f2_a,
            f2_b,
            transition2
        )

        skip3 = self.skip3(
            f3_a,
            f3_b,
            transition3
        )

        # ----------------------------------
        # Stage4 -> Stage3
        # ----------------------------------
        x = self.up1(x)

        x = torch.cat([
            x,
            skip3
        ], dim=1)

        x = self.conv1(x)

        # ----------------------------------
        # Stage3 -> Stage2
        # ----------------------------------
        x = self.up2(x)

        x = torch.cat([
            x,
            skip2
        ], dim=1)

        x = self.conv2(x)

        # ----------------------------------
        # Stage2 -> Stage1
        # ----------------------------------
        x = self.up3(x)

        x = torch.cat([
            x,
            skip1
        ], dim=1)

        x = self.conv3(x)

        # ----------------------------------
        # Stage1 -> Output
        # ----------------------------------
        x = self.up4(x)

        x = self.conv4(x)

        return self.out(x)

# =========================================================
# 4. CD_Model
# =========================================================
class CD_Model(nn.Module):
    """
    Fully upgraded change detection network
    """

    #inp_A_channels / inp_B_channels / out_channels
    def __init__(self,
                 backbone,
                 feature_dim=256,
                 inp_A_channels=3,
                 inp_B_channels=3,
                 out_channels=1):
        super().__init__()

        self.backbone = backbone
        self.inp_A_channels = inp_A_channels
        self.inp_B_channels = inp_B_channels
        self.out_channels = out_channels

        # temporal interaction
        self.eti = ETI(feature_dim)

        # change modeling bottleneck
        self.ctb = ChangeBottleneck(feature_dim)

        self.stm = SemanticTransitionModule(feature_dim)
        self.spg = FeatureGuidedSPG(feature_dim)               

        self.decoder = Decoder(feature_dim, out_channels)

        self.feature_dim = feature_dim

    # ---------------------------------------------------------
    def forward_backbone(self, x):
        """
        expected backbone output:
        (B, C, H, W)
        """
        return self.backbone(x)

    # ---------------------------------------------------------
    def forward(self, img_a, img_b, sem_a, sem_b):

        feats_a = self.forward_backbone(img_a)
        feats_b = self.forward_backbone(img_b)

        f1_a, f2_a, f3_a, f4_a = feats_a
        f1_b, f2_b, f3_b, f4_b = feats_b

        feat_a = f4_a
        feat_b = f4_b

        transition = self.stm(

            sem_a,

            sem_b
        )

        transition1, transition2, transition3 = self.spg(

            transition,

            torch.abs(f1_a - f1_b),

            torch.abs(f2_a - f2_b),

            torch.abs(f3_a - f3_b)
        )

        # =========================
        # Temporal Interaction
        # =========================
        feat_a, feat_b = self.eti(feat_a, feat_b)


        # =========================
        # Transition Resize
        # =========================
        transition = F.interpolate(
            transition,
            size=feat_a.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # =========================
        change_feat = self.ctb(
            feat_a,
            feat_b,
            transition
        )

        # ========== Prevent features from being zeroed out ==========
        sem_weight = torch.sigmoid(transition)
        change_feat = change_feat * sem_weight + change_feat


        out = self.decoder(

            change_feat,

            f1_a,
            f2_a,
            f3_a,

            f1_b,
            f2_b,
            f3_b,

            transition1,
            transition2,
            transition3
        )


        out = F.interpolate(
            out,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        return out, transition, change_feat
