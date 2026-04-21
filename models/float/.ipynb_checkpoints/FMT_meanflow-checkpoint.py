import os, math, torch
import torch.nn as nn
import torch.nn.functional as F

from models import BaseModel
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


# ─────────────────────────────────────────────
# 原版辅助函数（保持不变）
# ─────────────────────────────────────────────

def enc_dec_mask(T, S, frame_width=1, expansion=2):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return mask == 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = torch.Tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    return sinusoid_table


# ─────────────────────────────────────────────
# Attention / FMTBlock / Decoder（与原版相同）
# ─────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            # x = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask,
            #                                     dropout_p=self.attn_drop.p if self.training else 0.)
            scale = q.shape[-1] ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            x = attn @ v
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SequenceEmbed(nn.Module):
    def __init__(self, dim_w, dim_h, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)
        self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class FMTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def framewise_modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c, mask=None):
        assert mask is not None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), mask=mask)
        x = x + gate_mlp * self.mlp(self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, dim_w):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, dim_w, bias=True)

    def framewise_modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.framewise_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ─────────────────────────────────────────────
# MeanFlow 时间步嵌入（支持两个时间变量 t, r）
# ─────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """将标量时间步嵌入为向量表示（支持单个时间变量）"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class MeanFlowTimestepEmbedder(nn.Module):
    """
    MeanFlow 双时间变量嵌入器。
    条件化方式：(t, t-r)，即嵌入时间 t 和时间间隔 delta_t = t - r。
    论文 Table 1c 中该方式效果最优。
    输出维度：hidden_size（两个嵌入相加后送入条件）
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.freq_size = frequency_embedding_size
        # 两个独立MLP：一个嵌入 t，一个嵌入 delta_t
        self.mlp_t = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.mlp_dt = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, r):
        """
        t: (B,)  当前时间
        r: (B,)  起始时间（目标时间）
        返回: (B, hidden_size) 时间条件嵌入
        """
        delta_t = t - r                                          # (B,)
        t_emb  = self.mlp_t(self.timestep_embedding(t,       self.freq_size))   # (B, H)
        dt_emb = self.mlp_dt(self.timestep_embedding(delta_t, self.freq_size))  # (B, H)
        return t_emb + dt_emb                                    # (B, H)


# ─────────────────────────────────────────────
# MeanFlow版 FlowMatchingTransformer
# ─────────────────────────────────────────────

class MeanFlowTransformer(BaseModel):
    """
    将 FLOAT 的 FlowMatchingTransformer 改造为 MeanFlow 版本。

    主要变化：
    1. 时间条件：原来只有 t，现在新增 r，使用 (t, t-r) 双嵌入
    2. forward 输出：建模平均速度 u(z, r, t)，而非瞬时速度 v(z, t)
    3. 推理：支持单步采样 z0 = z1 - u(z1, r=0, t=1)
    4. 训练：通过 compute_meanflow_loss 计算 MeanFlow Loss（含 JVP）
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames     = int(opt.num_prev_frames)
        self.num_total_frames    = self.num_prev_frames + self.num_frames_for_clip

        self.hidden_size = opt.dim_h
        self.mlp_ratio   = opt.mlp_ratio
        self.fmt_depth   = opt.fmt_depth
        self.num_heads   = opt.num_heads

        # ── 序列嵌入（与原版相同）
        self.x_embedder = SequenceEmbed(opt.dim_w, self.hidden_size)

        # ── 位置编码（与原版相同）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_total_frames, self.hidden_size), requires_grad=False
        )

        # ── 时间嵌入：MeanFlow 双时间变量嵌入器（替换原版 TimestepEmbedder）
        self.t_embedder = MeanFlowTimestepEmbedder(self.hidden_size)

        # ── 条件嵌入（与原版相同）
        self.c_embedder = nn.Linear(opt.dim_w + opt.dim_a + opt.dim_e, self.hidden_size)

        # ── Transformer 块 & 解码器（与原版相同）
        self.blocks  = nn.ModuleList([
            FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
            for _ in range(self.fmt_depth)
        ])
        self.decoder = Decoder(self.hidden_size, opt.dim_w)

        self.initialize_weights()

        # ── 注意力掩码（与原版相同）
        alignment_mask = enc_dec_mask(
            self.num_total_frames, self.num_total_frames, 1, expansion=opt.attention_window
        ).to(opt.rank)
        self.register_buffer('alignment_mask', alignment_mask)

    # ──────────────────────────────────────────
    # 权重初始化（与原版一致）
    # ──────────────────────────────────────────

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_sinusoid_encoding_table(self.num_total_frames, self.hidden_size)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # adaLN 调制层初始化为零
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.linear.weight, 0)
        nn.init.constant_(self.decoder.linear.bias, 0)

    # ──────────────────────────────────────────
    # 条件 dropout（与原版相同）
    # ──────────────────────────────────────────

    def sequence_embedder(self, sequence, dropout_prob, train=False):
        if train:
            mask = (torch.rand(sequence.shape[0], device=sequence.device) >= dropout_prob).float()
            mask = mask.view(-1, 1, 1)
            sequence = sequence * mask
        return sequence

    # ──────────────────────────────────────────
    # 核心 forward：建模平均速度 u(z, r, t)
    # ──────────────────────────────────────────

    def forward(self, t, r, x, wa, wr, we, prev_x=None, prev_wa=None, train=True):
        """
        参数说明：
            t  : (B,)         当前时间步 [0,1]
            r  : (B,)         目标时间步 [0,1]，r < t
            x  : (B, L, dim_w)  噪声运动序列
            wa : (B, L, dim_w)  音频特征序列
            wr : (B, dim_w)     参考身份特征
            we : (B, 1, dim_e)  情感特征
            prev_x  : (B, L', dim_w)  上一段运动序列
            prev_wa : (B, L', dim_w)  上一段音频序列

        返回：
            u  : (B, L, dim_w)  平均速度预测（MeanFlow）
        """
        # ── 时间条件（MeanFlow双时间嵌入）
        t_emb = self.t_embedder(t, r).unsqueeze(1)   # (B, 1, H)

        # ── 条件 dropout
        wa = self.sequence_embedder(wa, dropout_prob=self.opt.audio_dropout_prob, train=train)
        wr = self.sequence_embedder(wr.unsqueeze(1),  dropout_prob=self.opt.ref_dropout_prob,   train=train)
        we = self.sequence_embedder(we,               dropout_prob=self.opt.emotion_dropout_prob, train=train)

        # ── 自回归拼接前序帧
        if prev_x is not None:
            prev_x  = self.sequence_embedder(prev_x,  dropout_prob=0.5, train=train)
            prev_wa = self.sequence_embedder(prev_wa, dropout_prob=0.5, train=train)
            x  = torch.cat([prev_x,  x],  dim=1)
            wa = torch.cat([prev_wa, wa], dim=1)

        # ── 序列嵌入 + 位置编码
        x = self.x_embedder(x) + self.pos_embed     # (B, L_total, H)

        # ── 条件融合
        wr = wr.repeat(1, wa.shape[1], 1)
        we = we.repeat(1, wa.shape[1], 1)
        c = self.c_embedder(torch.cat([wr, wa, we], dim=-1))  # (B, L_total, H)
        c = t_emb + c                                          # 时间条件注入

        # ── Transformer 块
        for block in self.blocks:
            x = block(x, c, self.alignment_mask)

        return self.decoder(x, c)   # (B, L_total, dim_w)：平均速度 u

    # ──────────────────────────────────────────
    # CFG 推理（与原版结构相同，增加 r 参数）
    # ──────────────────────────────────────────

    @torch.no_grad()
    def forward_with_cfg(self, t, r, x, wa, wr, we, prev_x, prev_wa,
                         a_cfg_scale=1.0, r_cfg_scale=1.0, e_cfg_scale=1.0):
        """
        分类器自由引导推理。
        返回平均速度 u，可直接用于单步采样：z0 = z1 - (t-r)*u
        """
        if a_cfg_scale != 1.0 or r_cfg_scale != 1.0 or e_cfg_scale != 1.0:
            null_wa = torch.zeros_like(wa)
            null_we = torch.zeros_like(we)

            audio_cat   = torch.cat([null_wa, wa,  wa],      dim=0)
            ref_cat     = torch.cat([wr,      wr,  wr],      dim=0)
            emotion_cat = torch.cat([null_we, we,  null_we], dim=0)
            x_cat       = torch.cat([x,       x,   x],       dim=0)
            t_cat       = t.repeat(3)
            r_cat       = r.repeat(3)

            prev_x_cat  = torch.cat([prev_x,  prev_x,  prev_x],  dim=0)
            prev_wa_cat = torch.cat([prev_wa, prev_wa, prev_wa], dim=0)

            out = self.forward(t_cat, r_cat, x_cat, audio_cat, ref_cat, emotion_cat,
                               prev_x_cat, prev_wa_cat, train=False)
            uncond, all_cond, audio_uncond_emotion = torch.chunk(out, chunks=3, dim=0)

            return uncond + a_cfg_scale * (audio_uncond_emotion - uncond) \
                         + e_cfg_scale * (all_cond - audio_uncond_emotion)
        else:
            return self.forward(t, r, x, wa, wr, we, prev_x, prev_wa, train=False)

    # ──────────────────────────────────────────
    # MeanFlow 训练 Loss 计算
    # ──────────────────────────────────────────

    def compute_meanflow_loss(self, x0, wa, wr, we, prev_x=None, prev_wa=None,
                               t=None, r=None, p_adaptive=1.0, c_adaptive=1e-3):
        """
        计算 MeanFlow Loss（论文 Eq.9-11 + Algorithm 1）。

        流程：
          1. 构造加噪轨迹 z_t = (1-t)*x0 + t*eps
          2. 计算条件速度 v_t = eps - x0
          3. 用 JVP 计算 d/dt u_theta(z_t, r, t)
          4. 构造目标 u_tgt = v_t - (t-r) * dudt
          5. 自适应加权 L2 loss

        参数：
            x0 : (B, L, dim_w)  干净运动序列（训练目标）
            wa, wr, we          : 条件特征
            t  : (B,) 当前时间，若 None 则随机采样
            r  : (B,) 目标时间，若 None 则随机采样（保证 r <= t）
            p_adaptive          : 自适应权重指数（论文推荐 p=1.0）
            c_adaptive          : 数值稳定常数（默认 1e-3）

        返回：
            loss       : 标量 loss
            loss_dict  : 包含各分量的字典，用于日志
        """
        B, L, D = x0.shape
        device   = x0.device
        # ── 若无前序帧，补零（保证 pos_embed 维度一致）
        if prev_x is None:
            prev_x  = torch.zeros(B, self.num_prev_frames, D, device=device, dtype=x0.dtype)
        if prev_wa is None:
            prev_wa = torch.zeros(B, self.num_prev_frames, D, device=device, dtype=x0.dtype)
        # ── 采样时间对 (r, t)，lognorm(-0.4, 1.0)，论文 Table 1d 最优
        if t is None or r is None:
            t, r = self._sample_time_pair(B, device)

        # ── 构造加噪轨迹
        eps = torch.randn_like(x0)
        t_bc = t.view(B, 1, 1)                    # broadcast
        z_t  = (1.0 - t_bc) * x0 + t_bc * eps     # (B, L, D)
        v_t  = eps - x0                            # 条件速度

        # ── 定义被 JVP 包装的函数：只对 (z, t) 敏感
        # 注意：JVP 的切向量是 (v_t, 0, 1)，对应 (dz/dt, dr/dt, dt/dt)
        def fn_for_jvp(z_in, t_in):
            # r 作为外部闭包，不参与 JVP 对 t 的求导
            return self.forward(t_in, r, z_in, wa, wr, we, prev_x, prev_wa, train=True)

        # ── 计算 u 和 JVP：dudt = v * ∂u/∂z + ∂u/∂t
        # torch.func.jvp(fn, primals, tangents)
        # primals: (z_t, t)   tangents: (v_t, ones) 对应论文切向量 (v, 0, 1)
        # 注意 r 的切向量为 0，所以将 r 放在闭包中而非 primals 中
        u_pred, dudt = torch.func.jvp(
            fn_for_jvp,
            (z_t, t),
            (v_t, torch.ones_like(t))
        )
        # u_pred : (B, L_total, D)  平均速度预测
        # dudt   : (B, L_total, D)  d/dt u 的 JVP 结果

        # ── 截取当前帧（去掉 prev 帧部分）
        if prev_x is not None:
            u_pred = u_pred[:, self.num_prev_frames:]
            dudt   = dudt[:, self.num_prev_frames:]

        # ── 构造 MeanFlow 目标（论文 Eq.11，stop-gradient 在外部通过 detach 实现）
        dt_bc   = (t - r).view(B, 1, 1)           # (B, 1, 1)
        u_tgt   = v_t - dt_bc * dudt.detach()     # stop-gradient on dudt

        # ── 自适应加权 L2 Loss（论文 Eq.22，p=1.0）
        error   = u_pred - u_tgt.detach()          # stop-gradient on target
        loss_l2 = (error ** 2).sum(dim=-1)          # (B, L)

        if p_adaptive > 0:
            weight = 1.0 / (loss_l2.detach() + c_adaptive) ** p_adaptive
            loss   = (weight * loss_l2).mean()
        else:
            loss   = loss_l2.mean()

        loss_dict = {
            'loss_meanflow': loss_l2.mean().item(),
            'u_norm':   u_pred.detach().norm(dim=-1).mean().item(),
            'tgt_norm': u_tgt.detach().norm(dim=-1).mean().item(),
            't_mean':   t.mean().item(),
            'r_mean':   r.mean().item(),
        }
        return loss, loss_dict

    # ──────────────────────────────────────────
    # 时间对采样（lognorm，论文推荐）
    # ──────────────────────────────────────────

    def _sample_time_pair(self, B, device, ratio_r_neq_t=0.25,
                           lognorm_mu=-0.4, lognorm_sigma=1.0):
        """
        采样 (r, t) 对。
        - 25% 概率 r != t（MeanFlow 生效）
        - 75% 概率 r == t（退化为标准 Flow Matching，稳定训练）
        论文 Table 1a：ratio=25% 效果最优
        """
        # logit-normal 采样 t
        normal_samples = torch.randn(B, device=device) * lognorm_sigma + lognorm_mu
        t = torch.sigmoid(normal_samples).clamp(1e-4, 1 - 1e-4)   # (B,)

        # 采样 r：uniformly in [0, t]
        r = torch.rand(B, device=device) * t

        # 部分样本令 r = t（退化为 FM）
        use_fm = torch.rand(B, device=device) > ratio_r_neq_t
        r = torch.where(use_fm, t.clone(), r)

        return t, r

    # ──────────────────────────────────────────
    # 单步采样（MeanFlow 核心推理，Algorithm 2）
    # ──────────────────────────────────────────

    @torch.no_grad()
    def sample_one_step(self, x1, wa, wr, we, prev_x, prev_wa,
                         a_cfg_scale=1.0, r_cfg_scale=1.0, e_cfg_scale=1.0):
        """
        单步推理：z0 = z1 - u(z1, r=0, t=1)
        对应论文 Algorithm 2。
        """
        B = x1.shape[0]
        device = x1.device
        t = torch.ones(B,  device=device)    # t = 1
        r = torch.zeros(B, device=device)    # r = 0

        u = self.forward_with_cfg(
            t, r, x1, wa, wr, we, prev_x, prev_wa,
            a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale, e_cfg_scale=e_cfg_scale
        )
        # 截取当前帧
        u_current = u[:, self.num_prev_frames:]
        # z0 = z1 - (t - r) * u = z1 - 1.0 * u
        z0 = x1 - u_current
        return z0

    # ──────────────────────────────────────────
    # 少步采样（可选，N 步 Euler）
    # ──────────────────────────────────────────

    @torch.no_grad()
    def sample_n_steps(self, x1, wa, wr, we, prev_x, prev_wa, nfe=4,
                        a_cfg_scale=1.0, r_cfg_scale=1.0, e_cfg_scale=1.0):
        """
        N 步 Euler 采样（论文 Eq.12）：
            z_r = z_t - (t - r) * u(z_t, r, t)
        从 t=1 → t=0，分 nfe 步。
        """
        B = x1.shape[0]
        device = x1.device
        times = torch.linspace(1.0, 0.0, nfe + 1, device=device)   # [1, ..., 0]

        z = x1
        for i in range(nfe):
            t_val = times[i]
            r_val = times[i + 1]
            t = t_val.expand(B)
            r = r_val.expand(B)

            u = self.forward_with_cfg(
                t, r, z, wa, wr, we, prev_x, prev_wa,
                a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale, e_cfg_scale=e_cfg_scale
            )
            u_current = u[:, self.num_prev_frames:]
            z = z - (t_val - r_val) * u_current

        return z
