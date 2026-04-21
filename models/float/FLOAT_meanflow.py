import torch, math
import torch.nn as nn
import torch.nn.functional as F

from models import BaseModel
from models.float.generator import Generator
from models.float.FMT_meanflow import MeanFlowTransformer

# 复用原版 AudioEncoder 和 Audio2Emotion（无需修改）
from models.float.FLOAT import AudioEncoder, Audio2Emotion


class FLOAT_MeanFlow(BaseModel):
    """
    FLOAT + MeanFlow 融合版主模型。

    与原版 FLOAT 的差异：
      - FlowMatchingTransformer → MeanFlowTransformer
      - 推理默认使用单步采样（nfe=1）
      - 训练使用 MeanFlow Loss（含 JVP）
      - 保留多步 Euler 采样作为可选项
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames     = int(opt.num_prev_frames)

        # ── 冻结的运动自编码器（Phase 1，与原版相同）
        self.motion_autoencoder = Generator(
            size=opt.input_size, style_dim=opt.dim_w, motion_dim=opt.dim_m
        )
        self.motion_autoencoder.requires_grad_(False)

        # ── 条件编码器（与原版相同）
        self.audio_encoder   = AudioEncoder(opt)
        self.emotion_encoder = Audio2Emotion(opt)

        # ── MeanFlow Transformer（替换原版 FMT）
        self.fmt = MeanFlowTransformer(opt)

    # ──────────────────────────────────────────
    # 图像编解码（与原版相同，保持不变）
    # ──────────────────────────────────────────

    @torch.no_grad()
    def encode_image_into_latent(self, x):
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats

    @torch.no_grad()
    def encode_identity_into_motion(self, x_r):
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x

    @torch.no_grad()
    def decode_latent_into_image(self, s_r, s_r_feats, r_d):
        T = r_d.shape[1]
        d_hat = []
        for t_idx in range(T):
            s_r_d_t = s_r + r_d[:, t_idx]
            img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)
            d_hat.append(img_t)
        d_hat = torch.stack(d_hat, dim=1).squeeze()
        return {'d_hat': d_hat}

    # ──────────────────────────────────────────
    # 训练前向（Phase 2 训练入口）
    # ──────────────────────────────────────────

    def forward_train(self, data):
        """
        训练时的前向传播。
        data 中应包含：
            'x0'   : (B, L, dim_w)  干净运动潜变量序列（由 motion_autoencoder 编码）
            'wa'   : (B, L, dim_w)  音频特征
            'wr'   : (B, dim_w)     参考身份特征
            'we'   : (B, 1, dim_e)  情感特征
            'prev_x'  (可选): 上一段运动序列
            'prev_wa' (可选): 上一段音频序列

        返回：
            loss      : 标量 MeanFlow loss
            loss_dict : 日志字典
        """
        x0     = data['x0']
        wa     = data['wa']
        wr     = data['wr']
        we     = data['we']
        prev_x  = data.get('prev_x',  None)
        prev_wa = data.get('prev_wa', None)

        loss, loss_dict = self.fmt.compute_meanflow_loss(
            x0=x0, wa=wa, wr=wr, we=we,
            prev_x=prev_x, prev_wa=prev_wa,
            p_adaptive=self.opt.mf_p_adaptive,
            c_adaptive=self.opt.mf_c_adaptive,
        )
        return loss, loss_dict

    # ──────────────────────────────────────────
    # 推理：单步或少步采样
    # ──────────────────────────────────────────

    @torch.no_grad()
    def sample(self, data, a_cfg_scale=1.0, r_cfg_scale=1.0, e_cfg_scale=1.0,
               emo=None, nfe=1, seed=None):
        """
        运动序列采样。
        nfe=1  → 单步 MeanFlow 采样（Algorithm 2）
        nfe>1  → 少步 Euler 采样
        """
        r_s, a = data['r_s'], data['a']
        B = a.shape[0]
        a = a.to(self.opt.rank)

        # 编码音频
        T  = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        wa = self.audio_encoder.inference(a, seq_len=T)

        # 编码情感
        emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
        if emo_idx is None:
            we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
        else:
            we = F.one_hot(
                torch.tensor(emo_idx, device=a.device), num_classes=self.opt.dim_e
            ).unsqueeze(0).unsqueeze(0).float()

        sample = []
        num_clips = int(math.ceil(T / self.num_frames_for_clip))

        for chunk_idx in range(num_clips):
            # 初始噪声
            if self.opt.fix_noise_seed:
                s = self.opt.seed if seed is None else seed
                g = torch.Generator(self.opt.rank)
                g.manual_seed(s)
                x1 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w,
                                  device=self.opt.rank, generator=g)
            else:
                x1 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w,
                                  device=self.opt.rank)

            # 前序帧
            if chunk_idx == 0:
                prev_x  = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                prev_wa = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
            else:
                prev_x  = sample_chunk[:, -self.num_prev_frames:]
                prev_wa = wa_chunk[:, -self.num_prev_frames:]

            # 当前音频片段
            wa_chunk = wa[:, chunk_idx * self.num_frames_for_clip:
                             (chunk_idx + 1) * self.num_frames_for_clip]
            if wa_chunk.shape[1] < self.num_frames_for_clip:
                wa_chunk = F.pad(wa_chunk,
                                  (0, 0, 0, self.num_frames_for_clip - wa_chunk.shape[1]),
                                  mode='replicate')

            # ── 采样（单步或少步）
            if nfe == 1:
                sample_chunk = self.fmt.sample_one_step(
                    x1=x1, wa=wa_chunk, wr=r_s, we=we,
                    prev_x=prev_x, prev_wa=prev_wa,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                )
            else:
                sample_chunk = self.fmt.sample_n_steps(
                    x1=x1, wa=wa_chunk, wr=r_s, we=we,
                    prev_x=prev_x, prev_wa=prev_wa, nfe=nfe,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                )

            sample.append(sample_chunk)

        sample = torch.cat(sample, dim=1)[:, :T]
        return sample

    # ──────────────────────────────────────────
    # 完整推理接口（与原版 inference 接口兼容）
    # ──────────────────────────────────────────

    @torch.no_grad()
    def inference(self, data, a_cfg_scale=None, r_cfg_scale=None, e_cfg_scale=None,
                  emo=None, nfe=1, seed=None):
        """
        完整推理：图像编码 → 运动采样 → 图像解码。
        与原版 FLOAT.inference 接口保持兼容。
        """
        s, a = data['s'], data['a']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))

        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s

        if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
        if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
        if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale

        sample   = self.sample(data, a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale,
                                e_cfg_scale=e_cfg_scale, emo=emo, nfe=nfe, seed=seed)
        data_out = self.decode_latent_into_image(s_r=s_r, s_r_feats=s_r_feats, r_d=sample)
        return data_out
