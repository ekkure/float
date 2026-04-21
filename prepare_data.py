"""
prepare_data.py
────────────────────────────────────────────────────────────
从【一张图片 + 一段音频/视频】提取训练所需的 .pt 特征文件。

用法：
    python prepare_data.py \
        --ref_path  assets/test.jpg \
        --aud_path  assets/test.wav \
        --ckpt_path checkpoints/float.pth \
        --out_dir   data/train \
        --n_clips   4

输出：
    data/train/sample_000.pt
    data/train/sample_001.pt
    ...（每个 .pt 对应一个音频片段窗口）

每个 .pt 包含：
    x0 : (L, 512)   运动潜变量序列（干净）
    wa : (L, 512)   音频特征序列
    wr : (512,)     参考身份特征
    we : (1, 7)     情感特征
"""

import os, math, argparse, cv2, librosa, torch
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
import torch.nn.functional as F

from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

from options.base_options import BaseOptions
from models.float.FLOAT_meanflow import FLOAT_MeanFlow


# ──────────────────────────────────────────────────────────────
# 图像预处理（复用 generate.py 里的逻辑，简化版）
# ──────────────────────────────────────────────────────────────

def load_and_transform_image(img_path, input_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(height=input_size, width=input_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A_pytorch.ToTensorV2(),
    ])
    return transform(image=img)['image'].unsqueeze(0)   # (1, 3, H, W)


def load_audio(aud_path, wav2vec_preprocessor, sampling_rate):
    """支持 .wav / .mp3，也支持从视频中提取音频"""
    speech_array, sr = librosa.load(aud_path, sr=sampling_rate)
    return wav2vec_preprocessor(
        speech_array, sampling_rate=sampling_rate, return_tensors='pt'
    ).input_values[0]   # (T_audio,)


# ──────────────────────────────────────────────────────────────
# 主提取逻辑
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(opt, ref_path, aud_path, out_dir, n_clips):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.rank = device

    # ── 加载模型
    print("[1/4] 加载模型...")
    model = FLOAT_MeanFlow(opt)
    state_dict = torch.load(opt.ckpt_path, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
            elif 'wav2vec2' in name:
                pass
    model.to(device)
    model.eval()

    # ── 音频预处理器
    wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
        opt.wav2vec_model_path, local_files_only=True
    )

    # ── 加载图像
    print("[2/4] 处理图像...")
    s = load_and_transform_image(ref_path, opt.input_size).to(device)  # (1,3,H,W)

    # 提取图像 appearance 特征
    s_r, r_s_lambda, s_r_feats = model.encode_image_into_latent(s)
    r_s = model.motion_autoencoder.dec.direction(r_s_lambda)   # (1, 512) 参考身份
    wr  = r_s.squeeze(0).cpu()                                  # (512,)

    # ── 加载音频
    print("[3/4] 处理音频...")
    a = load_audio(aud_path, wav2vec_preprocessor, opt.sampling_rate).to(device)
    a = a.unsqueeze(0)   # (1, T_audio)

    # 计算总帧数
    T_total = math.ceil(a.shape[-1] * opt.fps / opt.sampling_rate)
    L = int(opt.wav2vec_sec * opt.fps)   # 每个片段帧数（默认 50）

    # 提取完整音频特征
    wa_all = model.audio_encoder.inference(a, seq_len=T_total)  # (1, T_total, 512)

    # 提取情感特征
    we = model.emotion_encoder.predict_emotion(a).unsqueeze(1).cpu()  # (1, 1, 7)

    # ── 分片段保存
    print(f"[4/4] 提取运动潜变量并保存（共 {n_clips} 个片段）...")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    total_clips  = math.ceil(T_total / L)
    actual_clips = min(n_clips, total_clips)

    for i in range(actual_clips):
        t_start = i * L
        t_end   = min((i + 1) * L, T_total)

        # 截取并 pad 音频片段特征
        wa_clip = wa_all[:, t_start:t_end]
        if wa_clip.shape[1] < L:
            wa_clip = F.pad(wa_clip, (0, 0, 0, L - wa_clip.shape[1]), mode='replicate')

        # 单步去噪得到 x0（仅用于测试 pipeline，非真实标注）
        x1      = torch.randn(1, L, opt.dim_w, device=device)
        prev_x  = torch.zeros(1, opt.num_prev_frames, opt.dim_w, device=device)
        prev_wa = torch.zeros(1, opt.num_prev_frames, opt.dim_w, device=device)

        x0 = model.fmt.sample_one_step(
            x1=x1, wa=wa_clip, wr=r_s, we=we.to(device),
            prev_x=prev_x, prev_wa=prev_wa,
            a_cfg_scale=opt.a_cfg_scale,
            r_cfg_scale=opt.r_cfg_scale,
            e_cfg_scale=opt.e_cfg_scale,
        )   # (1, L, 512)

        save_path = os.path.join(out_dir, f"sample_{i:03d}.pt")
        torch.save({
            'x0': x0.squeeze(0).cpu(),       # (L, 512)
            'wa': wa_clip.squeeze(0).cpu(),   # (L, 512)
            'wr': wr,                         # (512,)
            'we': we.squeeze(0).cpu(),        # (1, 7)
        }, save_path)
        print(f"    保存: {save_path}")

    print(f"\n完成！共保存 {actual_clips} 个 .pt 文件到 {out_dir}")


# ──────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────

class PrepareOptions(BaseOptions):
    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument('--ref_path',  required=True,  type=str, help='参考图片路径')
        parser.add_argument('--aud_path',  required=True,  type=str, help='音频路径 .wav/.mp3')
        parser.add_argument('--ckpt_path', required=True,  type=str, help='float.pth 路径')
        parser.add_argument('--out_dir',   default='data/train', type=str, help='输出目录')
        parser.add_argument('--n_clips',   default=4,      type=int, help='提取几个片段用于测试')
        return parser


if __name__ == '__main__':
    opt = PrepareOptions().parse()
    opt.rank = 0

    extract_features(
        opt      = opt,
        ref_path = opt.ref_path,
        aud_path = opt.aud_path,
        out_dir  = opt.out_dir,
        n_clips  = opt.n_clips,
    )