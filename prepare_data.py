"""
prepare_data.py  （正确版本）
────────────────────────────────────────────────────────────────────
从真实说话人视频中提取 MeanFlow Phase-2 训练所需的 .pt 特征文件。

核心修正：
  x0 = dec.direction(enc.fc(enc(真实视频帧)))   ← 真实运动潜变量
  而不是 sample_one_step() 生成的伪造数据

数据流：
  真实视频
    ├── 视频流 → 人脸检测裁剪 → 逐帧编码 → x0  (B, L, 512)
    ├── 音频流 → wav2vec2     → 投影     → wa  (B, L, 512)
    ├── 音频流 → emotion      → softmax  → we  (B, 1,   7)
    └── 参考帧 → motion enc   → identity → wr  (B,    512)

用法（单个视频）：
    python prepare_data.py \
        --mode single \
        --video_path assets/talk.mp4 \
        --ckpt_path  checkpoints/float.pth \
        --out_dir    data/train

用法（批量数据集，如 VoxCeleb2）：
    python prepare_data.py \
        --mode  dataset \
        --data_root /data/voxceleb2 \
        --ckpt_path checkpoints/float.pth \
        --out_dir   data/train \
        --num_workers 4

依赖：
    pip install opencv-python librosa albumentations face_alignment
    transformers torch torchvision tqdm
"""

import os, math, argparse, cv2, librosa, torch, subprocess, tempfile, json, traceback
import numpy as np
import torch.nn.functional as F
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
import face_alignment

from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed

from options.base_options import BaseOptions
from models.float.FLOAT_meanflow import FLOAT_MeanFlow


# ═══════════════════════════════════════════════════════════════
# 一、图像处理工具
# ═══════════════════════════════════════════════════════════════

class FaceCropper:
    """
    与 generate.py 完全一致的人脸裁剪逻辑。
    训练和推理必须使用相同的裁剪策略，否则 motion encoder 的输出分布会偏移。
    """
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False
        )
        self.transform = A.Compose([
            A.Resize(height=input_size, width=input_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A_pytorch.ToTensorV2(),
        ])

    def crop(self, img: np.ndarray) -> np.ndarray:
        """输入 RGB ndarray，返回裁剪后的 RGB ndarray"""
        mult = 360.0 / img.shape[0]
        resized = cv2.resize(img, (0, 0), fx=mult, fy=mult,
                             interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)
        bboxes = self.fa.face_detector.detect_from_image(resized)
        bboxes = [
            (int(x1/mult), int(y1/mult), int(x2/mult), int(y2/mult), sc)
            for x1, y1, x2, y2, sc in bboxes if sc > 0.95
        ]
        if not bboxes:
            return None
        x1, y1, x2, y2, _ = bboxes[0]
        bsy = int((y2 - y1) / 2)
        bsx = int((x2 - x1) / 2)
        my  = int((y1 + y2) / 2)
        mx  = int((x1 + x2) / 2)
        bs  = int(max(bsy, bsx) * 1.6)
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        my += bs; mx += bs
        crop = img[my-bs:my+bs, mx-bs:mx+bs]
        return cv2.resize(crop, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """裁剪后的 RGB ndarray → (1,3,H,W) tensor，归一化到 [-1,1]"""
        return self.transform(image=img)['image'].unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# 二、视频 / 音频 IO
# ═══════════════════════════════════════════════════════════════

def extract_audio_from_video(video_path: str, sampling_rate: int) -> np.ndarray:
    """
    从视频中提取音频（通过临时 wav 文件）。
    返回 float32 numpy array，采样率为 sampling_rate。
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_wav = f.name
    try:
        cmd = (f"ffmpeg -y -i {video_path} -ar {sampling_rate} "
               f"-ac 1 -vn {tmp_wav} -loglevel error")
        subprocess.run(cmd, shell=True, check=True)
        audio, _ = librosa.load(tmp_wav, sr=sampling_rate)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    return audio


def extract_frames(video_path: str, target_fps: int) -> list:
    """
    按 target_fps 提取视频帧，返回 list of RGB ndarray。
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0
    step = max(1, round(src_fps / target_fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


# ═══════════════════════════════════════════════════════════════
# 三、核心特征提取（正确版本）
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_real_motion_latents(
    model: FLOAT_MeanFlow,
    cropped_frames: list,       # list of RGB ndarray, already cropped
    cropper: FaceCropper,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    ✅ 正确做法：从真实视频帧提取运动潜变量序列。

    每帧通过 motion_autoencoder.enc 编码后，经 dec.direction 映射到
    与 FMT 输出相同的运动方向向量空间。

    返回：
        x0 : (T, dim_w)  真实运动潜变量序列，dim_w 通常为 512
    """
    x0_list = []
    for i in range(0, len(cropped_frames), batch_size):
        batch_frames = cropped_frames[i:i+batch_size]
        # 转 tensor：(B, 3, H, W)
        tensors = torch.cat([cropper.to_tensor(f) for f in batch_frames], dim=0).to(device)

        # ── 关键步骤：用真实帧编码得到运动潜变量 ──
        # x_r      : (B, style_dim)   appearance latent
        # x_r_lambda: (B, style_dim)  style code
        x_r, _, _ = model.motion_autoencoder.enc(tensors, input_target=None)
        x_r_lambda = model.motion_autoencoder.enc.fc(x_r)

        # r_d : (B, dim_w)  运动方向向量，与 FMT 预测的目标空间一致
        r_d = model.motion_autoencoder.dec.direction(x_r_lambda)
        x0_list.append(r_d.cpu())

    return torch.cat(x0_list, dim=0)   # (T, dim_w)


@torch.no_grad()
def extract_reference_identity(
    model: FLOAT_MeanFlow,
    ref_frame: np.ndarray,
    cropper: FaceCropper,
    device: torch.device,
) -> torch.Tensor:
    """
    从参考帧提取身份特征 wr。
    通常取视频的第一帧（或随机抽取一帧）作为参考。

    返回：
        wr : (dim_w,)
    """
    s = cropper.to_tensor(ref_frame).to(device)
    x_r, _, _ = model.motion_autoencoder.enc(s, input_target=None)
    x_r_lambda = model.motion_autoencoder.enc.fc(x_r)
    r_s = model.motion_autoencoder.dec.direction(x_r_lambda)   # (1, dim_w)
    return r_s.squeeze(0).cpu()


# ═══════════════════════════════════════════════════════════════
# 四、主提取函数：处理单个视频
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def process_single_video(
    model        : FLOAT_MeanFlow,
    wav2vec_prep : Wav2Vec2FeatureExtractor,
    cropper      : FaceCropper,
    video_path   : str,
    out_dir      : str,
    opt,
    video_id     : str = None,
    min_frames   : int = 25,
) -> int:
    """
    处理一个视频文件，提取所有片段并保存为 .pt 文件。

    返回：成功保存的片段数
    """
    device   = next(model.parameters()).device
    L        = int(opt.wav2vec_sec * opt.fps)       # 每片段帧数，如 50
    n_prev   = int(opt.num_prev_frames)

    # ── 1. 提取视频帧
    frames = extract_frames(video_path, target_fps=opt.fps)
    if len(frames) < min_frames:
        return 0  # 视频太短，跳过

    # ── 2. 人脸裁剪（逐帧）
    cropped = []
    for f in frames:
        c = cropper.crop(f)
        if c is not None:
            cropped.append(c)
    if len(cropped) < min_frames:
        return 0  # 人脸检测失败帧太多

    # ── 3. 提取音频
    try:
        audio = extract_audio_from_video(video_path, opt.sampling_rate)
    except Exception:
        return 0

    # ── 4. 编码音频特征
    T_frames = len(cropped)
    audio_input = wav2vec_prep(
        audio, sampling_rate=opt.sampling_rate, return_tensors='pt'
    ).input_values[0].unsqueeze(0).to(device)
    wa_all = model.audio_encoder.inference(audio_input, seq_len=T_frames)  # (1, T, 512)
    wa_all = wa_all.squeeze(0).cpu()   # (T, 512)

    # ── 5. 提取情感特征（整段音频取一个情感向量）
    we = model.emotion_encoder.predict_emotion(audio_input)   # (1, 7)
    we = we.unsqueeze(1).cpu()                                 # (1, 1, 7)

    # ── 6. ✅ 提取真实运动潜变量 x0（核心修正）
    x0_all = extract_real_motion_latents(model, cropped, cropper, device)  # (T, 512)

    # ── 7. 提取参考身份 wr（使用第一帧）
    wr = extract_reference_identity(model, cropped[0], cropper, device)    # (512,)

    # ── 8. 对齐长度（以最短的为准）
    T_min = min(x0_all.shape[0], wa_all.shape[0])
    x0_all = x0_all[:T_min]
    wa_all = wa_all[:T_min]

    # ── 9. 分片段保存
    os.makedirs(out_dir, exist_ok=True)
    vid_id   = video_id or Path(video_path).stem
    n_saved  = 0
    n_clips  = T_min // L

    for i in range(n_clips):
        t_start = i * L
        t_end   = t_start + L

        x0_clip = x0_all[t_start:t_end]   # (L, 512)
        wa_clip = wa_all[t_start:t_end]   # (L, 512)

        # ── 关键：同时保存 prev 帧，供训练时自回归使用
        if i == 0:
            prev_x  = torch.zeros(n_prev, x0_all.shape[-1])
            prev_wa = torch.zeros(n_prev, wa_all.shape[-1])
        else:
            prev_start = max(0, t_start - n_prev)
            prev_x  = x0_all[prev_start:t_start]
            prev_wa = wa_all[prev_start:t_start]
            # 如果不足 n_prev 帧，左侧补零
            if prev_x.shape[0] < n_prev:
                pad = torch.zeros(n_prev - prev_x.shape[0], prev_x.shape[-1])
                prev_x  = torch.cat([pad, prev_x],  dim=0)
                prev_wa = torch.cat([pad, prev_wa], dim=0)

        save_path = os.path.join(out_dir, f"{vid_id}_clip{i:04d}.pt")
        torch.save({
            'x0'     : x0_clip.float(),      # (L, 512)  ✅ 真实运动潜变量
            'wa'     : wa_clip.float(),      # (L, 512)
            'wr'     : wr.float(),           # (512,)
            'we'     : we.squeeze(0).float(),# (1, 7)
            'prev_x' : prev_x.float(),       # (n_prev, 512)  自回归上下文
            'prev_wa': prev_wa.float(),      # (n_prev, 512)
            'video_id': vid_id,
            'clip_idx': i,
        }, save_path)
        n_saved += 1

    return n_saved


# ═══════════════════════════════════════════════════════════════
# 五、数据集批量处理
# ═══════════════════════════════════════════════════════════════

def find_all_videos(data_root: str) -> list:
    """递归查找所有视频文件"""
    exts = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
    root = Path(data_root)
    return [str(p) for p in root.rglob('*') if p.suffix.lower() in exts]


def process_dataset(opt, model, wav2vec_prep, cropper):
    """批量处理整个数据集目录"""
    video_list = find_all_videos(opt.data_root)
    print(f"[数据集模式] 找到 {len(video_list)} 个视频")

    # 已处理的视频（断点续传）
    done_path = os.path.join(opt.out_dir, '.done_videos.json')
    done_set  = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done_set = set(json.load(f))
        print(f"  已跳过 {len(done_set)} 个已处理视频（断点续传）")

    total_clips = 0
    failed      = []

    for video_path in tqdm(video_list, desc="处理视频"):
        video_id = Path(video_path).stem
        if video_id in done_set:
            continue

        # 按视频来源组织输出目录（保留一级子目录结构）
        rel_parent = Path(video_path).parent.name
        out_subdir = os.path.join(opt.out_dir, rel_parent)

        try:
            n = process_single_video(
                model, wav2vec_prep, cropper,
                video_path, out_subdir, opt,
                video_id=f"{rel_parent}_{video_id}",
                min_frames=int(opt.wav2vec_sec * opt.fps),
            )
            total_clips += n
            if n > 0:
                done_set.add(video_id)
                # 定期保存进度
                if len(done_set) % 100 == 0:
                    with open(done_path, 'w') as f:
                        json.dump(list(done_set), f)
        except Exception as e:
            failed.append((video_path, str(e)))
            if opt.verbose:
                traceback.print_exc()

    # 保存最终进度
    with open(done_path, 'w') as f:
        json.dump(list(done_set), f)

    print(f"\n完成！共保存 {total_clips} 个片段")
    if failed:
        print(f"失败视频数：{len(failed)}")
        fail_log = os.path.join(opt.out_dir, 'failed_videos.txt')
        with open(fail_log, 'w') as f:
            for path, err in failed:
                f.write(f"{path}\t{err}\n")
        print(f"失败列表已保存到 {fail_log}")


# ═══════════════════════════════════════════════════════════════
# 六、更新后的 MotionLatentDataset（配合新格式）
# ═══════════════════════════════════════════════════════════════

# 注：将此类复制到 train_meanflow.py 中替换原有 MotionLatentDataset

UPDATED_DATASET_CODE = '''
class MotionLatentDataset(torch.utils.data.Dataset):
    """
    读取 prepare_data.py（正确版本）生成的 .pt 文件。
    支持 prev_x / prev_wa 的自回归上下文。
    """
    def __init__(self, data_root, split="train"):
        self.root  = Path(data_root) / split
        self.files = sorted(self.root.rglob("*.pt"))
        assert len(self.files) > 0, f"No .pt files found in {self.root}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        return {
            "x0"     : data["x0"].float(),       # (L, 512)
            "wa"     : data["wa"].float(),       # (L, 512)
            "wr"     : data["wr"].float(),       # (512,)
            "we"     : data["we"].float(),       # (1, 7)
            "prev_x" : data["prev_x"].float(),   # (n_prev, 512)
            "prev_wa": data["prev_wa"].float(),  # (n_prev, 512)
        }
'''


# ═══════════════════════════════════════════════════════════════
# 七、命令行入口
# ═══════════════════════════════════════════════════════════════

class PrepareOptions(BaseOptions):
    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument('--mode',
            choices=['single', 'dataset'], default='single',
            help='single: 处理单个视频; dataset: 批量处理数据集目录')
        # 单视频模式
        parser.add_argument('--video_path',
            default=None, type=str, help='[single] 视频路径 .mp4/.avi/...')
        # 数据集模式
        parser.add_argument('--data_root',
            default=None, type=str, help='[dataset] 视频数据集根目录')
        # 公共参数
        parser.add_argument('--ckpt_path',
            required=True, type=str, help='float.pth 路径')
        parser.add_argument('--out_dir',
            default='data/train', type=str, help='输出目录')
        parser.add_argument('--frame_batch_size',
            default=8, type=int, help='帧编码批大小（根据显存调整）')
        parser.add_argument('--verbose',
            action='store_true', help='打印详细错误信息')
        return parser


def load_model(opt, device):
    """加载 FLOAT_MeanFlow 并冻结所有参数（仅做特征提取）"""
    model = FLOAT_MeanFlow(opt)
    state_dict = torch.load(opt.ckpt_path, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


if __name__ == '__main__':
    opt = PrepareOptions().parse()
    opt.rank = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ── 加载模型
    print("[1/3] 加载模型...")
    opt.rank = device
    model = load_model(opt, device)

    # ── 加载音频预处理器
    wav2vec_prep = Wav2Vec2FeatureExtractor.from_pretrained(
        opt.wav2vec_model_path, local_files_only=True
    )

    # ── 初始化人脸裁剪器
    print("[2/3] 初始化人脸检测器...")
    cropper = FaceCropper(opt.input_size)

    # ── 开始处理
    print("[3/3] 开始提取特征...")
    os.makedirs(opt.out_dir, exist_ok=True)

    if opt.mode == 'single':
        assert opt.video_path, "--video_path 不能为空（single 模式）"
        n = process_single_video(
            model, wav2vec_prep, cropper,
            opt.video_path, opt.out_dir, opt,
            min_frames=int(opt.wav2vec_sec * opt.fps),
        )
        print(f"\n完成！保存了 {n} 个片段到 {opt.out_dir}")

    elif opt.mode == 'dataset':
        assert opt.data_root, "--data_root 不能为空（dataset 模式）"
        process_dataset(opt, model, wav2vec_prep, cropper)

    print("\n─── 提示 ───")
    print("请将生成的 .pt 文件按 train/val 划分：")
    print(f"  {opt.out_dir}/train/  ← 训练集")
    print(f"  {opt.out_dir}/val/    ← 验证集（可从 train 中抽取 5%）")
    print("然后更新 train_meanflow.py 中的 MotionLatentDataset：")
    print("  支持 prev_x / prev_wa 字段（已在新版 Dataset 中添加）")
    print(UPDATED_DATASET_CODE)