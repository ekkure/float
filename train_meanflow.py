"""
train_meanflow.py
─────────────────────────────────────────────────────────────
FLOAT + MeanFlow Phase-2 训练脚本（最小化改动版）

用法示例：
    torchrun --nproc_per_node=4 train_meanflow.py \
        --input_size 512 \
        --wav2vec_model_path ./checkpoints/wav2vec2-base-960h \
        --audio2emotion_path ./checkpoints/wav2vec-english-speech-emotion-recognition \
        --pretrained_autoencoder ./checkpoints/autoencoder.pt \
        --data_root /data/voxceleb2 \
        --save_dir ./checkpoints/meanflow \
        --batch_size 4 \
        --lr 1e-4 \
        --num_epochs 100

依赖：
    原版 FLOAT 所有依赖 + torch >= 2.0（torch.func.jvp 需要）
"""

import os, math, argparse, json, logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ── 导入模型
from models.float.FLOAT_meanflow import FLOAT_MeanFlow
#from base_options import BaseOptions
from options.base_options import BaseOptions

# ──────────────────────────────────────────────────────────────
# 配置解析（继承 BaseOptions 并新增 MeanFlow 专属选项）
# ──────────────────────────────────────────────────────────────

class MeanFlowOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)

        # ── 训练基础
        parser.add_argument('--data_root',   type=str, required=True)
        parser.add_argument('--save_dir',    type=str, default='./checkpoints/meanflow')
        parser.add_argument('--resume',      type=str, default=None, help='checkpoint path to resume')
        parser.add_argument('--pretrained_autoencoder', type=str, required=True,
                            help='Phase-1 autoencoder checkpoint')
        parser.add_argument('--num_epochs',  type=int,   default=100)
        parser.add_argument('--batch_size',  type=int,   default=4)
        parser.add_argument('--lr',          type=float, default=1e-4)
        parser.add_argument('--weight_decay',type=float, default=0.0)
        parser.add_argument('--adam_beta1',  type=float, default=0.9)
        parser.add_argument('--adam_beta2',  type=float, default=0.95)
        parser.add_argument('--ema_decay',   type=float, default=0.9999)
        parser.add_argument('--log_every',   type=int,   default=50)
        parser.add_argument('--save_every',  type=int,   default=5000)
        parser.add_argument('--num_workers', type=int,   default=4)
        

        #parser.add_argument('--max_grad_norm',   type=float, default=1.0)
        #parser.add_argument('--mf_p_adaptive',   type=float, default=1.0)
        #parser.add_argument('--mf_c_adaptive',   type=float, default=1e-3)

        # ── MeanFlow 专属超参
        # parser.add_argument('--mf_p_adaptive',    type=float, default=1.0,
        #                     help='自适应 loss 权重指数 p（论文 Table 1e，p=1.0 最优）')
        # parser.add_argument('--mf_c_adaptive',    type=float, default=1e-3,
        #                     help='自适应 loss 数值稳定常数')
        # parser.add_argument('--mf_ratio_r_neq_t', type=float, default=0.25,
        #                     help='r != t 的采样比例（论文 Table 1a，0.25 最优）')
        # parser.add_argument('--mf_lognorm_mu',    type=float, default=-0.4,
        #                     help='lognorm 时间采样均值（论文 Table 1d）')
        # parser.add_argument('--mf_lognorm_sigma', type=float, default=1.0,
        #                     help='lognorm 时间采样标准差')

        # ── DDP
        parser.add_argument('--rank',        type=int, default=0)
        parser.add_argument('--world_size',  type=int, default=1)
        parser.add_argument('--local_rank',  type=int, default=0)

        return parser


# ──────────────────────────────────────────────────────────────
# EMA 工具
# ──────────────────────────────────────────────────────────────

class EMA:
    """指数移动平均，用于稳定训练。"""
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = {k: v.clone().float() for k, v in model.named_parameters() if v.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.float()
                )

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype))


# ──────────────────────────────────────────────────────────────
# 数据集（占位符，需根据实际数据格式实现）
# ──────────────────────────────────────────────────────────────

class MotionLatentDataset(torch.utils.data.Dataset):
    """
    预处理好的运动潜变量数据集。
    每条样本应包含：
        x0   : (L, dim_w)   干净运动潜变量（由冻结的 motion_autoencoder 预先提取）
        wa   : (L, dim_w)   音频特征（由冻结的 audio_encoder 预先提取）
        wr   : (dim_w,)     参考身份特征
        we   : (1, dim_e)   情感特征（由冻结的 emotion_encoder 预先提取）

    建议在训练前用独立脚本将所有特征离线提取存成 .pt 文件，
    以避免训练时反复计算冻结模型的前向传播，提升训练效率。
    """
    def __init__(self, data_root, split='train'):
        self.root  = Path(data_root) / split
        self.files = sorted(self.root.rglob('*.pt'))
        assert len(self.files) > 0, f"No .pt files found in {self.root}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #data = torch.load(self.files[idx], map_location='cpu'， weights_only=True)
        data = torch.load(self.files[idx], map_location='cpu', weights_only=True)
        return {
            'x0': data['x0'].float(),    # (L, dim_w)
            'wa': data['wa'].float(),    # (L, dim_w)
            'wr': data['wr'].float(),    # (dim_w,)
            'we': data['we'].float(),    # (1, dim_e)
        }


def collate_fn(batch):
    """简单 collate，将 list of dict 转为 dict of tensors。"""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


# ──────────────────────────────────────────────────────────────
# 日志工具
# ──────────────────────────────────────────────────────────────

def setup_logger(save_dir, rank):
    logger = logging.getLogger('MeanFlow')
    logger.setLevel(logging.INFO)
    if rank == 0:
        fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────
# 保存 / 加载 checkpoint
# ──────────────────────────────────────────────────────────────

def save_checkpoint(save_dir, step, model, optimizer, ema, opt):
    ckpt = {
        'step':      step,
        'model':     model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema':       ema.shadow,
        'opt':       vars(opt),
    }
    path = os.path.join(save_dir, f'ckpt_{step:08d}.pt')
    torch.save(ckpt, path)
    # 保留最近 3 个 checkpoint
    all_ckpts = sorted(Path(save_dir).glob('ckpt_*.pt'))
    for old in all_ckpts[:-3]:
        old.unlink()
    return path


def load_checkpoint(path, model, optimizer, ema):
    ckpt = torch.load(path, map_location='cpu')
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    ema.shadow = ckpt['ema']
    return ckpt['step']


# ──────────────────────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────────────────────

def train(opt):
    # ── DDP 初始化
    is_distributed = opt.world_size > 1
    if is_distributed:
        dist.init_process_group('nccl')
        opt.rank       = dist.get_rank()
        opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        opt.world_size = dist.get_world_size()
    torch.cuda.set_device(opt.local_rank)
    device = torch.device(f'cuda:{opt.local_rank}')
    opt.rank = device   # FLOAT 内部用 opt.rank 标识设备

    # ── 日志 & 目录
    os.makedirs(opt.save_dir, exist_ok=True)
    logger = setup_logger(opt.save_dir, dist.get_rank() if is_distributed else 0)
    logger.info(f"MeanFlow Training | world_size={opt.world_size}")
    logger.info(json.dumps(vars(opt), indent=2, default=str))

    # ── 模型
    model = FLOAT_MeanFlow(opt).to(device)

    # 加载 Phase-1 autoencoder 权重
    ckpt_ae = torch.load(opt.pretrained_autoencoder, map_location='cpu')
    model.motion_autoencoder.load_state_dict(ckpt_ae, strict=False)
    model.motion_autoencoder.requires_grad_(False)
    logger.info(f"Loaded autoencoder from {opt.pretrained_autoencoder}")

    # ── DDP 包装（只训练 fmt）
    if is_distributed:
        model.fmt = DDP(model.fmt, device_ids=[opt.local_rank], find_unused_parameters=False)
    fmt_module = model.fmt.module if is_distributed else model.fmt

    # ── 优化器（只优化 FMT 参数）
    trainable_params = list(fmt_module.parameters())
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=opt.lr,
        betas=(opt.adam_beta1, opt.adam_beta2),
        weight_decay=opt.weight_decay,
    )

    # ── EMA
    ema = EMA(fmt_module, decay=opt.ema_decay)

    # ── 数据集
    dataset = MotionLatentDataset(opt.data_root, split='train')
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    loader  = DataLoader(
        dataset, batch_size=opt.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── 恢复训练
    global_step = 0
    if opt.resume:
        global_step = load_checkpoint(opt.resume, model, optimizer, ema)
        logger.info(f"Resumed from step {global_step}")

    # ──────────────────────────────────────────
    # 训练循环
    # ──────────────────────────────────────────
    model.train()
    model.motion_autoencoder.eval()   # 确保 autoencoder 始终在 eval 模式

    for epoch in range(opt.num_epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        for batch in loader:
            # 数据移至设备
            data = {k: v.to(device) for k, v in batch.items()}

            # ── 前向 + MeanFlow Loss
            optimizer.zero_grad()
            loss, loss_dict = model.forward_train(data)
            loss.backward()

            # 梯度裁剪（对 Transformer 训练重要）
            nn.utils.clip_grad_norm_(trainable_params, opt.max_grad_norm)

            optimizer.step()
            ema.update(fmt_module)
            global_step += 1

            # ── 日志
            if global_step % opt.log_every == 0:
                is_main = (not is_distributed) or (dist.get_rank() == 0)
                if is_main:
                    log_str = (
                        f"[epoch {epoch:03d} | step {global_step:07d}] "
                        f"loss={loss_dict['loss_meanflow']:.4f}  "
                        f"t={loss_dict['t_mean']:.3f}  "
                        f"r={loss_dict['r_mean']:.3f}  "
                        f"|u|={loss_dict['u_norm']:.3f}  "
                        f"|tgt|={loss_dict['tgt_norm']:.3f}"
                    )
                    logger.info(log_str)

            # ── 保存 checkpoint
            is_main = (not is_distributed) or (dist.get_rank() == 0)
            if global_step % opt.save_every == 0 and is_main:
                path = save_checkpoint(opt.save_dir, global_step, model, optimizer, ema, opt)
                logger.info(f"Saved checkpoint: {path}")

    # ── 最终保存
    is_main = (not is_distributed) or (dist.get_rank() == 0)
    if is_main:
        path = save_checkpoint(opt.save_dir, global_step, model, optimizer, ema, opt)
        logger.info(f"Training complete. Final checkpoint: {path}")

    if is_distributed:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt_parser = MeanFlowOptions()
    opt = opt_parser.parse()
    train(opt)
