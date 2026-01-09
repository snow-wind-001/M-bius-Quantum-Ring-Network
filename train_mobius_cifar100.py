#!/usr/bin/env python3
"""
莫比乌斯量子环形网络训练脚本 - CIFAR-100
基于HTML文档中的UHR (Unistochastic Hamiltonian Ring) 架构

核心特性:
1. 酉矩阵参数化 (H = |U|² 自动获得双随机性质)
2. 推理环(实部)和更新环(虚部分离
3. 哈密顿动力学优化
4. CIFAR-100数据集支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import argparse
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
import math

# 导入我们的模型
from mobius_quantum_ring import (
    MöbiusQuantumRing,
    HamiltonianOptimizer,
    create_mobius_model
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mobius_training.log"),
        logging.StreamHandler()
    ]
)


def get_cifar100_loaders(batch_size=64, num_workers=4, download=True):
    """
    获取CIFAR-100数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载工作线程数
        download: 是否下载数据集
        
    Returns:
        train_loader, test_loader
    """
    # CIFAR-100的均值和标准差
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # 训练数据增强
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5)
    ])
    
    # 测试数据转换
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 加载CIFAR-100数据集
    train_set = datasets.CIFAR100(
        root='./data',
        train=True,
        download=download,
        transform=transform_train
    )
    
    test_set = datasets.CIFAR100(
        root='./data',
        train=False,
        download=download,
        transform=transform_test
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def mixup_data(x, y, alpha=0.2):
    """
    Mixup数据增强
    
    Args:
        x: 输入数据
        y: 标签
        alpha: Beta分布参数
        
    Returns:
        mixed_x, y_a, y_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with soft targets:
      L = - mean_b sum_c target[b,c] * log_softmax(logits[b,c])
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def one_hot_targets(y: torch.Tensor, num_classes: int, *, device: torch.device) -> torch.Tensor:
    t = torch.zeros(y.size(0), num_classes, device=device, dtype=torch.float32)
    return t.scatter_(1, y.view(-1, 1), 1.0)


def apply_label_smoothing(target: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return target
    C = target.size(1)
    return (1.0 - eps) * target + (eps / float(C))


def rand_bbox(size, lam):
    """Generate a random rectangle bbox for CutMix."""
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix augmentation. Returns mixed images + (y_a, y_b, lam)."""
    if alpha <= 0:
        raise ValueError("cutmix alpha must be > 0")
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0), device=x.device)
    y_a = y
    y_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_mixed = x.clone()
    x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to exactly match pixel ratio.
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(x.size(2) * x.size(3)))
    return x_mixed, y_a, y_b, float(lam)


def compute_cosine_lr(epoch: int, total_epochs: int, base_lr: float, *, warmup_epochs: int = 0, min_lr_ratio: float = 0.01) -> float:
    """Linear warmup + cosine decay to base_lr * min_lr_ratio."""
    if total_epochs <= 0:
        return base_lr
    warmup_epochs = int(max(0, warmup_epochs))
    min_lr = base_lr * float(min_lr_ratio)
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

def load_model_state_dict_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Load a checkpoint state_dict into a model.

    This first tries a strict load. If it fails due to shape mismatches (e.g. when
    changing LoRA rank), it falls back to loading only compatible tensors and will
    *partially copy* known-expandable tensors (currently: LoRA injection weights).

    Returns:
        A dict with load mode ("strict" or "flexible") and a brief summary.
    """
    try:
        model.load_state_dict(state_dict)
        return {"mode": "strict", "loaded": len(state_dict), "partial": 0, "skipped": 0}
    except RuntimeError as e:
        logging.warning(f"Strict state_dict load failed; falling back to flexible load: {e}")

        model_sd = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        partial_keys: List[str] = []
        skipped_keys: List[str] = []

        for k, v in state_dict.items():
            if k not in model_sd:
                skipped_keys.append(k)
                continue

            if model_sd[k].shape == v.shape:
                filtered[k] = v
                continue

            # Allow rank expansion/shrink for LoRA injection weights by copying overlap.
            if k.endswith("ring.injection.down.weight") or k.endswith("ring.injection.up.weight"):
                new_t = model_sd[k].clone()
                # Both are 2D weights.
                r = min(new_t.shape[0], v.shape[0])
                c = min(new_t.shape[1], v.shape[1])
                new_t[:r, :c] = v[:r, :c]
                filtered[k] = new_t
                partial_keys.append(k)
            else:
                skipped_keys.append(k)

        model.load_state_dict(filtered, strict=False)

        # Log a compact summary (avoid spamming huge key lists).
        logging.info(
            f"Flexible load summary: loaded={len(filtered)}/{len(state_dict)}, "
            f"partial={len(partial_keys)}, skipped={len(skipped_keys)}"
        )
        if partial_keys:
            logging.info(f"  Partial-copied tensors (first 5): {partial_keys[:5]}")
        if skipped_keys:
            logging.info(f"  Skipped tensors (first 5): {skipped_keys[:5]}")

        return {
            "mode": "flexible",
            "loaded": len(filtered),
            "partial": len(partial_keys),
            "skipped": len(skipped_keys),
            "error": str(e),
        }


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    encoder_optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_hamiltonian: bool = False,
    use_eqprop: bool = False,
    eqprop_lr: float = 3e-4,
    eqprop_unitary_lr_ratio: float = 0.5,
    eqprop_injection_lr_ratio: float = 1.0,
    eqprop_readout_lr_ratio: float = 1.0,
    eqprop_adjoint_steps: int = 20,
    eqprop_state_target_weight: float = 0.0,
    eqprop_state_target_lr_ratio: float = 1.0,
    eqprop_h_mix_beta_lr_ratio: float = 1.0,
    eqprop_encoder_lr_ratio: float = 1.0,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.2,
    mixup_prob: float = 0.5,
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 0.0,
    ortho_loss_weight: float = 0.01
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        use_hamiltonian: 是否使用哈密顿优化器
        mixup_alpha: Mixup参数
        mixup_prob: Mixup概率
        ortho_loss_weight: 正交损失权重
        
    Returns:
        训练指标字典
    """
    model.train()
    
    running_loss = 0.0
    running_ortho_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        if use_eqprop:
            # Strict HTML mode: no BPTT/autograd through relaxation.
            pass
        else:
            # 清零梯度
            optimizer.zero_grad()
        
        # 应用Mixup
        apply_cutmix = False
        apply_mixup = False
        lam = 1.0
        targets_a = None
        targets_b = None

        r = np.random.rand()
        if cutmix_prob > 0.0 and r < cutmix_prob:
            apply_cutmix = True
            data, targets_a, targets_b, lam = cutmix_data(data, target, cutmix_alpha)
        else:
            r2 = np.random.rand()
            if mixup_prob > 0.0 and r2 < mixup_prob:
                apply_mixup = True
                data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha)
        
        if use_eqprop:
            # In eqprop mode, we update parameters directly inside the model
            # using the HTML fixed-point + adjoint-state procedure.
            num_classes = 100
            soft_target = None
            if apply_mixup or apply_cutmix or (label_smoothing and label_smoothing > 0.0):
                if apply_mixup or apply_cutmix:
                    assert targets_a is not None and targets_b is not None
                    ta = one_hot_targets(targets_a, num_classes, device=device)
                    tb = one_hot_targets(targets_b, num_classes, device=device)
                    soft_target = lam * ta + (1.0 - lam) * tb
                else:
                    soft_target = one_hot_targets(target, num_classes, device=device)
                soft_target = apply_label_smoothing(soft_target, float(label_smoothing))

            if soft_target is not None:
                step_info = model.eqprop_update_step(
                    data,
                    soft_target,
                    lr=eqprop_lr,
                    unitary_lr_ratio=eqprop_unitary_lr_ratio,
                    injection_lr_ratio=eqprop_injection_lr_ratio,
                    readout_lr_ratio=eqprop_readout_lr_ratio,
                    adjoint_steps=eqprop_adjoint_steps,
                    state_target_weight=eqprop_state_target_weight,
                    state_target_lr_ratio=eqprop_state_target_lr_ratio,
                    h_mix_beta_lr_ratio=eqprop_h_mix_beta_lr_ratio,
                    encoder_lr_ratio=eqprop_encoder_lr_ratio,
                    encoder_optimizer=encoder_optimizer,
                )
                output = step_info["logits"]
                cls_loss = torch.tensor(step_info["loss"], device=device)
            else:
                step_info = model.eqprop_update_step(
                    data,
                    target,
                    lr=eqprop_lr,
                    unitary_lr_ratio=eqprop_unitary_lr_ratio,
                    injection_lr_ratio=eqprop_injection_lr_ratio,
                    readout_lr_ratio=eqprop_readout_lr_ratio,
                    adjoint_steps=eqprop_adjoint_steps,
                    state_target_weight=eqprop_state_target_weight,
                    state_target_lr_ratio=eqprop_state_target_lr_ratio,
                    h_mix_beta_lr_ratio=eqprop_h_mix_beta_lr_ratio,
                    encoder_lr_ratio=eqprop_encoder_lr_ratio,
                    encoder_optimizer=encoder_optimizer,
                )
                output = step_info["logits"]
                cls_loss = torch.tensor(step_info["loss"], device=device)
        else:
            # 前向传播
            output = model(data)
            
            # 计算分类损失
            if apply_mixup or apply_cutmix or (label_smoothing and label_smoothing > 0.0):
                num_classes = 100
                if apply_mixup or apply_cutmix:
                    assert targets_a is not None and targets_b is not None
                    ta = one_hot_targets(targets_a, num_classes, device=device)
                    tb = one_hot_targets(targets_b, num_classes, device=device)
                    soft_target = lam * ta + (1.0 - lam) * tb
                else:
                    soft_target = one_hot_targets(target, num_classes, device=device)
                soft_target = apply_label_smoothing(soft_target, float(label_smoothing))
                cls_loss = soft_target_cross_entropy(output, soft_target)
            else:
                cls_loss = criterion(output, target)
        
        # 计算正交约束损失
        ortho_loss = model.get_orthogonal_loss()
        
        if not use_eqprop:
            # 总损失
            total_loss = cls_loss + ortho_loss_weight * ortho_loss
            
            # 反向传播
            if use_hamiltonian:
                # 哈密顿优化器会自动处理backward
                loss = total_loss
                loss.backward()
                optimizer.step()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # 统计
        running_loss += cls_loss.item()
        running_ortho_loss += ortho_loss.item()
        
        if (not apply_mixup) and (not apply_cutmix):
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # 日志输出
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            acc = 100. * correct / total if total > 0 else 0
            
            logging.info(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)] '
                f'Loss: {cls_loss.item():.6f} '
                f'Ortho: {ortho_loss.item():.6f} '
                f'Acc: {acc:.2f}% '
                f'Time: {elapsed:.1f}s'
            )
    
    return {
        'loss': running_loss / len(train_loader),
        'ortho_loss': running_ortho_loss / len(train_loader),
        'accuracy': 100. * correct / total if total > 0 else 0
    }


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        评估指标字典
    """
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    # 每个类别的准确率统计
    class_correct = [0] * 100
    class_total = [0] * 100
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 统计每个类别的准确率
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == target[i]:
                    class_correct[label] += 1
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(100):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            class_accuracies.append(acc)
    
    avg_class_acc = np.mean(class_accuracies) if class_accuracies else 0
    
    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'avg_class_accuracy': avg_class_acc
    }


def train_mobius_quantum_ring(args):
    """
    训练莫比乌斯量子环形网络
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 数据加载器
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    logging.info(f'Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches')
    
    # 创建模型
    relaxation_steps = args.relaxation_steps if args.relaxation_steps is not None else args.depth
    model = create_mobius_model(
        num_classes=100,  # CIFAR-100
        img_size=32,
        in_channels=3,
        image_encoder=args.image_encoder,
        patch_size=args.patch_size,
        patch_embed_dim=args.patch_embed_dim,
        patch_pool=args.patch_pool,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        vit_mlp_dim=args.vit_mlp_dim,
        vit_dropout=args.vit_dropout,
        vit_pool=args.vit_pool,
        embed_dim=args.embed_dim,           # hidden_dim (ring nodes)
        depth=relaxation_steps,             # relaxation steps K
        alpha=args.alpha,                   # dissipation/injection coefficient
        lora_rank=args.lora_rank,           # LoRA rank r
        inj_activation=args.inj_activation, # injection nonlinearity (optional)
        state_activation=args.state_activation, # ring state nonlinearity (optional)
        h_mix_beta=args.h_mix_beta,
        learnable_h_mix_beta=args.learnable_h_mix_beta,
        dynamics_mode=args.dynamics_mode,
        measurement=args.measurement,
        readout_dim=args.readout_dim,       # local sampling size |S|
        readout_mode=args.readout_mode,
        proto_tau=args.proto_tau,
        base_unitary_init=args.base_unitary_init,
        base_unitary_scale=args.base_unitary_scale,
        base_unitary_seed=args.base_unitary_seed,
        learnable_state_targets=args.eqprop_learnable_state_targets,
        # legacy args kept for compatibility (ignored by the new implementation)
        num_heads=args.num_heads,
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Model created with {total_params:,} parameters ({trainable_params:,} trainable)')
    
    # 选择优化器（eqprop模式不使用PyTorch优化器更新参数）
    encoder_optimizer = None
    if args.use_eqprop:
        optimizer = None
        logging.info('Using strict EQPROP (Holomorphic Equilibrium Propagation) mode: no BPTT/autograd through relaxation')
        # Optional: train the image encoder with a standard optimizer (the ring still uses EQProp).
        if args.eqprop_encoder_optim != "none" and args.image_encoder in ("patch", "vit"):
            enc_params = []
            if getattr(model, "patch_embed", None) is not None:
                enc_params.extend(list(model.patch_embed.parameters()))
            if getattr(model, "patch_norm", None) is not None:
                enc_params.extend(list(model.patch_norm.parameters()))
            if getattr(model, "vit", None) is not None:
                enc_params.extend(list(model.vit.parameters()))
            if enc_params:
                if args.eqprop_encoder_optim == "adamw":
                    encoder_optimizer = optim.AdamW(enc_params, lr=args.lr * args.eqprop_encoder_lr_ratio, weight_decay=args.weight_decay)
                elif args.eqprop_encoder_optim == "sgd":
                    encoder_optimizer = optim.SGD(enc_params, lr=args.lr * args.eqprop_encoder_lr_ratio, momentum=0.9, weight_decay=args.weight_decay)
                logging.info(f'Eqprop encoder optimizer: {args.eqprop_encoder_optim} ({len(enc_params)} params)')
    elif args.use_hamiltonian:
        logging.info('Using Hamiltonian Optimizer')
        optimizer = HamiltonianOptimizer(
            model.parameters(),
            lr=args.lr,
            momentum=0.9
        )
    else:
        # 分离酉矩阵参数和其他参数
        unitary_params = []
        regular_params = []
        
        for name, param in model.named_parameters():
            # New MQR implementation stores A as real/imag parts under `ring.unitary_param`.
            if 'unitary_param.A_real' in name or 'unitary_param.A_imag' in name:
                unitary_params.append(param)
            else:
                regular_params.append(param)
        
        logging.info(f'Unitary params: {len(unitary_params)}, Regular params: {len(regular_params)}')
        
        optimizer = optim.AdamW([
            {'params': regular_params, 'lr': args.lr},
            {'params': unitary_params, 'lr': args.lr * args.unitary_lr_ratio}
        ], weight_decay=args.weight_decay)
    
    # 学习率调度器（eqprop模式手动计算cosine lr）
    scheduler = None
    if optimizer is not None and not args.use_hamiltonian:
        if args.warmup_epochs > 0 or abs(args.min_lr_ratio - 0.01) > 1e-12:
            def lr_lambda(ep: int) -> float:
                # warmup to 1.0, then cosine to min_lr_ratio
                if args.warmup_epochs > 0 and ep < args.warmup_epochs:
                    return float(ep + 1) / float(args.warmup_epochs)
                t = (ep - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs))
                return float(args.min_lr_ratio) + 0.5 * (1.0 - float(args.min_lr_ratio)) * (1.0 + math.cos(math.pi * t))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * float(args.min_lr_ratio)
            )
    
    # 损失函数（软标签/label smoothing 在 train_epoch 内部统一处理）
    criterion = nn.CrossEntropyLoss()
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/mobius_quantum_ring_{timestamp}')
    
    # 训练循环
    best_acc = 0.0
    start_epoch = 0
    
    # 自动查找最新checkpoint
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        # 查找save_dir中最新的checkpoint
        if os.path.exists(args.save_dir):
            import glob
            checkpoints = glob.glob(os.path.join(args.save_dir, 'checkpoint_epoch_*.pth'))
            if checkpoints:
                # 按epoch数排序
                def get_epoch(path):
                    try:
                        return int(os.path.basename(path).replace('checkpoint_epoch_', '').replace('.pth', ''))
                    except:
                        return -1
                checkpoints.sort(key=get_epoch, reverse=True)
                resume_path = checkpoints[0]
                logging.info(f'Auto-resume: found latest checkpoint {resume_path}')
    
    # 恢复训练
    if resume_path:
        if os.path.exists(resume_path):
            logging.info(f'Loading checkpoint from {resume_path}...')
            checkpoint = torch.load(resume_path, map_location=device)
            load_info = load_model_state_dict_flexible(model, checkpoint['model_state_dict'])
            if load_info.get("mode") == "flexible":
                logging.warning(
                    "Checkpoint was loaded in FLEXIBLE mode (some tensors were skipped/partially copied). "
                    "This is expected when changing shapes such as --lora-rank."
                )
            
            # 恢复optimizer状态
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info('Optimizer state restored')
                except Exception as e:
                    logging.warning(f'Could not restore optimizer state: {e}')
            
            # 恢复scheduler状态
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logging.info('Scheduler state restored')
                except Exception as e:
                    logging.warning(f'Could not restore scheduler state: {e}')
            
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            
            # 如果指定了extra_epochs，调整总epochs
            if args.extra_epochs > 0:
                args.epochs = start_epoch + args.extra_epochs
                logging.info(f'Extra epochs mode: will train {args.extra_epochs} more epochs (total: {args.epochs})')
            
            logging.info(f'Resumed from epoch {checkpoint["epoch"]}, next epoch: {start_epoch}, best_acc: {best_acc:.2f}%')
            if 'train_acc' in checkpoint:
                logging.info(f'  Last train_acc: {checkpoint["train_acc"]:.2f}%, train_loss: {checkpoint.get("train_loss", 0):.4f}')
        else:
            logging.warning(f'Checkpoint {resume_path} not found, starting from scratch')
    
    logging.info(f'Starting training for {args.epochs} epochs...')
    
    try:
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            
            # 训练
            # In eqprop mode, compute LR manually (warmup + cosine).
            eqprop_lr = args.lr
            if args.use_eqprop:
                eqprop_lr = compute_cosine_lr(
                    epoch,
                    args.epochs,
                    args.lr,
                    warmup_epochs=args.warmup_epochs,
                    min_lr_ratio=args.min_lr_ratio,
                )
            train_results = train_epoch(
                model,
                train_loader,
                optimizer,
                encoder_optimizer,
                criterion,
                device,
                epoch,
                use_hamiltonian=args.use_hamiltonian,
                use_eqprop=args.use_eqprop,
                eqprop_lr=eqprop_lr,
                eqprop_unitary_lr_ratio=args.eqprop_unitary_lr_ratio,
                eqprop_injection_lr_ratio=args.eqprop_injection_lr_ratio,
                eqprop_readout_lr_ratio=args.eqprop_readout_lr_ratio,
                eqprop_adjoint_steps=args.eqprop_adjoint_steps,
                eqprop_state_target_weight=args.eqprop_state_target_weight,
                eqprop_state_target_lr_ratio=args.eqprop_state_target_lr_ratio,
                eqprop_h_mix_beta_lr_ratio=args.h_mix_beta_lr_ratio,
                eqprop_encoder_lr_ratio=args.eqprop_encoder_lr_ratio,
                label_smoothing=args.label_smoothing,
                mixup_alpha=args.mixup_alpha,
                mixup_prob=args.mixup_prob,
                cutmix_alpha=args.cutmix_alpha,
                cutmix_prob=args.cutmix_prob,
                ortho_loss_weight=args.ortho_loss_weight
            )
            
            # 评估
            eval_results = evaluate(model, test_loader, criterion, device)
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 记录日志
            logging.info(
                f'Epoch {epoch}: '
                f'Train Acc: {train_results["accuracy"]:.2f}%, '
                f'Test Acc: {eval_results["accuracy"]:.2f}%, '
                f'Train Loss: {train_results["loss"]:.4f}, '
                f'Test Loss: {eval_results["loss"]:.4f}, '
                f'Ortho Loss: {train_results["ortho_loss"]:.6f}, '
                f'Time: {epoch_time:.1f}s'
            )
            
            # TensorBoard记录
            global_step = epoch * len(train_loader)
            writer.add_scalar('Loss/Train', train_results['loss'], global_step)
            writer.add_scalar('Loss/Test', eval_results['loss'], global_step)
            writer.add_scalar('Loss/Ortho', train_results['ortho_loss'], global_step)
            writer.add_scalar('Accuracy/Train', train_results['accuracy'], global_step)
            writer.add_scalar('Accuracy/Test', eval_results['accuracy'], global_step)
            writer.add_scalar('Accuracy/Test_ClassAvg', eval_results['avg_class_accuracy'], global_step)
            
            # 保存最佳模型
            if eval_results['accuracy'] > best_acc:
                best_acc = eval_results['accuracy']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'train_acc': train_results['accuracy'],
                    'train_loss': train_results['loss'],
                    'test_loss': eval_results['loss'],
                    'config': {
                        'embed_dim': args.embed_dim,
                        'depth': args.depth,
                        'num_heads': args.num_heads,
                        'alpha': args.alpha,
                        'lora_rank': args.lora_rank,
                        'readout_dim': args.readout_dim,
                        'readout_mode': args.readout_mode,
                    }
                }
                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
                save_path = os.path.join(args.save_dir, 'mobius_quantum_ring_best.pth')
                torch.save(checkpoint, save_path)
                logging.info(f'New best model saved: {best_acc:.2f}% accuracy')
            
            # 定期保存检查点
            if (epoch + 1) % args.save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'train_acc': train_results['accuracy'],
                    'train_loss': train_results['loss'],
                    'test_loss': eval_results['loss'],
                    'config': {
                        'embed_dim': args.embed_dim,
                        'depth': args.depth,
                        'num_heads': args.num_heads,
                        'alpha': args.alpha,
                        'lora_rank': args.lora_rank,
                        'readout_dim': args.readout_dim,
                        'readout_mode': args.readout_mode,
                    }
                }
                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
                save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, save_path)
                logging.info(f'Checkpoint saved: {save_path}')
    
    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
    
    except Exception as e:
        logging.error(f'Training failed: {e}')
        raise
    
    finally:
        writer.close()
        logging.info(f'Training completed! Best accuracy: {best_acc:.2f}%')
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train Möbius Quantum Ring on CIFAR-100')
    
    # 模型参数
    parser.add_argument('--embed-dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=20, help='Number of ring relaxation steps K (legacy name)')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--relaxation-steps', type=int, default=None,
                       help='Alias of --depth for MQR fixed-point relaxation steps K (if set, overrides --depth)')
    parser.add_argument('--image-encoder', type=str, default='flatten', choices=['flatten', 'patch', 'vit'],
                       help='Image encoder before the ring: flatten raw pixels, conv patch embedding, or a ViT backbone (recommended for high accuracy).')
    parser.add_argument('--patch-size', type=int, default=4,
                       help='Patch size for --image-encoder patch/vit (kernel=stride=patch_size).')
    parser.add_argument('--patch-embed-dim', type=int, default=256,
                       help='Embedding dim (channels) for patch encoder.')
    parser.add_argument('--patch-pool', type=str, default='mean', choices=['mean', 'flatten'],
                       help='How to aggregate patch tokens into a ring input vector: mean pooling (recommended) or flatten.')
    parser.add_argument('--vit-dim', type=int, default=384, help='ViT token dimension (only used when --image-encoder vit).')
    parser.add_argument('--vit-depth', type=int, default=12, help='Number of ViT transformer blocks (only for --image-encoder vit).')
    parser.add_argument('--vit-heads', type=int, default=6, help='Number of ViT attention heads (only for --image-encoder vit).')
    parser.add_argument('--vit-mlp-dim', type=int, default=1536, help='ViT MLP hidden dim (only for --image-encoder vit).')
    parser.add_argument('--vit-dropout', type=float, default=0.0, help='ViT dropout (only for --image-encoder vit).')
    parser.add_argument('--vit-pool', type=str, default='cls', choices=['cls', 'mean'], help='ViT pooling: cls token or mean over patch tokens.')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Dissipation/injection coefficient alpha in (0,1]')
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA injection rank r for J(x)=W_up W_down x')
    parser.add_argument('--inj-activation', type=str, default='none', choices=['none', 'relu', 'tanh', 'gelu'],
                       help='Optional nonlinearity inside the injection J(x)=W_up * act(W_down x). '
                            'Default none keeps the strict HTML linear injection.')
    parser.add_argument('--state-activation', type=str, default='none', choices=['none', 'relu', 'tanh'],
                       help='Optional nonlinearity inside the ring relaxation. '
                            'For relu/tanh the fixed-point contraction proof still holds (1-Lipschitz).')
    parser.add_argument('--h-mix-beta', type=float, default=1.0,
                       help='Self-retention mixing coefficient beta in H_eff=(1-beta)I+betaH. '
                            'beta=1 keeps the strict HTML coupling; smaller beta reduces over-mixing.')
    parser.add_argument('--learnable-h-mix-beta', action='store_true',
                       help='Make beta learnable (updated in eqprop mode via manual gradient; in autograd mode via optimizer).')
    parser.add_argument('--h-mix-beta-lr-ratio', type=float, default=1.0,
                       help='LR ratio for learnable beta update in eqprop mode.')
    parser.add_argument('--eqprop-encoder-lr-ratio', type=float, default=1.0,
                       help='LR ratio for updating image encoder parameters in eqprop mode (e.g., patch embedding).')
    parser.add_argument('--eqprop-encoder-optim', type=str, default='adamw', choices=['none', 'adamw', 'sgd'],
                       help='Optimizer for training the image encoder in eqprop mode. The ring is still updated via EQProp.')
    parser.add_argument('--dynamics-mode', type=str, default='unistochastic', choices=['unistochastic', 'unitary'],
                       help='Ring dynamics mode: unistochastic (H=|U|^2) or unitary (complex propagation with U^H).')
    parser.add_argument('--measurement', type=str, default='identity', choices=['identity', 'abs', 'real'],
                       help='State measurement used for readout/state losses. For unitary dynamics, identity defaults to abs.')
    parser.add_argument('--readout-dim', type=int, default=16,
                       help='Local projective sampling size |S| (default: first k nodes)')
    parser.add_argument('--readout-mode', type=str, default='linear', choices=['linear', 'proto'],
                       help='Readout mode: linear local readout, or prototype-distance logits (proto). '
                            'Proto mode requires --eqprop-learnable-state-targets.')
    parser.add_argument('--proto-tau', type=float, default=1.0,
                       help='Temperature tau for prototype-distance logits (proto readout).')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs for cosine LR schedule')
    parser.add_argument('--min-lr-ratio', type=float, default=0.01, help='Cosine schedule minimum LR ratio (eta_min = lr * min_lr_ratio)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--unitary-lr-ratio', type=float, default=0.5,
                       help='Learning rate ratio for unitary parameters')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing epsilon. In eqprop mode this is implemented via soft targets.')
    
    # 哈密顿优化
    parser.add_argument('--use-hamiltonian', action='store_true',
                       help='Use Hamiltonian optimizer instead of AdamW')
    parser.add_argument('--use-eqprop', action='store_true',
                       help='Strict HTML mode: Holomorphic Equilibrium Propagation (no BPTT/autograd through relaxation)')
    parser.add_argument('--eqprop-adjoint-steps', type=int, default=20,
                       help='Adjoint fixed-point solver iterations (h^dagger) for eqprop mode')
    parser.add_argument('--eqprop-unitary-lr-ratio', type=float, default=0.5,
                       help='LR ratio for unitary manifold parameters (A_real/A_imag) in eqprop mode')
    parser.add_argument('--eqprop-injection-lr-ratio', type=float, default=1.0,
                       help='LR ratio for LoRA injection parameters in eqprop mode')
    parser.add_argument('--eqprop-readout-lr-ratio', type=float, default=1.0,
                       help='LR ratio for readout parameters in eqprop mode')
    parser.add_argument('--eqprop-learnable-state-targets', action='store_true',
                       help='Enable learnable GT equilibrium targets (class prototypes in hidden state space)')
    parser.add_argument('--eqprop-state-target-weight', type=float, default=0.0,
                       help='Weight for GT equilibrium-state matching loss in hidden space (requires --eqprop-learnable-state-targets)')
    parser.add_argument('--eqprop-state-target-lr-ratio', type=float, default=1.0,
                       help='LR ratio for GT equilibrium target parameters in eqprop mode')

    # Dual-unitary / frozen world-model unitary
    parser.add_argument('--base-unitary-init', type=str, default='identity', choices=['identity', 'random'],
                       help='Frozen world-model unitary U_base initialization (identity=backward compatible)')
    parser.add_argument('--base-unitary-scale', type=float, default=0.01,
                       help='Scale for random U_base init (Cayley skew-Hermitian scale)')
    parser.add_argument('--base-unitary-seed', type=int, default=None,
                       help='Seed for random U_base init (for reproducibility)')
    
    # Mixup参数
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--mixup-prob', type=float, default=0.5, help='Mixup probability')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0, help='CutMix alpha parameter')
    parser.add_argument('--cutmix-prob', type=float, default=0.0, help='CutMix probability (applied before mixup)')
    
    # 正交约束
    parser.add_argument('--ortho-loss-weight', type=float, default=0.01,
                       help='Weight for orthogonal constraint loss')
    
    # 其他参数
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=20, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--extra-epochs', type=int, default=0, 
                       help='Extra epochs to train beyond checkpoint epoch (use with --resume)')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Auto-resume from latest checkpoint in save-dir')
    
    args = parser.parse_args()
    
    # 打印配置
    logging.info('=' * 60)
    logging.info('Möbius Quantum Ring Training Configuration')
    logging.info('=' * 60)
    for arg in vars(args):
        logging.info(f'{arg}: {getattr(args, arg)}')
    logging.info('=' * 60)
    
    # 开始训练
    best_accuracy = train_mobius_quantum_ring(args)
    logging.info(f'Final best accuracy: {best_accuracy:.2f}%')


if __name__ == '__main__':
    main()
