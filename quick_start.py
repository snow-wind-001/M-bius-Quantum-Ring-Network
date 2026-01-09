#!/usr/bin/env python3
"""
莫比乌斯量子环形网络 - 快速开始示例

展示如何:
1. 创建模型
2. 进行前向传播
3. 计算损失
4. 训练一个batch
5. 保存和加载模型
"""

import torch
import torch.nn as nn

from mobius_quantum_ring import create_mobius_model


def quick_start_example():
    """快速开始示例"""
    print("="*60)
    print("莫比乌斯量子环形网络 - 快速开始")
    print("="*60)
    
    # 1. 创建模型
    print("\n1. 创建模型...")
    model = create_mobius_model(
        num_classes=100,  # CIFAR-100
        img_size=32,
        patch_size=4,
        embed_dim=256,    # 较小的嵌入维度用于快速测试
        depth=6,          # 较少的层数
        num_heads=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型创建成功!")
    print(f"   总参数量: {total_params:,}")
    
    # 2. 准备输入数据
    print("\n2. 准备输入数据...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)  # CIFAR-100图像
    labels = torch.randint(0, 100, (batch_size,))
    print(f"   输入形状: {images.shape}")
    print(f"   标签形状: {labels.shape}")
    
    # 3. 前向传播
    print("\n3. 前向传播...")
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    print(f"   输出形状: {outputs.shape}")
    print(f"   预测类别: {outputs.argmax(dim=1)}")
    print(f"   真实标签: {labels}")
    
    # 4. 计算损失
    print("\n4. 计算损失...")
    criterion = nn.CrossEntropyLoss()
    cls_loss = criterion(outputs, labels)
    ortho_loss = model.get_orthogonal_loss()
    total_loss = cls_loss + 0.01 * ortho_loss
    
    print(f"   分类损失: {cls_loss.item():.4f}")
    print(f"   正交约束损失: {ortho_loss.item():.6f}")
    print(f"   总损失: {total_loss.item():.4f}")
    
    # 5. 训练一个batch
    print("\n5. 训练一个batch...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # 前向传播
    outputs = model(images)
    cls_loss = criterion(outputs, labels)
    ortho_loss = model.get_orthogonal_loss()
    total_loss = cls_loss + 0.01 * ortho_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"   训练batch完成!")
    print(f"   Loss: {total_loss.item():.4f}")
    
    # 验证梯度流动
    has_grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for _ in model.parameters())
    print(f"   梯度流动: {has_grad_count}/{total_params_count} 参数有梯度")
    
    # 6. 保存和加载模型
    print("\n6. 保存和加载模型...")
    checkpoint_path = '/tmp/mobius_quick_start.pth'
    
    # 保存
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
        'config': {
            'embed_dim': 256,
            'depth': 6,
            'num_heads': 8
        }
    }, checkpoint_path)
    print(f"   模型已保存到: {checkpoint_path}")
    
    # 加载
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   模型已加载!")
    print(f"   加载的损失: {checkpoint['loss']:.4f}")
    
    # 7. 推理示例
    print("\n7. 推理示例...")
    model.eval()
    
    # 模拟测试数据
    test_images = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        predictions = model(test_images)
        probabilities = torch.softmax(predictions, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, 5)
    
    print(f"   测试样本1:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        print(f"      Top-{i+1}: 类别 {idx.item():3d}, 概率 {prob.item()*100:.2f}%")
    
    print("\n" + "="*60)
    print("快速开始示例完成!")
    print("="*60)
    
    print("\n下一步:")
    print("1. 运行完整训练: python train_mobius_cifar100.py --epochs 200")
    print("2. 查看训练进度: tensorboard --logdir runs/")
    print("3. 测试模型: python test_mobius_model.py")
    print("\n更多信息请参考 README.md")


def model_architecture_demo():
    """模型架构演示"""
    print("\n" + "="*60)
    print("模型架构详解")
    print("="*60)
    
    from mobius_quantum_ring import CayleyUnistochasticParam, MoebiusQuantumRing
    
    # 1. Unitary parameterization (Cayley) demo
    print("\n1. Unitary parameterization (CayleyUnistochasticParam)")
    print("-" * 40)
    n = 4
    unitary_param = CayleyUnistochasticParam(n)
    U = unitary_param.unitary()
    
    print(f"   酉矩阵 U (shape: {U.shape}):")
    print(f"   {U}")
    
    # 验证酉性: U^† U ≈ I
    UdU = U.conj().transpose(-2, -1) @ U
    print(f"\n   验证酉性 (U^† U ≈ I):")
    print(f"   {UdU}")
    print(f"   与单位矩阵的误差: {torch.norm(UdU - torch.eye(n)).item():.6f}")
    
    # 2. Unistochastic / doubly-stochastic H demo
    print("\n2. Unistochastic weights (H = |U|^2)")
    print("-" * 40)
    dim = 4
    generator = CayleyUnistochasticParam(dim)
    H = generator.unistochastic()
    
    print(f"   双随机矩阵 H = |U|² (shape: {H.shape}):")
    print(f"   {H}")
    
    print(f"\n   验证双随机性质:")
    print(f"   行和: {H.sum(dim=1)}")
    print(f"   列和: {H.sum(dim=0)}")
    print(f"   所有元素非负: {(H >= 0).all()}")
    
    # 3. MQR ring dynamics (vector) demo
    print("\n3. Möbius Quantum Ring (vector) demo")
    print("-" * 40)
    input_dim = 64
    hidden_dim = 32
    output_dim = 10
    ring = MoebiusQuantumRing(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        alpha=0.1,
        relaxation_steps=10,
        lora_rank=8,
        readout_dim=16,
    )
    
    x = torch.randn(2, input_dim)
    y, state = ring(x, return_state=True)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   State shape: {state.h.shape}")
    
    # 4. 完整模型架构
    print("\n4. 完整模型架构 (MöbiusQuantumRing)")
    print("-" * 40)
    model = create_mobius_model(
        num_classes=100,
        embed_dim=128,  # ring hidden_dim (nodes)
        depth=10,       # relaxation steps
        lora_rank=8,
        readout_dim=16,
        alpha=0.1,
    )
    
    print(f"   模型组件:")
    print(f"   - Headless ring dynamics (fixed-point relaxation)")
    print(f"   - Unistochastic manifold: H = |U|^2 with U from Cayley transform")
    print(f"   - LoRA injection: J(x)=W_up W_down x")
    print(f"   - Local sampling readout: first k nodes -> classifier")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   总参数量: {total_params:,}")
    
    print(f"\n   Parameter count (total): {total_params:,}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # 运行快速开始示例
    quick_start_example()
    
    # 运行模型架构演示
    model_architecture_demo()
    
    print("\n✓ 所有示例运行完成!")
