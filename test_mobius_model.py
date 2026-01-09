#!/usr/bin/env python3
"""
莫比乌斯量子环形网络快速测试脚本
用于验证模型实现和基本功能
"""

import torch
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_unitary_matrix_param():
    """Test Cayley unitary parameterization"""
    logging.info("Testing UnitaryMatrixParam...")
    
    from mobius_quantum_ring import CayleyUnistochasticParam
    
    # 创建酉矩阵参数化
    n = 8
    unitary_param = CayleyUnistochasticParam(n)
    
    # 获取酉矩阵
    U = unitary_param.unitary()
    
    # 检查形状
    assert U.shape == (n, n), f"Expected shape ({n}, {n}), got {U.shape}"
    
    # 检查酉性: U^† U should be close to Identity
    UdU = U.conj().transpose(-2, -1) @ U
    I = torch.eye(n, dtype=U.dtype, device=U.device)
    ortho_error = torch.norm(UdU - I, p='fro').item()
    
    assert ortho_error < 1e-4, f"Unitarity error {ortho_error} too large"
    
    logging.info(f"✓ UnitaryMatrixParam test passed (ortho_error={ortho_error:.6f})")
    return True


def test_unistochastic_weight():
    """Test H = |U|^2 is (approximately) doubly-stochastic."""
    logging.info("Testing UnistochasticWeightGenerator...")
    
    from mobius_quantum_ring import CayleyUnistochasticParam
    
    # 创建权重生成器
    dim = 8
    generator = CayleyUnistochasticParam(dim)
    
    # 生成双随机矩阵
    H = generator.unistochastic()
    
    # 检查形状
    assert H.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {H.shape}"
    
    # 检查非负性
    assert (H >= 0).all(), "H contains negative values"
    
    # 检查行和
    row_sums = H.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(dim), atol=1e-5), \
        f"Row sums not close to 1: {row_sums}"
    
    # 检查列和
    col_sums = H.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones(dim), atol=1e-5), \
        f"Column sums not close to 1: {col_sums}"
    
    logging.info(f"✓ UnistochasticWeightGenerator test passed")
    logging.info(f"  Row sums range: [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}]")
    logging.info(f"  Col sums range: [{col_sums.min().item():.6f}, {col_sums.max().item():.6f}]")
    return True


def test_mobius_ring_cell():
    """Test MoebiusQuantumRing vector dynamics and gradients."""
    logging.info("Testing MoebiusQuantumRing (vector)...")
    
    from mobius_quantum_ring import MoebiusQuantumRing
    
    input_dim = 32
    hidden_dim = 64
    output_dim = 10
    model = MoebiusQuantumRing(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        alpha=0.1,
        relaxation_steps=10,
        lora_rank=8,
        readout_dim=16,
    )
    
    x = torch.randn(4, input_dim)
    y, state = model(x, return_state=True)
    assert y.shape == (4, output_dim), f"Expected output (4,{output_dim}), got {tuple(y.shape)}"
    assert state.h.shape == (4, hidden_dim), f"Expected state (4,{hidden_dim}), got {tuple(state.h.shape)}"
    
    loss = y.sum()
    loss.backward()
    
    no_grad_params = [name for name, param in model.named_parameters() if param.grad is None]
    assert len(no_grad_params) == 0, f"Parameters without gradient: {no_grad_params}"
    
    logging.info("✓ MoebiusQuantumRing (vector) test passed")
    return True


def test_mobius_quantum_ring():
    """测试完整模型"""
    logging.info("Testing MöbiusQuantumRing...")
    
    from mobius_quantum_ring import create_mobius_model
    
    # 创建模型(小规模用于快速测试)
    model = create_mobius_model(
        num_classes=100,  # CIFAR-100
        img_size=32,
        embed_dim=128,     # ring hidden_dim
        depth=10,          # relaxation steps
        alpha=0.1,
        lora_rank=8,
        readout_dim=16,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Total parameters: {total_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)  # batch_size=2, CIFAR-100图像大小
    y = model(x)
    
    # 检查输出
    assert y.shape == (2, 100), f"Expected output shape (2, 100), got {y.shape}"
    
    # 测试正交损失计算
    ortho_loss = model.get_orthogonal_loss()
    assert ortho_loss.item() >= 0, "Orthogonal loss should be non-negative"
    
    # 测试训练步骤
    criterion = torch.nn.CrossEntropyLoss()
    target = torch.randint(0, 100, (2,))
    loss = criterion(y, target) + 0.01 * ortho_loss
    
    loss.backward()
    
    # 检查梯度
    no_grad_params = [name for name, param in model.named_parameters() 
                     if param.grad is None]
    assert len(no_grad_params) == 0, f"Parameters without gradient: {no_grad_params}"
    
    logging.info(f"✓ MöbiusQuantumRing test passed")
    logging.info(f"  Output shape: {y.shape}")
    logging.info(f"  Orthogonal loss: {ortho_loss.item():.6f}")
    logging.info(f"  Classification loss: {loss.item():.6f}")
    return True


def test_patch_encoder_forward_and_eqprop():
    """Sanity-check the optional patch embedding front-end."""
    logging.info("Testing patch encoder (forward + eqprop)...")

    from mobius_quantum_ring import create_mobius_model

    model = create_mobius_model(
        num_classes=100,
        img_size=32,
        in_channels=3,
        image_encoder="patch",
        patch_size=4,
        patch_embed_dim=64,
        patch_pool="mean",
        embed_dim=64,
        depth=6,
        alpha=0.1,
        lora_rank=16,
        readout_dim=16,
    )

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 100)

    target = torch.randint(0, 100, (2,))
    info = model.eqprop_update_step(
        x,
        target,
        lr=1e-3,
        adjoint_steps=5,
        unitary_lr_ratio=0.5,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
    )
    assert info.get("did_update", False) is True

    logging.info("✓ Patch encoder test passed")
    return True


def test_h_mix_beta_doubly_stochastic_and_learnable():
    """Check H_eff=(1-beta)I+betaH remains doubly-stochastic and beta can be learned."""
    logging.info("Testing H-mix beta (doubly-stochastic + learnable update)...")

    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(0)
    model = MoebiusQuantumRing(
        input_dim=16,
        hidden_dim=16,
        output_dim=5,
        alpha=0.1,
        relaxation_steps=5,
        lora_rank=8,
        inj_activation="none",
        state_activation="none",
        h_mix_beta=0.5,
        learnable_h_mix_beta=False,
        readout_dim=8,
    )

    H_base = model._current_H_base(device=torch.device("cpu"), dtype=torch.float32)
    H_eff = model._current_H(device=torch.device("cpu"), dtype=torch.float32)
    row_err = (H_eff.sum(dim=1) - 1.0).abs().max().item()
    col_err = (H_eff.sum(dim=0) - 1.0).abs().max().item()
    assert row_err < 1e-5 and col_err < 1e-5, "H_eff must remain doubly-stochastic"
    assert H_eff.diag().mean().item() > H_base.diag().mean().item(), "Mixing with I should increase diagonal mass"

    # Learnable beta should update under eqprop
    model2 = MoebiusQuantumRing(
        input_dim=16,
        hidden_dim=16,
        output_dim=5,
        alpha=0.1,
        relaxation_steps=5,
        lora_rank=8,
        h_mix_beta=0.5,
        learnable_h_mix_beta=True,
        readout_dim=8,
    )
    beta_before = model2._h_mix_beta_value(device=torch.device("cpu"), dtype=torch.float32).item()
    param_before = model2.h_mix_beta_param.detach().clone()

    x = torch.randn(8, 16)
    y = torch.randint(0, 5, (8,))
    _ = model2.eqprop_update_step(
        x,
        y,
        lr=1e-2,
        unitary_lr_ratio=0.5,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=5,
        h_mix_beta_lr_ratio=1.0,
    )

    beta_after = model2._h_mix_beta_value(device=torch.device("cpu"), dtype=torch.float32).item()
    param_after = model2.h_mix_beta_param.detach().clone()
    assert (param_after - param_before).abs().max().item() > 0.0, "Learnable beta parameter should update"
    # beta itself may change extremely slightly for small toy problems; the parameter update is the key check.

    logging.info("✓ H-mix beta test passed")
    return True


def test_unitary_dynamics_mode_sanity():
    """Sanity-check complex unitary dynamics: forward works and EQProp updates reduce loss on a fixed batch."""
    logging.info("Testing unitary dynamics mode (complex inference)...")

    import torch.nn.functional as F
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(0)
    model = MoebiusQuantumRing(
        input_dim=16,
        hidden_dim=32,
        output_dim=5,
        alpha=0.1,
        relaxation_steps=8,
        lora_rank=16,
        inj_activation="none",
        state_activation="none",
        dynamics_mode="unitary",
        measurement="abs",
        readout_dim=8,
        readout_mode="linear",
    )

    x = torch.randn(16, 16)
    target = torch.randint(0, 5, (16,))

    with torch.no_grad():
        logits, st = model(x, return_state=True)
        assert torch.is_complex(st.h), "Unitary dynamics should produce a complex state"
        loss0 = F.cross_entropy(logits, target).item()

    # A few EQProp steps on the same batch should reduce loss (sanity, not a benchmark).
    for _ in range(20):
        model.eqprop_update_step(
            x,
            target,
            lr=5e-2,
            unitary_lr_ratio=0.5,
            injection_lr_ratio=1.0,
            readout_lr_ratio=1.0,
            adjoint_steps=10,
        )

    with torch.no_grad():
        logits = model(x)
        loss1 = F.cross_entropy(logits, target).item()

    assert loss1 < loss0, f"Expected loss to decrease in unitary mode (got {loss0:.4f} -> {loss1:.4f})"
    logging.info("✓ Unitary dynamics mode test passed")
    return True


def test_hamiltonian_optimizer():
    """测试哈密顿优化器"""
    logging.info("Testing HamiltonianOptimizer...")
    
    from mobius_quantum_ring import HamiltonianOptimizer
    
    # 创建简单模型
    model = torch.nn.Linear(10, 10)
    
    # 创建哈密顿优化器
    optimizer = HamiltonianOptimizer(model.parameters(), lr=1e-3)
    
    # 模拟训练步骤
    x = torch.randn(4, 10)
    y = model(x).sum()
    
    optimizer.zero_grad()
    y.backward()
    optimizer.step()
    
    # 检查参数是否更新
    initial_param = model.weight.clone()
    y = model(x).sum()
    optimizer.zero_grad()
    y.backward()
    optimizer.step()
    
    param_changed = not torch.allclose(initial_param, model.weight)
    assert param_changed, "Parameters should have been updated"
    
    logging.info(f"✓ HamiltonianOptimizer test passed")
    return True


def test_model_export():
    """测试模型导出"""
    logging.info("Testing model export...")
    
    from mobius_quantum_ring import create_mobius_model
    
    # 创建小模型
    model = create_mobius_model(
        num_classes=100,
        img_size=32,
        embed_dim=64,
        depth=2,
        num_heads=2
    )
    
    # 测试保存
    checkpoint_path = '/tmp/test_mobius_checkpoint.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 2
        }
    }, checkpoint_path)
    
    # 测试加载
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logging.info(f"✓ Model export test passed")
    return True


def test_eqprop_update_step():
    """Test strict EQPROP update (no BPTT) runs and updates parameters."""
    logging.info("Testing EQPROP update step...")
    
    from mobius_quantum_ring import MoebiusQuantumRing
    
    torch.manual_seed(0)
    model = MoebiusQuantumRing(
        input_dim=16,
        hidden_dim=32,
        output_dim=5,
        alpha=0.1,
        relaxation_steps=8,
        lora_rank=4,
        readout_dim=8,
    )
    
    x = torch.randn(6, 16)
    target = torch.randint(0, 5, (6,))
    
    # Snapshot a parameter to verify update
    A_real_before = model.unitary_param.A_real.detach().clone()
    
    info = model.eqprop_update_step(
        x,
        target,
        lr=1e-2,
        unitary_lr_ratio=0.5,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=10,
    )
    
    assert "loss" in info and isinstance(info["loss"], float), "eqprop should return a float loss"
    assert "logits" in info and info["logits"].shape == (6, 5), "logits shape mismatch"
    
    A_real_after = model.unitary_param.A_real.detach().clone()
    max_diff = (A_real_after - A_real_before).abs().max().item()
    assert max_diff > 0.0, "Unitary parameters should update in eqprop mode"
    
    # Verify unitarity / doubly-stochastic constraints remain satisfied numerically
    U = model.unitary_param.unitary()
    I = torch.eye(U.size(0), dtype=U.dtype)
    ortho_err = torch.linalg.matrix_norm(U.conj().T @ U - I, ord="fro").item()
    assert ortho_err < 1e-4, f"Unitarity error too large after eqprop update: {ortho_err}"
    
    H = model.unitary_param.unistochastic()
    row_err = (H.sum(dim=1) - 1.0).abs().max().item()
    col_err = (H.sum(dim=0) - 1.0).abs().max().item()
    assert row_err < 1e-4 and col_err < 1e-4, "H should remain (approximately) doubly-stochastic"
    
    logging.info("✓ EQPROP update step test passed")
    return True


def test_eqprop_dual_unitary_and_state_targets():
    """Test online forward+reverse step with frozen U_base and learnable GT state targets."""
    logging.info("Testing EQPROP dual-unitary + learnable state targets...")

    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(0)
    model = MoebiusQuantumRing(
        input_dim=16,
        hidden_dim=32,
        output_dim=5,
        alpha=0.1,
        relaxation_steps=8,
        lora_rank=4,
        readout_dim=8,
        base_unitary_init="random",
        base_unitary_seed=123,
        base_unitary_scale=0.01,
        learnable_state_targets=True,
    )

    x = torch.randn(6, 16)
    target = torch.randint(0, 5, (6,))

    A_before = model.unitary_param.A_real.detach().clone()
    U_base_before = model.U_base.detach().clone()
    P_before = model.state_targets.detach().clone()

    info = model.eqprop_update_step(
        x,
        target,
        lr=1e-2,
        unitary_lr_ratio=0.5,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=10,
        state_target_weight=1.0,
        state_target_lr_ratio=1.0,
    )

    assert info.get("did_update", False) is True, "Should perform reverse-ring update when GT is present"

    A_after = model.unitary_param.A_real.detach().clone()
    P_after = model.state_targets.detach().clone()

    assert (A_after - A_before).abs().max().item() > 0.0, "Policy unitary params should update"
    assert (P_after - P_before).abs().max().item() > 0.0, "State targets should update"
    assert torch.allclose(model.U_base, U_base_before), "U_base should remain frozen"

    # Verify U_total stays unitary
    U_total = model._unitary_total()
    I = torch.eye(U_total.size(0), dtype=U_total.dtype, device=U_total.device)
    ortho_err = torch.linalg.matrix_norm(U_total.conj().T @ U_total - I, ord="fro").item()
    assert ortho_err < 1e-4, f"U_total unitarity error too large: {ortho_err}"

    # Verify H is (approximately) doubly-stochastic
    H = U_total.abs().pow(2)
    row_err = (H.sum(dim=1) - 1.0).abs().max().item()
    col_err = (H.sum(dim=0) - 1.0).abs().max().item()
    assert row_err < 1e-4 and col_err < 1e-4, "H should remain (approximately) doubly-stochastic"

    logging.info("✓ EQPROP dual-unitary + state targets test passed")
    return True


def test_eqprop_proto_readout():
    """Test prototype-distance readout ties classification to learnable state targets."""
    logging.info("Testing EQPROP proto readout...")

    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(0)
    model = MoebiusQuantumRing(
        input_dim=16,
        hidden_dim=32,
        output_dim=5,
        alpha=0.1,
        relaxation_steps=6,
        lora_rank=4,
        readout_dim=8,
        readout_mode="proto",
        proto_tau=1.0,
        learnable_state_targets=True,
        base_unitary_init="identity",
    )

    x = torch.randn(6, 16)
    target = torch.randint(0, 5, (6,))

    # Forward shape check
    logits, state = model(x, return_state=True)
    assert logits.shape == (6, 5)
    assert state.h.shape == (6, 32)

    # Snapshot parameters
    P_before = model.state_targets.detach().clone()
    W_readout_before = model.readout.readout.weight.detach().clone()

    info = model.eqprop_update_step(
        x,
        target,
        lr=1e-2,
        unitary_lr_ratio=0.5,
        injection_lr_ratio=1.0,
        readout_lr_ratio=1.0,
        adjoint_steps=10,
        state_target_weight=0.0,  # proto readout already uses state_targets via classification
        state_target_lr_ratio=1.0,
    )

    assert info.get("did_update", False) is True
    P_after = model.state_targets.detach().clone()
    W_readout_after = model.readout.readout.weight.detach().clone()

    assert (P_after - P_before).abs().max().item() > 0.0, "Proto targets should update from classification loss"
    assert torch.allclose(W_readout_after, W_readout_before), "Linear readout weights should remain untouched in proto mode"

    logging.info("✓ EQPROP proto readout test passed")
    return True


def run_all_tests():
    """运行所有测试"""
    logging.info("="*60)
    logging.info("Möbius Quantum Ring Network - Test Suite")
    logging.info("="*60)
    
    tests = [
        test_unitary_matrix_param,
        test_unistochastic_weight,
        test_mobius_ring_cell,
        test_mobius_quantum_ring,
        test_patch_encoder_forward_and_eqprop,
        test_h_mix_beta_doubly_stochastic_and_learnable,
        test_unitary_dynamics_mode_sanity,
        test_hamiltonian_optimizer,
        test_model_export,
        test_eqprop_update_step,
        test_eqprop_dual_unitary_and_state_targets,
        test_eqprop_proto_readout,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logging.error(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    logging.info("="*60)
    logging.info(f"Test Results: {passed} passed, {failed} failed")
    logging.info("="*60)
    
    if failed == 0:
        logging.info("🎉 All tests passed!")
        return 0
    else:
        logging.error(f"❌ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
