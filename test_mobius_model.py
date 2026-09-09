#!/usr/bin/env python3
"""
莫比乌斯量子环形网络快速测试脚本
用于验证模型实现和基本功能
"""

import logging
import math
import sys

import torch

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


def test_cayley_phase_recovers_every_unistochastic_representative():
    """A global phase removes Cayley's -1 obstruction without changing H."""
    logging.info("Testing Cayley-to-unistochastic coverage despite eigenvalue -1...")

    from mobius_quantum_ring import CayleyUnistochasticParam

    torch.manual_seed(3)
    n = 5
    raw = torch.randn(n, n, dtype=torch.complex128)
    basis, _ = torch.linalg.qr(raw)
    phases = torch.tensor(
        [math.pi, -1.1, -0.2, 0.7, 1.8], dtype=torch.float64
    )
    original = basis @ torch.diag(torch.exp(1j * phases)) @ basis.conj().T
    identity = torch.eye(n, dtype=torch.complex128)
    assert torch.linalg.svdvals(identity + original).min().item() < 1e-12

    parameter = CayleyUnistochasticParam(n, coordinate_mode="minimal").double()
    diagnostics = parameter.set_from_unitary_representative_(original)
    assert diagnostics["phase_candidate_count"] > n
    assert diagnostics["selected_margin_sigma_min"] > 1e-3
    assert diagnostics["transition_reconstruction_error_fro"] < 1e-12
    torch.testing.assert_close(
        parameter.unistochastic(),
        original.abs().square(),
        rtol=1e-12,
        atol=1e-12,
    )
    logging.info("✓ Cayley phase choice preserves the complete unistochastic image")
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


def test_cayley_pullback_matches_autograd():
    """The analytic Cayley pullback must match PyTorch's complex autograd."""
    logging.info("Testing exact Cayley gradient pullback...")

    from mobius_quantum_ring import CayleyUnistochasticParam

    torch.manual_seed(7)
    param = CayleyUnistochasticParam(5).double()
    with torch.no_grad():
        param.A_real.mul_(12.0)
        param.A_imag.mul_(12.0)

    U = param.unitary()
    grad_H = torch.randn(5, 5, dtype=torch.float64)
    loss = (grad_H * U.abs().square()).sum()
    loss.backward()

    with torch.no_grad():
        grad_U = 2.0 * grad_H.to(U.dtype) * U
        grad_A = param.cayley_pullback(grad_U)

    torch.testing.assert_close(grad_A.real, param.A_real.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(grad_A.imag, param.A_imag.grad, rtol=1e-10, atol=1e-10)
    logging.info("✓ Exact Cayley pullback matches autograd")
    return True


def test_eqprop_implicit_gradients_match_autograd():
    """Converged real-ring adjoints should reproduce exact equilibrium gradients."""
    logging.info("Testing implicit real-ring gradients against autograd...")

    import torch.nn.functional as F
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(11)
    model = MoebiusQuantumRing(
        input_dim=5,
        hidden_dim=6,
        output_dim=3,
        alpha=0.3,
        relaxation_steps=160,
        lora_rank=4,
        readout_dim=6,
        h_mix_beta=0.7,
        base_unitary_init="random",
        base_unitary_scale=0.05,
        base_unitary_seed=5,
    ).double()
    with torch.no_grad():
        model.unitary_param.A_real.mul_(12.0)
        model.unitary_param.A_imag.mul_(12.0)

    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 1])
    logits, state = model(x, return_state=True)
    F.cross_entropy(logits, target).backward()

    with torch.no_grad():
        grad_y = logits.detach().softmax(dim=1)
        grad_y[torch.arange(target.numel()), target] -= 1.0
        grad_y /= target.numel()
        grad_h = grad_y @ model.readout.readout.weight
        h_star = state.h.detach()
        p = model.compute_adjoint_state_from_grad_h(h_star, grad_h, steps=240)

        grad_H_eff = model.approx_grad_H(h_star, p, normalize=False)
        beta = model._h_mix_beta_value(device=x.device, dtype=x.dtype)
        U = model._unitary_total()
        grad_U_total = 2.0 * (beta * grad_H_eff).to(U.dtype) * U
        grad_U_policy = grad_U_total @ model.U_base.to(U.dtype).conj().transpose(-2, -1)
        grad_A = model.unitary_param.cayley_pullback(grad_U_policy)

        pre = model.injection.down(x)
        z = model.injection._act(pre)
        p_eff = p * model._state_act_prime_from_h(h_star)
        grad_up = p_eff.transpose(0, 1) @ z
        dz = (p_eff @ model.injection.up.weight) * model.injection._act_prime(pre=pre, act=z)
        grad_down = dz.transpose(0, 1) @ x

    torch.testing.assert_close(grad_A.real, model.unitary_param.A_real.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(grad_A.imag, model.unitary_param.A_imag.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(grad_up, model.injection.up.weight.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(grad_down, model.injection.down.weight.grad, rtol=1e-8, atol=1e-10)
    logging.info("✓ Real-ring implicit gradients match autograd")
    return True


def test_complex_implicit_gradient_matches_autograd():
    """The complex unitary path must use the same exact Cayley pullback."""
    logging.info("Testing implicit complex-ring gradient against autograd...")

    import torch.nn.functional as F
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(17)
    model = MoebiusQuantumRing(
        input_dim=5,
        hidden_dim=6,
        output_dim=3,
        alpha=0.3,
        relaxation_steps=160,
        lora_rank=4,
        readout_dim=6,
        dynamics_mode="unitary",
        measurement="abs",
    ).double()
    with torch.no_grad():
        model.unitary_param.A_real.mul_(12.0)
        model.unitary_param.A_imag.mul_(12.0)

    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 1])
    logits, state = model(x, return_state=True)
    F.cross_entropy(logits, target).backward()

    with torch.no_grad():
        grad_y = logits.detach().softmax(dim=1)
        grad_y[torch.arange(target.numel()), target] -= 1.0
        grad_y /= target.numel()
        grad_measured = grad_y @ model.readout.readout.weight
        grad_h = model._pullback_measured_grad(state.h.detach(), grad_measured)
        p = model.compute_adjoint_state_from_grad_h(state.h.detach(), grad_h, steps=240)
        grad_U = ((1.0 - model.alpha) / model.alpha) * (p.conj().transpose(0, 1) @ state.h.detach())
        grad_A = model.unitary_param.cayley_pullback(grad_U)

    torch.testing.assert_close(grad_A.real, model.unitary_param.A_real.grad, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(grad_A.imag, model.unitary_param.A_imag.grad, rtol=1e-8, atol=1e-10)
    logging.info("✓ Complex-ring implicit gradient matches autograd")
    return True


def test_eqprop_unitary_update_is_descent_and_batch_invariant():
    """A small unitary-only step should descend and not change when a batch is duplicated."""
    logging.info("Testing unitary descent direction and batch-size invariance...")

    import copy
    import torch.nn.functional as F
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(23)
    model = MoebiusQuantumRing(
        input_dim=5,
        hidden_dim=7,
        output_dim=3,
        alpha=0.3,
        relaxation_steps=120,
        lora_rank=4,
        readout_dim=7,
    ).double()
    with torch.no_grad():
        model.unitary_param.A_real.mul_(12.0)
        model.unitary_param.A_imag.mul_(12.0)
    duplicate_model = copy.deepcopy(model)

    x = torch.randn(5, 5, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 1, 0])
    before = F.cross_entropy(model(x), target).item()

    kwargs = dict(
        lr=0.1,
        unitary_lr_ratio=1.0,
        injection_lr_ratio=0.0,
        readout_lr_ratio=0.0,
        adjoint_steps=180,
    )
    model.eqprop_update_step(x, target, **kwargs)
    duplicate_model.eqprop_update_step(torch.cat([x, x]), torch.cat([target, target]), **kwargs)
    after = F.cross_entropy(model(x), target).item()

    assert after < before, f"Expected a descent step, got {before:.12f} -> {after:.12f}"
    torch.testing.assert_close(
        model.unitary_param.A_real,
        duplicate_model.unitary_param.A_real,
        rtol=1e-9,
        atol=1e-11,
    )
    torch.testing.assert_close(
        model.unitary_param.A_imag,
        duplicate_model.unitary_param.A_imag,
        rtol=1e-9,
        atol=1e-11,
    )
    assert model.unitary_param.unitary_error_fro().item() < 1e-10
    logging.info("✓ Unitary update descends and is batch-size invariant")
    return True


def test_implicit_output_gradient_matches_cross_entropy_update():
    """The generic dL/dlogits API must reuse the exact CE update path."""
    logging.info("Testing generic implicit output-gradient update...")

    import copy
    import torch.nn.functional as F
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(31)
    reference = MoebiusQuantumRing(
        input_dim=6,
        hidden_dim=8,
        output_dim=4,
        alpha=0.25,
        relaxation_steps=80,
        lora_rank=3,
        readout_dim=8,
        state_activation="tanh",
    ).double()
    generic = copy.deepcopy(reference)
    x = torch.randn(4, 6, dtype=torch.float64)
    target = torch.tensor([0, 2, 1, 3])
    logits = generic(x).detach()
    loss = F.cross_entropy(logits, target)
    grad_logits = torch.softmax(logits, dim=1)
    grad_logits[torch.arange(target.numel()), target] -= 1.0
    grad_logits /= target.numel()

    kwargs = dict(
        lr=0.02,
        unitary_lr_ratio=0.4,
        injection_lr_ratio=0.8,
        readout_lr_ratio=1.0,
        adjoint_steps=100,
    )
    expected = reference.eqprop_update_step(x, target, **kwargs)
    actual = generic.implicit_update_from_output_gradient(
        x,
        grad_logits,
        loss_value=loss,
        **kwargs,
    )

    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        reference.named_parameters(), generic.named_parameters()
    ):
        assert name_a == name_b
        torch.testing.assert_close(parameter_a, parameter_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expected["h_dag"], actual["h_dag"], rtol=0.0, atol=0.0)
    assert abs(actual["loss"] - float(loss.item())) < 1e-12
    assert actual["unitary_error"] < 1e-10
    logging.info("✓ Generic output-gradient update matches the CE transaction")
    return True


def test_minimal_cayley_coordinates_and_mixing_rank():
    """Minimal coordinates must span u(N) without changing A, U, or H."""
    logging.info("Testing minimal Cayley coordinates and local mixing rank...")

    from mobius_quantum_ring import CayleyUnistochasticParam

    torch.manual_seed(37)
    n = 4
    raw = torch.randn(n, n, dtype=torch.complex128)
    A = 0.5 * (raw - raw.conj().transpose(0, 1))
    projected = CayleyUnistochasticParam(n, coordinate_mode="projected").double()
    minimal = CayleyUnistochasticParam(n, coordinate_mode="minimal").double()
    projected.set_from_skew_hermitian_(A)
    minimal.set_from_skew_hermitian_(A)

    torch.testing.assert_close(
        projected.skew_hermitian_A(), minimal.skew_hermitian_A(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(projected.unitary(), minimal.unitary(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        projected.unistochastic(), minimal.unistochastic(), rtol=0.0, atol=0.0
    )
    assert projected.raw_parameter_count == 2 * n * n
    assert minimal.raw_parameter_count == minimal.effective_dof == n * n

    grad_H = torch.randn(n, n, dtype=torch.float64)
    loss = (grad_H * minimal.unistochastic()).sum()
    loss.backward()
    with torch.no_grad():
        U = minimal.unitary()
        grad_A = minimal.cayley_pullback(2.0 * grad_H.to(U.dtype) * U)
        coordinate_gradients = minimal.coordinate_gradients(grad_A)
    torch.testing.assert_close(
        coordinate_gradients["A_real"], minimal.A_real.grad, rtol=1e-10, atol=1e-11
    )
    torch.testing.assert_close(
        coordinate_gradients["A_imag"], minimal.A_imag.grad, rtol=1e-10, atol=1e-11
    )

    assert minimal.modulus_square_jacobian_rank() == (n - 1) ** 2
    identity = CayleyUnistochasticParam(n, coordinate_mode="minimal").double()
    with torch.no_grad():
        identity.A_real.zero_()
        identity.A_imag.zero_()
    assert identity.modulus_square_jacobian_rank() == 0
    logging.info("✓ Minimal coordinates span u(N); |U|² degeneracy is diagnosed")
    return True


def test_nonlinear_certified_adjoint_matches_direct_and_autograd():
    """Certified tanh fixed points must yield the exact nonlinear adjoint."""
    logging.info("Testing certified nonlinear implicit gradient...")

    import torch.nn.functional as F
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(41)
    model = MoebiusQuantumRing(
        input_dim=5,
        hidden_dim=6,
        output_dim=3,
        alpha=0.15,
        relaxation_steps=600,
        relaxation_tol=1e-13,
        relaxation_min_steps=2,
        lora_rank=4,
        readout_dim=6,
        state_activation="tanh",
        h_mix_beta=0.65,
    ).double()
    with torch.no_grad():
        model.unitary_param.A_real.mul_(15.0)
        model.unitary_param.A_imag.mul_(15.0)

    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.tensor([0, 2, 1, 2])
    logits, state = model(x, return_state=True)
    assert state.converged is True
    F.cross_entropy(logits, target).backward()

    with torch.no_grad():
        grad_y = logits.detach().softmax(dim=1)
        grad_y[torch.arange(target.numel()), target] -= 1.0
        grad_y /= target.numel()
        grad_h = grad_y @ model.readout.readout.weight
        p_direct, direct_info = model.solve_adjoint_state_from_grad_h(
            state.h.detach(), grad_h, return_info=True
        )
        p_iter, iter_info = model.compute_adjoint_state_from_grad_h(
            state.h.detach(),
            grad_h,
            steps=800,
            tol=1e-13,
            min_steps=2,
            return_info=True,
        )
        torch.testing.assert_close(p_iter, p_direct, rtol=1e-9, atol=2e-11)
        assert direct_info["converged"] and iter_info["converged"]

        grad_H_eff = model.approx_grad_H(state.h.detach(), p_direct, normalize=False)
        beta = model._h_mix_beta_value(device=x.device, dtype=x.dtype)
        U = model._unitary_total()
        grad_U = 2.0 * (beta * grad_H_eff).to(U.dtype) * U
        grad_A = model.unitary_param.cayley_pullback(grad_U)
        coordinate_gradients = model.unitary_param.coordinate_gradients(grad_A)

        pre = model.injection.down(x)
        z = model.injection._act(pre)
        p_eff = p_direct * model._state_act_prime_from_h(state.h.detach())
        grad_up = p_eff.transpose(0, 1) @ z
        dz = (p_eff @ model.injection.up.weight) * model.injection._act_prime(
            pre=pre, act=z
        )
        grad_down = dz.transpose(0, 1) @ x

    torch.testing.assert_close(
        coordinate_gradients["A_real"],
        model.unitary_param.A_real.grad,
        rtol=2e-8,
        atol=2e-10,
    )
    torch.testing.assert_close(
        coordinate_gradients["A_imag"],
        model.unitary_param.A_imag.grad,
        rtol=2e-8,
        atol=2e-10,
    )
    torch.testing.assert_close(
        grad_up, model.injection.up.weight.grad, rtol=2e-8, atol=2e-10
    )
    torch.testing.assert_close(
        grad_down, model.injection.down.weight.grad, rtol=2e-8, atol=2e-10
    )
    logging.info("✓ Certified nonlinear adjoint matches direct solve and autograd")
    return True


def test_inexact_solver_update_is_rejected_atomically():
    """A requested but failed certificate must mutate neither parameters nor OGD."""
    logging.info("Testing atomic rejection of an inexact implicit update...")

    import copy
    from mqr.online import OrthogonalGradientMemory
    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(43)
    model = MoebiusQuantumRing(
        input_dim=4,
        hidden_dim=5,
        output_dim=2,
        alpha=0.01,
        relaxation_steps=2,
        relaxation_tol=1e-14,
        lora_rank=2,
        readout_dim=5,
    ).double()
    override = copy.deepcopy(model)
    x = torch.randn(3, 4, dtype=torch.float64)
    target = torch.tensor([0, 1, 0])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    memory = OrthogonalGradientMemory(max_rank=2)

    rejected = model.eqprop_update_step(
        x,
        target,
        lr=0.1,
        adjoint_steps=2,
        orthogonal_memory=memory,
        remember_gradient=True,
    )
    assert rejected["did_update"] is False
    assert rejected["solver_converged"] is False
    assert rejected["update_skip_reason"] == "solver_not_converged"
    assert rejected["grad_x"] is None
    assert memory.rank == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)

    accepted = override.eqprop_update_step(
        x,
        target,
        lr=0.1,
        adjoint_steps=2,
        allow_inexact_update=True,
    )
    assert accepted["did_update"] is True
    assert accepted["solver_converged"] is False
    assert accepted["allow_inexact_update"] is True
    logging.info("✓ Failed certificates are atomic; override is explicit and reported")
    return True


def test_time_varying_rings_contract_uniformly():
    """A sequence of different H_t matrices still contracts at rate 1-alpha."""
    logging.info("Testing uniform contraction for time-varying rings...")

    from mobius_quantum_ring import CayleyUnistochasticParam

    torch.manual_seed(47)
    alpha = 0.2
    q = 1.0 - alpha
    steps = 9
    n = 6
    first = torch.randn(3, n, dtype=torch.float64)
    second = torch.randn(3, n, dtype=torch.float64)
    initial_distance = (first - second).abs().amax()
    for _ in range(steps):
        H = CayleyUnistochasticParam(n, coordinate_mode="minimal").double().unistochastic()
        forcing = torch.randn(3, n, dtype=torch.float64)
        first = torch.tanh(q * (first @ H.transpose(0, 1)) + alpha * forcing)
        second = torch.tanh(q * (second @ H.transpose(0, 1)) + alpha * forcing)
    final_distance = (first - second).abs().amax()
    assert float(final_distance.item()) <= float((q**steps * initial_distance).item()) + 1e-12
    logging.info("✓ Time-varying unistochastic rings obey the uniform contraction bound")
    return True


def test_zero_initialized_readout_is_sidecar_noop():
    """A residual sidecar can start at exact zero while retaining a learning path."""
    logging.info("Testing zero-initialized sidecar readout...")

    from mobius_quantum_ring import MoebiusQuantumRing

    torch.manual_seed(53)
    model = MoebiusQuantumRing(
        input_dim=5,
        hidden_dim=7,
        output_dim=5,
        alpha=0.3,
        relaxation_steps=40,
        lora_rank=3,
        readout_dim=7,
        zero_init_readout=True,
    ).double()
    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 3])
    logits = model(x)
    torch.testing.assert_close(logits, torch.zeros_like(logits), rtol=0.0, atol=0.0)
    injection_before = {
        name: parameter.detach().clone()
        for name, parameter in model.injection.named_parameters()
    }
    unitary_before = {
        name: parameter.detach().clone()
        for name, parameter in model.unitary_param.named_parameters()
    }
    info = model.eqprop_update_step(x, target, lr=0.05, adjoint_steps=60)
    assert info["did_update"] is True
    assert model.readout.readout.weight.abs().amax().item() > 0.0
    for name, parameter in model.injection.named_parameters():
        torch.testing.assert_close(parameter, injection_before[name], rtol=0.0, atol=0.0)
    for name, parameter in model.unitary_param.named_parameters():
        torch.testing.assert_close(parameter, unitary_before[name], rtol=0.0, atol=0.0)
    logging.info("✓ Residual readout is initially no-op and learns without backbone drift")
    return True


def test_sinkhorn_implicit_pullback_matches_autograd():
    """The fair Sinkhorn baseline must not rely on a straight-through gradient."""
    logging.info("Testing converged Sinkhorn implicit pullback...")

    from mqr import SinkhornDoublyStochasticParam

    torch.manual_seed(59)
    n = 6
    parameter = SinkhornDoublyStochasticParam(
        n, iterations=240, temperature=0.8, init_scale=0.5
    ).double()
    H = parameter()
    grad_H = torch.randn_like(H)
    (grad_H * H).sum().backward()
    implicit = parameter.implicit_logit_pullback(grad_H, H=H.detach())
    torch.testing.assert_close(
        implicit, parameter.logits.grad, rtol=1e-10, atol=1e-11
    )
    row_error, column_error = parameter.doubly_stochastic_errors(H=H.detach())
    assert max(row_error.item(), column_error.item()) < 1e-12
    assert parameter.logits.numel() == n * n
    assert parameter.effective_dof == (n - 1) ** 2
    logging.info("✓ Sinkhorn implicit pullback matches converged unrolled autograd")
    return True


def run_all_tests():
    """运行所有测试"""
    logging.info("="*60)
    logging.info("Möbius Quantum Ring Network - Test Suite")
    logging.info("="*60)
    
    tests = [
        test_unitary_matrix_param,
        test_unistochastic_weight,
        test_cayley_phase_recovers_every_unistochastic_representative,
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
        test_cayley_pullback_matches_autograd,
        test_eqprop_implicit_gradients_match_autograd,
        test_complex_implicit_gradient_matches_autograd,
        test_eqprop_unitary_update_is_descent_and_batch_invariant,
        test_implicit_output_gradient_matches_cross_entropy_update,
        test_minimal_cayley_coordinates_and_mixing_rank,
        test_nonlinear_certified_adjoint_matches_direct_and_autograd,
        test_inexact_solver_update_is_rejected_atomically,
        test_time_varying_rings_contract_uniformly,
        test_zero_initialized_readout_is_sidecar_noop,
        test_sinkhorn_implicit_pullback_matches_autograd,
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
