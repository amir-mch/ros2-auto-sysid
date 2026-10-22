#!/usr/bin/env python3
"""Sim-to-Sim Synthetic System Identification Benchmark.

Generates ground-truth vehicle dynamics with hidden parameters, corrupts the
trajectory with Gaussian sensor noise, runs gradient-based grey-box sysid,
and evaluates parameter recovery and trajectory RMSE reduction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_core.rk4_integrator import RK4Integrator
from sysid_core.optimizer import SystemIdentificationOptimizer
from sysid_models.ackermann_dynamic import DynamicBicycleModel
from sysid_excite.signals import generate_slalom_trajectory
from sysid_exporter.yaml_exporter import export_to_ros2_yaml
from sysid_exporter.urdf_updater import (
    create_default_f1tenth_urdf,
    update_urdf_inertial,
    update_urdf_dynamics,
    create_default_f1tenth_sdf,
    update_sdf_inertial,
    update_sdf_dynamics,
)
from sysid_exporter.plotter import plot_sysid_benchmark_results


def run_benchmark(
    duration: float = 8.0,
    dt: float = 0.02,
    adam_epochs: int = 140,
    lbfgs_epochs: int = 25,
    noise_seed: int = 42,
    output_dir: str | Path = "benchmark_output",
) -> int:
    """Execute complete sim-to-sim validation pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ros2-auto-sysid: Sim-to-Sim Dynamic System Identification Benchmark")
    print("=" * 72)

    device = torch.device("cpu")
    dtype = torch.float64
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)

    # 1. Ground Truth Physical Parameters (True Hidden System)
    gt_params = {
        "mass": 3.40,           # [kg]
        "yaw_inertia": 0.045,   # [kg*m^2]
        "c_f": 75.0,            # [N/rad]
        "c_r": 85.0,            # [N/rad]
        "motor_gain": 20.0,     # [N*s/m]
        "drag_coeff": 0.25,     # [N*s^2/m^2]
        "steering_tau": 0.07,   # [s]
    }

    print("\n[1/5] Building Ground Truth System...")
    true_model = DynamicBicycleModel(
        mass=gt_params["mass"],
        yaw_inertia=gt_params["yaw_inertia"],
        c_f=gt_params["c_f"],
        c_r=gt_params["c_r"],
        motor_gain=gt_params["motor_gain"],
        drag_coeff=gt_params["drag_coeff"],
        steering_tau=gt_params["steering_tau"],
        dtype=dtype,
    ).to(device)

    # 2. Excitation Trajectory Generation
    print(f"[2/5] Generating Persistently Exciting Maneuver ({duration}s @ {1.0/dt:.0f}Hz)...")
    t_np, u_np = generate_slalom_trajectory(
        duration=duration,
        dt=dt,
        base_velocity=2.2,
        velocity_variation=0.6,
        steering_max=0.30,
        steering_freq=0.55,
    )
    u_tensor = torch.tensor(u_np, dtype=dtype, device=device)

    # Initial state: [p_x, p_y, psi, v_x, v_y, omega_z, delta]
    x0 = torch.tensor([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0], dtype=dtype, device=device)

    # 3. Simulate Ground Truth Trajectory via RK4
    integrator = RK4Integrator(dt=dt, hold_mode="linear", n_substeps=2)
    with torch.no_grad():
        x_true_tensor = integrator.rollout(true_model.forward, x0, u_tensor)
        y_true_tensor = true_model.observe(x_true_tensor)

    # Add realistic sensor noise (Odometry + IMU gyro)
    noise_sigmas = torch.tensor(
        [0.025, 0.025, 0.015, 0.035, 0.035, 0.025],
        dtype=dtype,
        device=device,
    )
    sensor_noise = torch.randn_like(y_true_tensor) * noise_sigmas
    y_noisy_tensor = y_true_tensor + sensor_noise

    # 4. Instantiate Uncalibrated Model with Perturbed Initial Guess
    print("\n[3/5] Instantiating Uncalibrated Grey-Box Model with Perturbed Priors:")
    prior_params = {
        "mass": 4.80,          # +41%
        "yaw_inertia": 0.075,  # +67%
        "c_f": 48.0,           # -36%
        "c_r": 58.0,           # -32%
        "motor_gain": 13.0,    # -35%
        "drag_coeff": 0.50,    # +100%
        "steering_tau": 0.14,  # +100%
    }
    for k, v in prior_params.items():
        err = ((v - gt_params[k]) / gt_params[k]) * 100.0
        print(f"  - {k:<13}: Prior = {v:>6.3f} | True = {gt_params[k]:>6.3f} | Error: {err:>+6.1f}%")

    est_model = DynamicBicycleModel(
        mass=prior_params["mass"],
        yaw_inertia=prior_params["yaw_inertia"],
        c_f=prior_params["c_f"],
        c_r=prior_params["c_r"],
        motor_gain=prior_params["motor_gain"],
        drag_coeff=prior_params["drag_coeff"],
        steering_tau=prior_params["steering_tau"],
        dtype=dtype,
    ).to(device)

    # Record uncalibrated rollout
    with torch.no_grad():
        x_uncal_tensor = integrator.rollout(est_model.forward, x0, u_tensor)
        y_uncal_tensor = est_model.observe(x_uncal_tensor)

    # 5. Differentiable System Identification via Adam + L-BFGS
    print(f"\n[4/5] Optimizing Physical Parameters via PyTorch BPTT...")
    optimizer = SystemIdentificationOptimizer(
        model=est_model,
        dt=dt,
        huber_delta=0.8,
        reg_weight=0.005,
        loss_type="huber",
    )

    opt_result = optimizer.fit(
        x0=x0,
        u_seq=u_tensor,
        y_target=y_noisy_tensor,
        adam_epochs=adam_epochs,
        adam_lr=0.06,
        lbfgs_epochs=lbfgs_epochs,
        lbfgs_lr=0.4,
        verbose=True,
    )

    # 6. Evaluation and Verification
    with torch.no_grad():
        x_cal_tensor = integrator.rollout(est_model.forward, x0, u_tensor)
        y_cal_tensor = est_model.observe(x_cal_tensor)

    initial_rmse = opt_result.initial_rmse
    final_rmse = opt_result.final_rmse
    rmse_reduction_pct = (1.0 - final_rmse / max(1e-6, initial_rmse)) * 100.0

    print("\n" + "=" * 72)
    print("CALIBRATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"Trajectory Observable RMSE: {initial_rmse:.4f} -> {final_rmse:.4f} ({rmse_reduction_pct:+.1f}% reduction)\n")
    print(f"{'Parameter':<14} | {'True Value':>10} | {'Prior Guess':>11} | {'Calibrated':>10} | {'Cal Error %':>11}")
    print("-" * 68)

    calibrated_dict = est_model.get_physical_parameters()
    # Map parameter names to ground truth keys
    param_map = {
        "mass": "mass",
        "yaw_inertia": "yaw_inertia",
        "c_f": "c_f",
        "c_r": "c_r",
        "motor_gain": "motor_gain",
        "drag_coeff": "drag_coeff",
        "steering_tau": "steering_tau",
    }

    clean_calibrated: dict[str, float] = {}
    for p_name, p_val in calibrated_dict.items():
        gt_key = param_map.get(p_name, p_name)
        if gt_key in gt_params:
            true_v = gt_params[gt_key]
            prior_v = prior_params[gt_key]
            cal_err = abs((p_val - true_v) / true_v) * 100.0
            clean_calibrated[gt_key] = p_val
            print(f"{gt_key:<14} | {true_v:>10.4f} | {prior_v:>11.4f} | {p_val:>10.4f} | {cal_err:>10.2f}%")

    # 7. Export calibrated files
    print("\n[5/5] Exporting Calibrated Artifacts...")
    yaml_path = export_to_ros2_yaml(
        params=clean_calibrated,
        output_path=out_dir / "calibrated_params.yaml",
        node_name="vehicle_dynamics_node",
        metadata={
            "initial_rmse": initial_rmse,
            "final_rmse": final_rmse,
            "rmse_reduction_pct": rmse_reduction_pct,
            "status": "converged",
        },
    )
    print(f"  [OK] Exported ROS 2 YAML: {yaml_path}")

    # Generate and update URDF (<inertial> and <dynamics>)
    raw_urdf = create_default_f1tenth_urdf(out_dir / "f1tenth_template.urdf")
    cal_urdf = update_urdf_inertial(
        urdf_input_path=raw_urdf,
        urdf_output_path=out_dir / "f1tenth_calibrated.urdf",
        calibrated_params=clean_calibrated,
        target_link_name="base_link",
    )
    # Calibrated steering tau translates to steering damping ~ tau * gain
    steering_damping = clean_calibrated.get("steering_tau", 0.07) * 0.5
    update_urdf_dynamics(
        urdf_input_path=cal_urdf,
        urdf_output_path=cal_urdf,
        joint_name="steering_joint",
        damping=steering_damping,
        friction=0.005,
    )
    print(f"  [OK] Exported Calibrated URDF (inertial + dynamics): {cal_urdf}")

    # Generate and update SDF (<inertial> and <dynamics>)
    raw_sdf = create_default_f1tenth_sdf(out_dir / "f1tenth_template.sdf")
    cal_sdf = update_sdf_inertial(
        sdf_input_path=raw_sdf,
        sdf_output_path=out_dir / "f1tenth_calibrated.sdf",
        calibrated_params=clean_calibrated,
        target_link_name="base_link",
    )
    update_sdf_dynamics(
        sdf_input_path=cal_sdf,
        sdf_output_path=cal_sdf,
        joint_name="steering_joint",
        damping=steering_damping,
        friction=0.005,
    )
    print(f"  [OK] Exported Calibrated SDF (inertial + dynamics): {cal_sdf}")

    # Plot diagnostic figures
    plot_path = plot_sysid_benchmark_results(
        time_steps=t_np,
        y_true=y_true_tensor.cpu().numpy(),
        y_noisy=y_noisy_tensor.cpu().numpy(),
        y_uncalibrated=y_uncal_tensor.cpu().numpy(),
        y_calibrated=y_cal_tensor.cpu().numpy(),
        initial_params=prior_params,
        optimized_params=clean_calibrated,
        ground_truth_params=gt_params,
        loss_history=opt_result.loss_history,
        output_path=out_dir / "sim2sim_benchmark_results.png",
    )
    print(f"  [OK] Publication Plot Saved: {plot_path}")

    print("\n[SUCCESS] Sim-to-Sim Benchmark completed successfully.\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ros2-auto-sysid Sim-to-Sim Benchmark")
    parser.add_argument("--duration", type=float, default=8.0, help="Trajectory duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Sampling time interval in seconds")
    parser.add_argument("--adam-epochs", type=int, default=140, help="Adam optimization epochs")
    parser.add_argument("--lbfgs-epochs", type=int, default=25, help="L-BFGS optimization epochs")
    parser.add_argument("--noise-seed", type=int, default=42, help="Random seed for noise generation")
    parser.add_argument("--output-dir", type=str, default="benchmark_output", help="Directory for output artifacts")
    args = parser.parse_args()

    sys.exit(
        run_benchmark(
            duration=args.duration,
            dt=args.dt,
            adam_epochs=args.adam_epochs,
            lbfgs_epochs=args.lbfgs_epochs,
            noise_seed=args.noise_seed,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
