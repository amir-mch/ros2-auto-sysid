"""Publication-ready diagnostic plotting for system identification validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless servers and CI
import matplotlib.pyplot as plt


def plot_sysid_benchmark_results(
    time_steps: np.ndarray,
    y_true: np.ndarray,
    y_noisy: np.ndarray,
    y_uncalibrated: np.ndarray,
    y_calibrated: np.ndarray,
    initial_params: Dict[str, float],
    optimized_params: Dict[str, float],
    ground_truth_params: Dict[str, float],
    loss_history: List[float],
    output_path: str | Path,
) -> Path:
    """Generate multi-panel figure comparing trajectory rollouts and parameter recovery.

    Args:
        time_steps: 1D array of time stamps [s].
        y_true: True trajectory array of shape [T, 6].
        y_noisy: Noisy sensor measurement array of shape [T, 6].
        y_uncalibrated: Rollout trajectory using uncalibrated prior params [T, 6].
        y_calibrated: Rollout trajectory using calibrated params [T, 6].
        initial_params: Initial prior parameter dictionary.
        optimized_params: Calibrated parameter dictionary.
        ground_truth_params: True parameter dictionary used to generate synthetic data.
        loss_history: List of loss values per iteration.
        output_path: Destination path for saved figure.

    Returns:
        Path to the saved image file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 11), dpi=300)
    grid = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Clean engineering color scheme
    c_noisy = "#888888"
    c_true = "#111111"
    c_uncal = "#d95f02"   # Distinct amber/orange for uncalibrated
    c_cal = "#1b9e77"     # Forest green / teal for calibrated solution

    # Panel 1: XY Trajectory in World Frame
    ax_xy = fig.add_subplot(grid[0:2, 0])
    ax_xy.scatter(
        y_noisy[::4, 0], y_noisy[::4, 1],
        color=c_noisy, alpha=0.35, s=8, label="Sensor Measurement (Noisy)",
    )
    ax_xy.plot(y_true[:, 0], y_true[:, 1], color=c_true, linewidth=2.2, label="Ground Truth Dynamics")
    ax_xy.plot(y_uncalibrated[:, 0], y_uncalibrated[:, 1], color=c_uncal, linestyle="--", linewidth=1.8, label="Uncalibrated Sim (Prior)")
    ax_xy.plot(y_calibrated[:, 0], y_calibrated[:, 1], color=c_cal, linestyle="-.", linewidth=2.0, label="Calibrated Sim (Identified)")
    ax_xy.set_title("World Trajectory Rollout ($X$ vs $Y$)", fontsize=13, fontweight="bold")
    ax_xy.set_xlabel("X Position [m]", fontsize=11)
    ax_xy.set_ylabel("Y Position [m]", fontsize=11)
    ax_xy.grid(True, linestyle=":", alpha=0.6)
    ax_xy.legend(loc="best", framealpha=0.9, fontsize=10)
    ax_xy.axis("equal")

    # Panel 2: Longitudinal and Lateral Velocities
    ax_vel = fig.add_subplot(grid[0, 1])
    ax_vel.plot(time_steps, y_true[:, 3], color=c_true, label="$v_x$ (True)", linewidth=1.8)
    ax_vel.plot(time_steps, y_uncalibrated[:, 3], color=c_uncal, linestyle="--", label="$v_x$ (Uncal)", linewidth=1.4)
    ax_vel.plot(time_steps, y_calibrated[:, 3], color=c_cal, linestyle="-.", label="$v_x$ (Cal)", linewidth=1.6)

    ax_vel.plot(time_steps, y_true[:, 4], color="#4575b4", label="$v_y$ (True)", linewidth=1.5)
    ax_vel.plot(time_steps, y_calibrated[:, 4], color="#91bfdb", linestyle=":", label="$v_y$ (Cal)", linewidth=1.5)
    ax_vel.set_title("Body Velocities ($v_x, v_y$)", fontsize=12, fontweight="bold")
    ax_vel.set_xlabel("Time [s]", fontsize=10)
    ax_vel.set_ylabel("Velocity [m/s]", fontsize=10)
    ax_vel.grid(True, linestyle=":", alpha=0.6)
    ax_vel.legend(loc="best", ncol=2, fontsize=8)

    # Panel 3: Yaw Rate (omega_z)
    ax_yaw = fig.add_subplot(grid[1, 1])
    ax_yaw.scatter(time_steps[::4], y_noisy[::4, 5], color=c_noisy, alpha=0.3, s=6, label=r"IMU Gyro $\omega_z$")
    ax_yaw.plot(time_steps, y_true[:, 5], color=c_true, label=r"True $\omega_z$", linewidth=1.8)
    ax_yaw.plot(time_steps, y_uncalibrated[:, 5], color=c_uncal, linestyle="--", label=r"Uncal $\omega_z$", linewidth=1.4)
    ax_yaw.plot(time_steps, y_calibrated[:, 5], color=c_cal, linestyle="-.", label=r"Cal $\omega_z$", linewidth=1.6)
    ax_yaw.set_title(r"Yaw Rate ($\omega_z$)", fontsize=12, fontweight="bold")
    ax_yaw.set_xlabel("Time [s]", fontsize=10)
    ax_yaw.set_ylabel("Yaw Rate [rad/s]", fontsize=10)
    ax_yaw.grid(True, linestyle=":", alpha=0.6)
    ax_yaw.legend(loc="best", fontsize=8)

    # Panel 4: Parameter Estimation Recovery Error Comparison
    ax_params = fig.add_subplot(grid[2, 0])
    shared_keys = [k for k in ground_truth_params.keys() if k in optimized_params and k in initial_params]
    x_indices = np.arange(len(shared_keys))
    bar_width = 0.35

    uncal_err_pct = [
        abs((initial_params[k] - ground_truth_params[k]) / ground_truth_params[k]) * 100.0
        for k in shared_keys
    ]
    cal_err_pct = [
        abs((optimized_params[k] - ground_truth_params[k]) / ground_truth_params[k]) * 100.0
        for k in shared_keys
    ]

    ax_params.bar(x_indices - bar_width / 2, uncal_err_pct, width=bar_width, color=c_uncal, label="Initial Prior % Error", alpha=0.85)
    ax_params.bar(x_indices + bar_width / 2, cal_err_pct, width=bar_width, color=c_cal, label="Calibrated % Error", alpha=0.85)
    ax_params.set_title("Parameter Estimation Error (% of Ground Truth)", fontsize=12, fontweight="bold")
    ax_params.set_xticks(x_indices)
    ax_params.set_xticklabels(shared_keys, rotation=25, ha="right", fontsize=9)
    ax_params.set_ylabel("Absolute Error [%]", fontsize=10)
    ax_params.grid(True, linestyle=":", alpha=0.6)
    ax_params.legend(loc="best", fontsize=9)

    # Panel 5: Optimization Loss Progression
    ax_loss = fig.add_subplot(grid[2, 1])
    ax_loss.plot(range(1, len(loss_history) + 1), loss_history, color="#2b5c8f", linewidth=1.8)
    ax_loss.set_yscale("log")
    ax_loss.set_title("Optimizer Convergence Curve", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Iteration Step", fontsize=10)
    ax_loss.set_ylabel("Composite Loss (Log Scale)", fontsize=10)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return out_file
