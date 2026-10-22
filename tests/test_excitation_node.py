"""Unit tests for ROS 2 SysIdExcitationNode."""

import csv
from pathlib import Path
import pytest
import numpy as np

try:
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Twist
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from sysid_excite.excitation_node import SysIdExcitationNode
from sysid_excite.safety_monitor import GeofenceLimits


@pytest.fixture(autouse=True)
def rclpy_init_shutdown():
    if not HAS_ROS2:
        pytest.skip("ROS 2 (rclpy) is not available.")
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_excitation_node_ackermann_mode(tmp_path: Path):
    """Verify excitation node initializes with AckermannDrive publisher and records NPZ."""
    npz_path = tmp_path / "test_rec.npz"
    node = SysIdExcitationNode(
        node_name="test_ackermann_node",
        dt=0.02,
        duration=1.0,
        output_data_path=str(npz_path),
        command_type="ackermann",
        signal_type="slalom",
    )

    assert node.drive_pub is not None
    assert node.twist_pub is None
    assert len(node.control_sequence) == 50

    # Step through 5 control iterations
    for _ in range(5):
        node._control_loop()

    assert len(node.recorded_timestamps) == 5
    assert len(node.recorded_controls) == 5
    assert len(node.recorded_states) == 5

    node.destroy_node()


def test_excitation_node_twist_mode(tmp_path: Path):
    """Verify excitation node initializes with Twist cmd_vel publisher and figure8 signals."""
    csv_path = tmp_path / "test_rec.csv"
    node = SysIdExcitationNode(
        node_name="test_twist_node",
        dt=0.02,
        duration=1.0,
        output_data_path=str(csv_path),
        command_type="twist",
        signal_type="figure8",
    )

    assert node.twist_pub is not None
    assert node.drive_pub is None

    # Step through entire sequence to trigger finish and CSV write
    for _ in range(len(node.control_sequence)):
        node._control_loop()

    # Final control loop invocation should trigger _finish_experiment
    node._control_loop()

    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["time", "v_cmd", "delta_cmd", "p_x", "p_y", "psi", "v_x", "v_y", "omega_z"]
        rows = list(reader)
        assert len(rows) == 50

    node.destroy_node()


def test_excitation_node_safety_estop(tmp_path: Path):
    """Verify exceeding safety limit immediately triggers e-stop and aborts excitation."""
    npz_path = tmp_path / "estop_rec.npz"
    limits = GeofenceLimits(x_min=-2.0, x_max=2.0)
    node = SysIdExcitationNode(
        node_name="test_estop_node",
        dt=0.02,
        duration=2.0,
        output_data_path=str(npz_path),
        limits=limits,
    )

    # Run one nominal step
    node._control_loop()
    assert len(node.recorded_timestamps) == 1

    # Simulate vehicle breaching geofence
    odom = Odometry()
    odom.pose.pose.position.x = 5.0  # Geofence breach (> 2.0m)
    node._odom_callback(odom)

    # Control loop should detect violation, trigger e-stop, and save partial recording
    node._control_loop()

    assert node.safety_monitor.emergency_stop_triggered
    assert npz_path.exists()

    node.destroy_node()
