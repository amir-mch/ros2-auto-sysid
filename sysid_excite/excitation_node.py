"""ROS 2 Node for safe, automated excitation signal injection and trajectory recording."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Literal, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object  # type: ignore

from sysid_excite.safety_monitor import SafetyMonitor, GeofenceLimits
from sysid_excite.signals import (
    generate_slalom_trajectory,
    generate_figure8_trajectory,
    generate_chirp,
    generate_step_sequence,
)


class SysIdExcitationNode(Node):
    """Generates exciting control trajectories and records synchronized dynamics data."""

    def __init__(
        self,
        node_name: str = "sysid_excitation_node",
        dt: float = 0.02,
        duration: float = 12.0,
        output_data_path: str = "sysid_recording.npz",
        command_type: Literal["ackermann", "twist"] = "ackermann",
        signal_type: Literal["slalom", "figure8", "chirp", "step"] = "slalom",
        wheelbase: float = 0.33,
        limits: Optional[GeofenceLimits] = None,
    ) -> None:
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 (rclpy) is not installed or sourced in this environment.")

        super().__init__(node_name)

        self.dt: float = dt
        self.duration: float = duration
        self.output_data_path: Path = Path(output_data_path)
        self.command_type: Literal["ackermann", "twist"] = command_type
        self.signal_type: Literal["slalom", "figure8", "chirp", "step"] = signal_type
        self.wheelbase: float = wheelbase
        self.safety_monitor: SafetyMonitor = SafetyMonitor(limits=limits)

        # Generate pre-planned persistently exciting trajectory based on requested signal
        if self.signal_type == "figure8":
            self.time_array, self.control_sequence = generate_figure8_trajectory(
                duration=self.duration,
                dt=self.dt,
            )
        elif self.signal_type == "chirp":
            n_samples = int(self.duration / self.dt)
            t = np.linspace(0.0, self.duration, n_samples, endpoint=False)
            v_cmd = np.full_like(t, 2.0)
            steer = generate_chirp(t, f0=0.2, f1=2.5, amplitude=0.30)
            self.time_array = t
            self.control_sequence = np.column_stack([v_cmd, steer])
        elif self.signal_type == "step":
            n_samples = int(self.duration / self.dt)
            t = np.linspace(0.0, self.duration, n_samples, endpoint=False)
            v_cmd = np.full_like(t, 2.0)
            steer = generate_step_sequence(
                t,
                step_times=[1.0, 3.0, 5.0, 7.0, 9.0],
                step_amplitudes=[0.15, -0.15, 0.25, -0.25, 0.0],
            )
            self.time_array = t
            self.control_sequence = np.column_stack([v_cmd, steer])
        else:  # Default "slalom"
            self.time_array, self.control_sequence = generate_slalom_trajectory(
                duration=self.duration,
                dt=self.dt,
            )

        self.step_idx: int = 0
        self.is_running: bool = False

        # Current sensor telemetry state: [x, y, psi, v_x, v_y, omega_z]
        self.current_state = np.zeros(6, dtype=np.float64)

        # Storage for synchronized dataset
        self.recorded_timestamps: List[float] = []
        self.recorded_controls: List[np.ndarray] = []
        self.recorded_states: List[np.ndarray] = []

        # ROS 2 Publishers and Subscribers
        if self.command_type == "ackermann":
            self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
            self.twist_pub = None
        else:
            self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.drive_pub = None

        self.estop_pub = self.create_publisher(Bool, "/sysid/estop", 10)

        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/imu/data", self._imu_callback, 10)

        self.timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f"SysIdExcitationNode initialized. Command: {self.command_type}, "
            f"Signal: {self.signal_type}, Duration: {self.duration}s"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y

        # Quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_state[2] = np.arctan2(siny_cosp, cosy_cosp)

        self.current_state[3] = msg.twist.twist.linear.x
        self.current_state[4] = msg.twist.twist.linear.y

    def _imu_callback(self, msg: Imu) -> None:
        self.current_state[5] = msg.angular_velocity.z

    def _control_loop(self) -> None:
        if self.step_idx >= len(self.control_sequence):
            self._finish_experiment()
            return

        # Check safety envelope
        px, py, psi, vx, vy, omega_z = self.current_state
        is_safe, violation = self.safety_monitor.check_state(px, py, vx, vy, omega_z)

        if not is_safe:
            self.get_logger().error(f"SAFETY ABORT TRIGGERED: {violation}")
            self._trigger_emergency_stop()
            return

        # Retrieve control command
        v_cmd, delta_cmd = self.control_sequence[self.step_idx]
        t_curr = self.time_array[self.step_idx]

        # Publish command to vehicle
        if self.command_type == "ackermann" and self.drive_pub is not None:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = "base_link"
            drive_msg.drive.speed = float(v_cmd)
            drive_msg.drive.steering_angle = float(delta_cmd)
            self.drive_pub.publish(drive_msg)
        elif self.twist_pub is not None:
            twist_msg = Twist()
            twist_msg.linear.x = float(v_cmd)
            # Map Ackermann steering angle to equivalent kinematic yaw rate: omega = v / L * tan(delta)
            twist_msg.angular.z = float(v_cmd / max(0.05, self.wheelbase) * np.tan(delta_cmd))
            self.twist_pub.publish(twist_msg)

        # Log synchronized record
        self.recorded_timestamps.append(t_curr)
        self.recorded_controls.append(np.array([v_cmd, delta_cmd], dtype=np.float64))
        self.recorded_states.append(self.current_state.copy())

        self.step_idx += 1

    def _trigger_emergency_stop(self) -> None:
        self._send_stop_command()

        estop_flag = Bool()
        estop_flag.data = True
        self.estop_pub.publish(estop_flag)

        self.timer.cancel()
        self._save_recorded_dataset()

    def _finish_experiment(self) -> None:
        self.get_logger().info("Excitation run completed successfully. Stopping vehicle.")
        self._send_stop_command()

        self.timer.cancel()
        self._save_recorded_dataset()

    def _send_stop_command(self) -> None:
        if self.drive_pub is not None:
            stop_msg = AckermannDriveStamped()
            stop_msg.drive.speed = 0.0
            stop_msg.drive.steering_angle = 0.0
            self.drive_pub.publish(stop_msg)
        if self.twist_pub is not None:
            stop_twist = Twist()
            self.twist_pub.publish(stop_twist)

    def _save_recorded_dataset(self) -> None:
        if not self.recorded_timestamps:
            return

        t_arr = np.array(self.recorded_timestamps)
        u_arr = np.array(self.recorded_controls)
        x_arr = np.array(self.recorded_states)

        self.output_data_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_data_path.suffix.lower() == ".csv":
            with open(self.output_data_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time", "v_cmd", "delta_cmd",
                    "p_x", "p_y", "psi", "v_x", "v_y", "omega_z",
                ])
                for t_val, u_val, x_val in zip(t_arr, u_arr, x_arr):
                    writer.writerow([t_val, u_val[0], u_val[1], *x_val.tolist()])
        else:
            np.savez_compressed(
                self.output_data_path,
                time=t_arr,
                controls=u_arr,
                states=x_arr,
                control_names=["v_cmd", "delta_cmd"],
                state_names=["p_x", "p_y", "psi", "v_x", "v_y", "omega_z"],
            )

        self.get_logger().info(f"Saved {len(t_arr)} synchronized data points to {self.output_data_path}")


def main(args: Optional[List[str]] = None) -> None:
    if not HAS_ROS2:
        raise RuntimeError("ROS 2 (rclpy) is not available.")
    rclpy.init(args=args)
    node = SysIdExcitationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
