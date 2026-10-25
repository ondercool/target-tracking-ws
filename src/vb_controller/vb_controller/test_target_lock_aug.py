#!/usr/bin/env python3
"""Drives the vehicle through a test scenario and logs per-frame tracking data to CSV.

The `model` parameter (NCA or NCA_AUG) selects the tag used in the CSV file name.
No metrics are computed here; raw data is stored for offline analysis.
"""

import csv
import math
import os
import random
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import Twist

from vb_msgs.msg import TargetPixelError

MODELS = ("NCA", "NCA_AUG")

SCENARIOS = {
    1: "linear_forward",
    2: "driveby",
    3: "circular",
    4: "sinusoidal",
}


class TestTargetLockNode(Node):
    def __init__(self):
        super().__init__("test_target_lock_node")

        self.declare_parameter("scenario", 1)
        self.declare_parameter("duration", 10.0)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("model", "NCA")

        model = self.get_parameter("model").get_parameter_value().string_value
        if model not in MODELS:
            raise ValueError(f"Invalid model '{model}', expected one of {MODELS}")
        self.model_tag = model.replace("_", "")  # NCA_AUG -> NCAAUG

        scenario_no = self.get_parameter("scenario").get_parameter_value().integer_value
        if scenario_no not in SCENARIOS:
            raise ValueError(
                f"Invalid scenario '{scenario_no}', expected one of {sorted(SCENARIOS)}"
            )

        self.scenario_no = scenario_no
        self.scenario_name = SCENARIOS[scenario_no]
        self.duration = (
            self.get_parameter("duration").get_parameter_value().double_value
        )
        control_rate = (
            self.get_parameter("control_rate").get_parameter_value().double_value
        )
        self.control_period = 1.0 / control_rate

        self.data_dir = self._resolve_data_dir(self.scenario_name)
        os.makedirs(self.data_dir, exist_ok=True)

        self.rows_lock = threading.Lock()
        self.rows = []
        self.recording = False
        self.start_time = None

        self.cmd_vel_pub = self.create_publisher(Twist, "/ika_nav/cmd_vel", 10)
        self.target_err_sub = self.create_subscription(
            TargetPixelError,
            "/shoot_camera/target_pixel_error",
            self.target_err_callback,
            10,
        )

        self.get_logger().info(
            f"Test target lock node started: scenario={self.scenario_name}, "
            f"model={model}, "
            f"duration={self.duration:.1f}s, saving to {self.data_dir}"
        )

    def _resolve_data_dir(self, scenario_name):
        """Prefer vb_controller/data/<scenario> under the workspace src tree, else fall back to the install share dir."""
        try:
            workspace_root = os.path.dirname(
                os.path.dirname(get_package_prefix("vb_controller"))
            )
            src_pkg_dir = os.path.join(workspace_root, "src", "vb_controller")
            if os.path.isdir(src_pkg_dir):
                return os.path.join(src_pkg_dir, "data", scenario_name)
        except Exception:
            pass
        return os.path.join(
            get_package_share_directory("vb_controller"), "data", scenario_name
        )

    def target_err_callback(self, msg):
        """Store one row per published frame (raw values, no processing)"""
        if not self.recording:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        row = (
            stamp,
            time.time() - self.start_time,
            int(msg.measurement_found),
            msg.u_err,
            msg.v_err,
            msg.depth,
            msg.radius_px,
        )
        with self.rows_lock:
            self.rows.append(row)

    # ------------------------------------------------------------------
    # Scenario command generation
    # ------------------------------------------------------------------
    def cmd_vel_for_scenario(self, elapsed_time):
        linear_x = 0.4 + 0.1 * math.sin(random.uniform(-math.pi, math.pi))
        if self.scenario_no == 3:  # linear_forward
            linear_x = 0.5 + 0.1 * math.sin(random.uniform(-math.pi, math.pi))
        angular_z = 0.0
        if self.scenario_no == 4:  # sinusoidal
            linear_x = 0.4 + 0.1 * math.sin(random.uniform(-math.pi, math.pi))
            angular_z = 1.25 * math.sin(elapsed_time)
        return linear_x, angular_z

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(self):
        self.start_time = time.time()
        self.recording = True

        try:
            while rclpy.ok():
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= self.duration:
                    break

                linear_x, angular_z = self.cmd_vel_for_scenario(elapsed_time)
                twist = Twist()
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                self.cmd_vel_pub.publish(twist)

                time.sleep(self.control_period)
        finally:
            self.recording = False
            self.cmd_vel_pub.publish(Twist())  # stop the vehicle

        self.save_results()

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------
    def save_results(self):
        with self.rows_lock:
            rows = list(self.rows)
        if not rows:
            self.get_logger().warn("No data collected, skipping save")
            return

        trial_no = self._find_free_index()
        trial_path = self._trial_path(trial_no)
        with open(trial_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "elapsed_time",
                    "measurement_found",
                    "u_err",
                    "v_err",
                    "depth",
                    "radius_px",
                ]
            )
            writer.writerows(rows)

        self.get_logger().info(
            f"Saved trial {trial_no} ({len(rows)} rows) to {trial_path}"
        )

    def _trial_path(self, trial_no):
        return os.path.join(
            self.data_dir, f"{self.scenario_name}_{self.model_tag}_{trial_no}.csv"
        )

    def _find_free_index(self):
        i = 0
        while os.path.exists(self._trial_path(i)):
            i += 1
        return i


def main(args=None):
    rclpy.init(args=args)

    try:
        node = TestTargetLockNode()
    except Exception as e:
        print(f"Failed to start test_target_lock_node: {e}")
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
