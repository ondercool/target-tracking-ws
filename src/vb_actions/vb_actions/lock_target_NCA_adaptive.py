#!/usr/bin/env python3
import math
import time
import threading

import numpy as np
import cv2
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from scipy.cluster.hierarchy import linkage, fcluster

from vb_actions.action import LockTargetDynamic
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from tf_transformations import quaternion_matrix
from tf2_ros import TransformListener, Buffer

from vb_msgs.msg import TargetPixelError
from std_msgs.msg import Header

# Camera frame (x forward, y left, z up) -> optical frame (x right, y down, z forward)
CAM_TO_OPTICAL = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def transform_to_matrix(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    mat = quaternion_matrix([q.x, q.y, q.z, q.w])  # 4x4, rotation only
    mat[0, 3] = t.x
    mat[1, 3] = t.y
    mat[2, 3] = t.z
    return mat


class LockTargetDynamicConfig:
    """Configuration for the dynamic target tracking action server"""

    def __init__(self):
        # Camera configuration
        self.cam_topic = "/shoot_camera/image_raw/compressed"
        self.cam_frame_id = "shoot_camera"
        self.cam_width = 1920
        self.cam_height = 1080
        self.optical_center_u = 960.5
        self.optical_center_v = 540.5
        self.focal_length_u = 1481.37
        self.focal_length_v = 1481.37

        # Goal acceptance
        self.accept_frames = 15

        # Goal acquisition (initial EKF seed)
        self.acquisition_iterations = 20
        self.grid_size = 20

        # Clustering
        self.cluster_max_dist_px = 50.0

        # Circle detection
        self.hough_param2 = 20.0  # accumulator threshold (higher = more selective)
        self.hough_min_radius = 7
        self.hough_max_radius = 50  # used only when no state prediction is available
        self.min_fill_ratio = 0.1  # min fraction of mask pixels inside a circle
        self.radius_margin_px = 3.0  # added to the predicted max radius
        self.roi_scale = 8.0  # multiplier on the ROI ellipse semi-axes
        self.roi_min_axis_px = 5.0  # floor on the ROI ellipse semi-axes

        # EKF noise
        self.measurement_noise = 50.0  # measurement noise (cm^2), diag on X,Y,Z

        # Adaptive measurement noise: C = diag(sigma_lat^2, sigma_lat^2, sigma_z^2)
        # in the camera frame (x = depth), rotated into base_link per measurement
        self.sigma_lateral_cm = 4.0  # lateral std (cm)
        self.sigma_depth_cap_cm = 33.0  # depth std cap (cm)
        self.sigma_radius_px = 1.0  # detected-radius error (px) -> depth noise
        self.initial_pos_var = 300.0
        self.initial_vel_var = 300.0
        self.var_acc = 2000.0

        # Controller
        self.control_frequency = 30.0  # Hz
        self.control_period = 1.0 / self.control_frequency
        self.camera_velocity_gain = 12.0  # rad/s per rad of pointing error (Kp)
        self.camera_velocity_kd = 0.6  # rad/s per rad/s of error rate (Kd)
        self.max_camera_velocity = 2.5  # rad/s

        # Success criterion (angular error tolerance, since goal has no explicit tolerance field)
        self.angular_tolerance_deg = 1.0

        # Real world target radius (cm), same as lock_target.py
        self.real_radius = 9.0


class LockTargetDynamicServer(Node):
    """Action server for tracking a moving target using an EKF and pan/tilt control"""

    def __init__(self):
        super().__init__("lock_target_adaptive_server")

        self.config = LockTargetDynamicConfig()
        self.callback_group = ReentrantCallbackGroup()
        self.image_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        self.bridge = CvBridge()

        # TF2 setup
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        # Shared state protected by lock
        self.data_lock = threading.Lock()

        # Camera intrinsics
        self.cam_matrix = None
        self.cam_dist_coeffs = None

        # Joint states
        self.current_tilt = 0.0
        self.current_pan = 0.0
        self.tilt_velocity_cmd = 0.0
        self.pan_velocity_cmd = 0.0
        self.prev_pan_error = None
        self.prev_tilt_error = None
        self.prev_control_time = None

        # Laser offset (same as lock_target.py)
        self.t_vec_l_c = np.array([[0.0], [0.0], [0.025]], dtype=np.float64)

        # Latest raw detections (list of (center, radius)) from the camera callback
        self.latest_detections = []
        self.latest_image = None
        self.new_frame_event = threading.Event()

        # EKF state: [X, Y, Z, vx, vy, vz] in camera optical frame (cm)
        self.ekf_x = None
        self.ekf_P = None
        self.ekf_last_time = None
        self.tracking_active = False

        # Feedback data
        self.current_tilt_error = 0.0
        self.current_pan_error = 0.0
        self.current_target_pixel_error = None
        self.current_target_pixels = None
        self.current_target_radius = None
        self.current_depth_estimate = None
        self.measurement_found = False
        self.disk_cache = {}

        self.debug_img = None

        # Subscriptions
        self.shoot_cam_sub = self.create_subscription(
            CompressedImage,
            self.config.cam_topic,
            self.shoot_cam_callback,
            10,
            callback_group=self.image_cb_group,
        )
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, "/shoot_camera/camera_info", self.camera_info_callback, 10
        )

        # Publishers
        self.shoot_debug_pub = self.create_publisher(
            Image, "/shoot_camera/debug_image", 10
        )
        self.shoot_velocity_cmd_pub = self.create_publisher(
            Float64MultiArray, "/shoot_camera_velocity_controller/commands", 10
        )

        self.target_pixel_err_pub = self.create_publisher(
            TargetPixelError, "/shoot_camera/target_pixel_error", 10
        )
        # Controller timer (always running, only acts while tracking_active)
        self.control_timer = self.create_timer(
            self.config.control_period,
            self.controller_timer_callback,
            callback_group=self.control_cb_group,
        )

        # Action server
        self._action_server = ActionServer(
            self,
            LockTargetDynamic,
            "/vb_actions/lock_target_dynamic",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Lock target dynamic action server initialized")

    # ------------------------------------------------------------------
    # Subscriptions callbacks
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg):
        with self.data_lock:
            self.current_tilt = msg.position[5]
            self.current_pan = msg.position[4]

    def camera_info_callback(self, msg):
        with self.data_lock:
            self.cam_matrix = np.array(msg.k).reshape(3, 3)
            self.cam_dist_coeffs = np.array(msg.d).reshape(-1, 1)

    def shoot_cam_callback(self, msg):
        """Camera callback: detect circles, run EKF thread logic when tracking is active"""
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        pred = self.predict_and_project() if self.tracking_active else None
        centers, radii, debug_img = self.detect_circles(img, pred)

        with self.data_lock:
            self.latest_detections = list(zip(centers, radii))
            self.latest_image = img

        self.new_frame_event.set()

        if self.tracking_active:
            self.ekf_update_step(centers, radii, debug_img, pred)
            self.publish_tracking_debug()
        else:
            self.debug_img = debug_img

        if self.debug_img is not None:
            with self.data_lock:
                if self.current_target_pixels is not None:
                    # self.get_logger().info(
                    #     f"Current target pixels: {self.current_target_pixels}"
                    # )
                    debug_img = cv2.circle(
                        debug_img,
                        (
                            int(self.current_target_pixels[0]),
                            int(self.current_target_pixels[1]),
                        ),
                        round(self.current_target_radius),
                        (0, 0, 255),
                        6,
                    )
                self.debug_img = debug_img
            debug_img_msg = self.bridge.cv2_to_imgmsg(self.debug_img)
            debug_img_msg.header.stamp = self.get_clock().now().to_msg()
            self.shoot_debug_pub.publish(debug_img_msg)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def predict_and_project(self):
        """
        EKF predict, then project the predicted position and its covariance
        onto the image. Returns None if no state / TF / valid projection.
        """
        cfg = self.config
        now = time.time()
        with self.data_lock:
            if self.ekf_x is None:
                return None
            dt = now - self.ekf_last_time if self.ekf_last_time else cfg.control_period
            self.ekf_predict(max(dt, 1e-3))
            self.ekf_last_time = now
            p_base = self.ekf_x[0:3, 0].copy()
            P_pos = self.ekf_P[0:3, 0:3].copy()
            K = self.cam_matrix

        if K is not None:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy = cfg.focal_length_u, cfg.focal_length_v
            cx, cy = cfg.optical_center_u, cfg.optical_center_v

        try:
            T_base_cam = transform_to_matrix(
                self.buffer.lookup_transform(
                    "base_link", cfg.cam_frame_id, rclpy.time.Time()
                )
            )
        except Exception:
            return None

        # p_opt = A (p_base - t), with A = CAM_TO_OPTICAL @ R_base_cam^T
        A = CAM_TO_OPTICAL @ T_base_cam[0:3, 0:3].T
        X, Y, Z = A @ (p_base - T_base_cam[0:3, 3])
        if Z < 1.0:
            return None

        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        if not (np.isfinite(u) and np.isfinite(v)):
            return None

        # First-order covariance propagation: S_uv = J A P A^T J^T
        J = np.array([[fx / Z, 0.0, -fx * X / Z**2], [0.0, fy / Z, -fy * Y / Z**2]])
        JA = J @ A
        S_uv = JA @ P_pos @ JA.T
        eigvals, V = np.linalg.eigh(0.5 * (S_uv + S_uv.T))  # ascending, V columns
        axes = np.maximum(
            cfg.roi_scale * np.sqrt(np.maximum(eigvals, 0.0)), cfg.roi_min_axis_px
        )  # axes[0] = semi-minor, axes[1] = semi-major

        # Axis-aligned half extents of the rotated ellipse
        half_u = float(np.sqrt(np.sum((axes * V[0, :]) ** 2)))
        half_v = float(np.sqrt(np.sum((axes * V[1, :]) ** 2)))

        # Projected target radius: r_px = f * R / Z (R = real radius, same units as Z)
        max_radius = (
            max(fx * cfg.real_radius / Z, fy * cfg.real_radius / Z)
            + cfg.radius_margin_px
        )
        return {
            "u": float(u),
            "v": float(v),
            "axes": axes,
            "V": V,
            "half_u": half_u,
            "half_v": half_v,
            "max_radius": float(max_radius),
            "T_base_cam": T_base_cam,
        }

    def disk(self, r):
        d = self.disk_cache.get(r)
        if d is None:
            yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
            d = ((xx * xx + yy * yy) <= r * r).astype(np.uint8) * 255
            self.disk_cache[r] = d
        return d

    def fill_ratio(self, mask, cx, cy, r):
        """Fraction of mask pixels set inside the (image-clipped) circle"""
        r = max(int(round(r)), 1)
        cx, cy = int(round(cx)), int(round(cy))
        hm, wm = mask.shape
        x0, x1 = max(cx - r, 0), min(cx + r + 1, wm)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, hm)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        d = self.disk(r)[y0 - (cy - r) : y1 - (cy - r), x0 - (cx - r) : x1 - (cx - r)]
        area = cv2.countNonZero(d)
        if area == 0:
            return 0.0
        return cv2.countNonZero(cv2.bitwise_and(mask[y0:y1, x0:x1], d)) / area

    def detect_circles(self, img, pred=None):
        """
        Detect circles. Returns (centers, radii, debug_img).
        With a state prediction the search is limited to the ROI ellipse and
        to the predicted maximum radius. Every circle must also pass the
        fill-ratio test.
        """
        cfg = self.config
        h, w = img.shape[:2]
        min_r = cfg.hough_min_radius

        if pred is None:
            x0, y0, x1, y1 = 0, 0, w, h
            max_r = cfg.hough_max_radius
        else:
            max_r = int(math.ceil(pred["max_radius"]))
            margin = max_r + 1
            x0 = max(int(math.floor(pred["u"] - pred["half_u"])) - margin, 0)
            x1 = min(int(math.ceil(pred["u"] + pred["half_u"])) + margin + 1, w)
            y0 = max(int(math.floor(pred["v"] - pred["half_v"])) - margin, 0)
            y1 = min(int(math.ceil(pred["v"] + pred["half_v"])) + margin + 1, h)
        max_r = max(max_r, min_r + 1)

        centers = []
        radii = []
        if x1 - x0 > 2 * min_r and y1 - y0 > 2 * min_r:
            sub = img[y0:y1, x0:x1]
            hsv = cv2.cvtColor(cv2.GaussianBlur(sub, (5, 5), 1.5), cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 35), (255, 100, 170))

            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=5,
                param1=100,
                param2=cfg.hough_param2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                c = np.around(circles[0]).astype(np.float64)  # (N, 3) crop coords
                if pred is not None:
                    d = (
                        c[:, 0:2]
                        + np.array([x0, y0])
                        - np.array([pred["u"], pred["v"]])
                    )
                    q = d @ pred["V"]  # coordinates along ellipse axes
                    inside = (q[:, 0] / pred["axes"][0]) ** 2 + (
                        q[:, 1] / pred["axes"][1]
                    ) ** 2 <= 1.0
                    c = c[inside]
                for cx, cy, r in c:
                    if self.fill_ratio(mask, cx, cy, r) > cfg.min_fill_ratio:
                        centers.append((float(cx + x0), float(cy + y0)))
                        radii.append(float(r))

        if pred is not None:
            angle = math.degrees(math.atan2(pred["V"][1, 1], pred["V"][0, 1]))
            cv2.ellipse(
                img,
                (int(round(pred["u"])), int(round(pred["v"]))),
                (int(round(pred["axes"][1])), int(round(pred["axes"][0]))),
                angle,
                0,
                360,
                (0, 255, 255),
                2,
            )
        for center, radius in zip(centers, radii):
            ic = (int(center[0]), int(center[1]))
            cv2.circle(img, ic, 1, (0, 0, 255), 1)
            cv2.circle(img, ic, int(radius), (0, 255, 0), 1)

        self.draw_feedback_text(img)
        return centers, radii, img

    def draw_feedback_text(self, img):
        tilt_text = f"Tilt Error: {self.current_tilt_error:.3f} deg"
        pan_text = f"Pan Error: {self.current_pan_error:.3f} deg"

        tilt_color = (0, 255, 0) if abs(self.current_tilt_error) < 0.3 else (0, 0, 255)
        pan_color = (0, 255, 0) if abs(self.current_pan_error) < 0.3 else (0, 0, 255)

        cv2.putText(
            img, tilt_text, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tilt_color, 4
        )
        cv2.putText(
            img, pan_text, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pan_color, 4
        )

    # ------------------------------------------------------------------
    # Clustering (complete linkage)
    # ------------------------------------------------------------------
    def cluster_indices(self, centers):
        """Complete-linkage clustering. Returns list of index-lists (clusters)."""
        n = len(centers)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        pts = np.array(centers, dtype=np.float64)
        Z = linkage(pts, method="complete")
        labels = fcluster(Z, t=self.config.cluster_max_dist_px, criterion="distance")

        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)
        return list(clusters.values())

    def cluster_stats(self, centers, radii, indices):
        """Compute (avg_center, max_radius) for a set of indices"""
        c = np.array([centers[i] for i in indices])
        r = np.array([radii[i] for i in indices])
        avg_center = (float(np.mean(c[:, 0])), float(np.mean(c[:, 1])))
        max_radius = float(np.max(r))
        return avg_center, max_radius

    # ------------------------------------------------------------------
    # World coordinates
    # ------------------------------------------------------------------
    def world_coordinates(self, center, radius):
        # turn image data into camera->target transformation
        with self.data_lock:
            cam_matrix = self.cam_matrix
            cam_dist_coeffs = self.cam_dist_coeffs

        if cam_matrix is None or cam_dist_coeffs is None:
            return None, None

        cx, cy = center
        radius_px = radius
        real_radius = self.config.real_radius

        image_points = np.array(
            [
                [cx, cy],
                [cx + radius_px, cy],
                [cx - radius_px, cy],
                [cx, cy + radius_px],
                [cx, cy - radius_px],
            ],
            dtype=np.float32,
        )

        object_points = np.array(
            [
                [0, 0, 0],
                [real_radius, 0, 0],
                [-real_radius, 0, 0],
                [0, real_radius, 0],
                [0, -real_radius, 0],
            ],
            dtype=np.float32,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, cam_matrix, cam_dist_coeffs
        )
        if success:
            tvec = np.array([[tvec[2, 0]], [-tvec[0, 0]], [-tvec[1, 0]]])
            return rvec, tvec
        return None, None

    # ------------------------------------------------------------------
    # Goal Acceptance
    # ------------------------------------------------------------------
    def goal_callback(self, goal_request):
        """Accept goal if target detected in at least one of N frames"""
        tracking_duration = goal_request.tracking_duration
        if tracking_duration <= 0:
            self.get_logger().warn(
                f"Goal rejected: Invalid tracking duration {tracking_duration}"
            )
            return GoalResponse.REJECT

        detected = False
        frames_checked = 0
        timeout_time = time.time() + 5.0

        while frames_checked < self.config.accept_frames and time.time() < timeout_time:
            self.new_frame_event.clear()
            if not self.new_frame_event.wait(timeout=1.0):
                continue
            with self.data_lock:
                dets = list(self.latest_detections)
            frames_checked += 1
            if len(dets) > 0:
                detected = True
                break

        if not detected:
            self.get_logger().warn(
                f"Goal rejected: No target detected in {frames_checked} frames"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Goal accepted: Target detected, tracking duration: {tracking_duration}s"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    # Goal Acquisition (initial EKF seed)
    # ------------------------------------------------------------------
    def acquire_initial_target(self):
        """
        For N iterations, detect + cluster circles, bin the best cluster's
        center-of-mass into a 20x20 grid, then pick the grid cell with the
        highest count and average its registered (center, radius) entries.
        """
        grid_size = self.config.grid_size
        grid_counts = np.zeros((grid_size, grid_size), dtype=int)
        grid_entries = {}

        for _ in range(self.config.acquisition_iterations):
            self.new_frame_event.clear()
            if not self.new_frame_event.wait(timeout=1.0):
                continue

            with self.data_lock:
                dets = list(self.latest_detections)

            if len(dets) == 0:
                continue

            centers = [d[0] for d in dets]
            radii = [d[1] for d in dets]

            clusters = self.cluster_indices(centers)
            if not clusters:
                continue

            best_cluster = max(clusters, key=len)
            mean_center, max_radius = self.cluster_stats(centers, radii, best_cluster)

            cell_x = int(mean_center[0] / self.config.cam_width * grid_size)
            cell_y = int(mean_center[1] / self.config.cam_height * grid_size)
            cell_x = min(max(cell_x, 0), grid_size - 1)
            cell_y = min(max(cell_y, 0), grid_size - 1)

            grid_counts[cell_y, cell_x] += 1
            grid_entries.setdefault((cell_y, cell_x), []).append(
                (mean_center, max_radius)
            )

        if grid_counts.max() == 0:
            return None, None

        best_cell = np.unravel_index(np.argmax(grid_counts), grid_counts.shape)
        entries = grid_entries[best_cell]

        avg_center = (
            float(np.mean([e[0][0] for e in entries])),
            float(np.mean([e[0][1] for e in entries])),
        )
        avg_radius = float(np.mean([e[1] for e in entries]))

        # Mark the initial goal
        with self.data_lock:
            if self.debug_img is not None:
                cv2.circle(
                    self.debug_img,
                    (int(avg_center[0]), int(avg_center[1])),
                    int(avg_radius),
                    (0, 0, 255),
                    2,
                )
                debug_img_msg = self.bridge.cv2_to_imgmsg(self.debug_img)
                debug_img_msg.header.stamp = self.get_clock().now().to_msg()
                self.shoot_debug_pub.publish(debug_img_msg)

                timestamp = int(time.time() * 1000)
                filename = f"/home/rosgeek/Documents/vb_tracker_debug_{timestamp}.png"
                cv2.imwrite(filename, self.debug_img)

        return avg_center, avg_radius

    # ------------------------------------------------------------------
    # EKF
    # ------------------------------------------------------------------
    def ekf_init(self, initial_tvec):
        assert initial_tvec.shape == (3, 1), "initial_tvec must be a 3x1 column vector"

        # Lookup transform dynamically
        try:
            T_base_cam = transform_to_matrix(
                self.buffer.lookup_transform(
                    "base_link", self.config.cam_frame_id, rclpy.time.Time()
                )
            )
        except Exception as e:
            self.get_logger().error(f"TF Error in init: {e}")
            return

        # Convert to homogeneous (4x1), multiply, extract (3x1)
        tvec_h = np.vstack((initial_tvec, [1.0]))
        base_tvec = (T_base_cam @ tvec_h)[0:3, :]

        self.ekf_x = np.vstack((base_tvec, np.zeros((3, 1))))

        P = np.eye(6)
        P[0:3, 0:3] *= self.config.initial_pos_var
        P[3:6, 3:6] *= self.config.initial_vel_var

        self.ekf_P = P
        self.ekf_last_time = time.time()

    def ekf_predict(self, dt):
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        Q = self.generate_Q_matrix(dt, self.config.var_acc)

        self.ekf_x = F @ self.ekf_x
        self.ekf_P = F @ self.ekf_P @ F.T + Q

    def measurement_noise_cov(self, depth, R_base_cam):
        """
        R_base = R_base_cam @ C @ R_base_cam^T, C = diag(s_l^2, s_l^2, s_z^2)
        in the camera frame (x = depth), s_z = min(cap, Z^2 / (f R) * sigma_r).
        """
        cfg = self.config
        f = 0.5 * (cfg.focal_length_u + cfg.focal_length_v)
        sigma_z = min(
            cfg.sigma_depth_cap_cm,
            depth**2 / (f * cfg.real_radius) * cfg.sigma_radius_px,
        )
        C = np.diag([sigma_z**2, cfg.sigma_lateral_cm**2, cfg.sigma_lateral_cm**2])
        return R_base_cam @ C @ R_base_cam.T

    def ekf_update(self, z, R=None):
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        if R is None:
            R = np.eye(3) * self.config.measurement_noise

        y_res = z - H @ self.ekf_x
        S = H @ self.ekf_P @ H.T + R
        K = self.ekf_P @ H.T @ np.linalg.inv(S)

        self.ekf_x = self.ekf_x + K @ y_res
        self.ekf_P = (np.eye(6) - K @ H) @ self.ekf_P

        return y_res, S

    def mahalanobis_distance(self, z):
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.eye(3) * self.config.measurement_noise

        y_res = z - H @ self.ekf_x
        S = H @ self.ekf_P @ H.T + R
        d2 = float(y_res.T @ np.linalg.inv(S) @ y_res)
        return d2

    def depth_est2target_cam_pixels(self, depth_estimate=None):
        # Using the depth estimate in PnP Frame to return camera pixels when laser points to the target
        # Will be used for pixel error calculations
        if depth_estimate is None:
            return None
        # Target coordinates in PnP frame when laser is assumed pointing on the target
        # y_offset = float(-self.t_vec_l_c[2, 0])
        z_depth = float(depth_estimate)

        u = (0.0 * self.config.focal_length_u) / z_depth + self.config.optical_center_u
        v = (
            self.config.real_radius * self.config.focal_length_v
        ) / z_depth + self.config.optical_center_v
        return np.array([u, v], dtype=float)

    def target2pixel_error(self, target_pixels=None, depth_estimate=None):
        if target_pixels is None:
            return None
        estimated_pixels = self.depth_est2target_cam_pixels(
            depth_estimate=depth_estimate
        )
        if estimated_pixels is None:
            return None
        error = estimated_pixels - target_pixels
        return np.array(error, dtype=float)

    def ekf_update_step(self, centers, radii, debug_img, pred):
        """EKF update run on every camera frame while tracking is active.
        The predict step already ran in predict_and_project()."""
        if self.ekf_x is None:
            with self.data_lock:
                self.measurement_found = False
            self.debug_img = debug_img
            return

        if pred is None or len(centers) == 0:
            with self.data_lock:
                self.measurement_found = False
                self.debug_img = debug_img
            return

        # Cluster the filtered detections, pick the cluster closest to the predicted pixel
        pred_px = np.array([pred["u"], pred["v"]])
        best_center = None
        best_radius = None
        best_dist = float("inf")
        for cluster in self.cluster_indices(centers):
            avg_center, max_radius = self.cluster_stats(centers, radii, cluster)
            dist = float(np.linalg.norm(np.array(avg_center) - pred_px))
            if dist < best_dist:
                best_center, best_radius, best_dist = avg_center, max_radius, dist

        _, tvec = self.world_coordinates(best_center, best_radius)
        if tvec is None:
            with self.data_lock:
                self.measurement_found = False
                self.debug_img = debug_img
            return

        z = (pred["T_base_cam"] @ np.vstack((tvec, [1.0])))[0:3, :]
        self.current_target_pixels = best_center
        self.current_target_radius = best_radius
        # Camera frame x is the same as PnP frame z axis
        self.current_depth_estimate = tvec[0, 0]
        self.current_target_pixel_error = self.target2pixel_error(
            target_pixels=best_center, depth_estimate=self.current_depth_estimate
        )
        with self.data_lock:
            self.ekf_update(
                z,
                self.measurement_noise_cov(
                    self.current_depth_estimate, pred["T_base_cam"][0:3, 0:3]
                ),
            )
            self.measurement_found = True

        cv2.circle(
            debug_img, (int(best_center[0]), int(best_center[1])), 4, (255, 0, 0), -1
        )
        with self.data_lock:
            self.debug_img = debug_img

    def publish_tracking_debug(self):
        """Publish per-frame tracking debug data (called once per camera frame)"""
        with self.data_lock:
            found = self.measurement_found
            err = self.current_target_pixel_error
            depth = self.current_depth_estimate
        nan = float("nan")
        msg = TargetPixelError()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.measurement_found = bool(found)
        if found and err is not None:
            msg.u_err = float(err[0])
            msg.v_err = float(err[1])
        else:
            msg.u_err = msg.v_err = nan
        if depth is not None and depth > 0.0:
            msg.depth = float(depth)
            msg.radius_px = float(
                self.config.focal_length_u * self.config.real_radius / depth
            )
        else:
            msg.depth = msg.radius_px = nan
        self.target_pixel_err_pub.publish(msg)

    def reset_measurement_attributes(self):
        self.current_target_pixels = None
        self.current_target_radius = None
        self.current_depth_estimate = None
        self.current_target_pixel_error = None
        self.ekf_x = None
        self.ekf_P = None

    # ------------------------------------------------------------------
    # Controller Thread (timer callback)
    # ------------------------------------------------------------------
    def publish_camera_velocity(self, pan_velocity, tilt_velocity):
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [float(pan_velocity), float(tilt_velocity)]
        self.shoot_velocity_cmd_pub.publish(cmd_msg)

    def controller_timer_callback(self):
        if not self.tracking_active or self.ekf_x is None:
            self.prev_pan_error = self.prev_tilt_error = self.prev_control_time = None
            self.publish_camera_velocity(0.0, 0.0)
            return

        try:
            with self.data_lock:
                x_state = self.ekf_x.copy()
                current_tilt = self.current_tilt
                current_pan = self.current_pan
                measurement_found = self.measurement_found
                last_ekf_time = self.ekf_last_time

            if not measurement_found and last_ekf_time is not None:
                # No measurement this frame: propagate the state (constant
                # velocity) to the instant the command is generated
                dt_pred = max(time.time() - last_ekf_time, 0.0)
                x_state[0:3] += x_state[3:6] * dt_pred

            try:
                T_base_cam = transform_to_matrix(
                    self.buffer.lookup_transform(
                        "base_link", self.config.cam_frame_id, rclpy.time.Time()
                    )
                )
                T_cam_laser = transform_to_matrix(
                    self.buffer.lookup_transform(
                        self.config.cam_frame_id, "shoot_laser", rclpy.time.Time()
                    )
                )
            except Exception as e:
                self.get_logger().warn(f"Controller TF error: {e}")
                self.publish_camera_velocity(0.0, 0.0)
                return

            T_o_l = T_base_cam @ T_cam_laser
            T_l_o = np.linalg.inv(T_o_l)

            T_o_t = np.eye(4)
            T_o_t[0:3, 3] = x_state[0:3, 0]  # Correct slicing for 6x1 vector

            T_l_t = T_l_o @ T_o_t
            P_l_t = T_l_t[0:3, 3]  # This creates a 1D array: [x, y, z]

            self.get_logger().info(f"P_l_t: {P_l_t}")

            tilt = math.atan2(P_l_t[0], P_l_t[2])
            # pan = math.atan2(P_l_t[0], math.sqrt(P_l_t[2] ** 2 + P_l_t[1] ** 2))
            pan = math.atan2(P_l_t[1], P_l_t[2])
            self.get_logger().info(f"tilt: {tilt}, pan: {pan}")

            now = self.get_clock().now().nanoseconds * 1e-9
            pan_rate = tilt_rate = 0.0
            if self.prev_control_time is not None:
                dt = now - self.prev_control_time
                if dt > 1e-4:
                    pan_rate = (pan - self.prev_pan_error) / dt
                    tilt_rate = (tilt - self.prev_tilt_error) / dt
            self.prev_pan_error, self.prev_tilt_error = pan, tilt
            self.prev_control_time = now

            vmax = self.config.max_camera_velocity
            self.pan_velocity_cmd = np.clip(
                self.config.camera_velocity_gain * pan
                + self.config.camera_velocity_kd * pan_rate,
                -vmax,
                vmax,
            )
            self.tilt_velocity_cmd = np.clip(
                self.config.camera_velocity_gain * tilt
                + self.config.camera_velocity_kd * tilt_rate,
                -vmax,
                vmax,
            )
            self.get_logger().info(
                f"pan_velocity_cmd: {self.pan_velocity_cmd}, tilt_velocity_cmd: {self.tilt_velocity_cmd}"
            )

            if abs(current_pan) >= np.pi and self.pan_velocity_cmd * current_pan > 0:
                self.pan_velocity_cmd = 0.0
            if abs(current_tilt) >= np.pi and self.tilt_velocity_cmd * current_tilt > 0:
                self.tilt_velocity_cmd = 0.0

            with self.data_lock:
                self.current_tilt_error = abs(math.degrees(tilt))
                self.current_pan_error = abs(math.degrees(pan))
            self.publish_camera_velocity(self.pan_velocity_cmd, self.tilt_velocity_cmd)

        except Exception as e:
            self.publish_camera_velocity(0.0, 0.0)
            self.get_logger().warn(f"Controller timer error: {str(e)}")

    # ------------------------------------------------------------------
    # Execute callback
    # ------------------------------------------------------------------
    def execute_callback(self, goal_handle):
        self.get_logger().info("Executing dynamic target locking...")

        tracking_duration = goal_handle.request.tracking_duration
        feedback_msg = LockTargetDynamic.Feedback()
        result = LockTargetDynamic.Result()

        # --- 1. Goal Acquisition ---
        avg_center, avg_radius = self.acquire_initial_target()
        if avg_center is None:
            result.success = False
            result.message = "Failed to acquire initial target during acquisition phase"
            goal_handle.abort()
            return result

        _, tvec = self.world_coordinates(avg_center, avg_radius)
        if tvec is None:
            result.success = False
            result.message = "Failed to compute initial world coordinates"
            goal_handle.abort()
            return result
        # tvec = np.array([[tvec[2, 0]], [-tvec[0, 0]], [-tvec[1, 0]]])

        self.ekf_init(tvec)
        self.tracking_active = True

        start_time = time.time()

        try:
            while rclpy.ok() and goal_handle.is_active:
                elapsed_time = time.time() - start_time

                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Goal was cancelled")
                    self.reset_measurement_attributes()
                    goal_handle.canceled()
                    result.success = False
                    result.message = "Goal cancelled by client"
                    self.tracking_active = False
                    return result

                if elapsed_time >= tracking_duration:
                    with self.data_lock:
                        tilt_err = self.current_tilt_error
                        pan_err = self.current_pan_error

                    self.tracking_active = False

                    if (
                        tilt_err < self.config.angular_tolerance_deg
                        and pan_err < self.config.angular_tolerance_deg
                    ):
                        result.success = True
                        result.message = (
                            f"Tracking completed successfully (tilt: {tilt_err:.2f}°, "
                            f"pan: {pan_err:.2f}°)"
                        )
                        self.reset_measurement_attributes()
                        goal_handle.succeed()
                    else:
                        result.success = False
                        result.message = (
                            f"Tracking duration elapsed with errors above tolerance "
                            f"(tilt: {tilt_err:.2f}°, pan: {pan_err:.2f}°)"
                        )
                        self.reset_measurement_attributes()
                        goal_handle.abort()

                    return result

                with self.data_lock:
                    feedback_msg.tilt_loss = self.current_tilt_error
                    feedback_msg.pan_loss = self.current_pan_error

                try:
                    goal_handle.publish_feedback(feedback_msg)
                except Exception as feedback_err:
                    # Action server/publisher may already be torn down (e.g. node shutting down)
                    self.get_logger().warn(
                        f"Failed to publish feedback: {feedback_err}"
                    )
                    break

                time.sleep(self.config.control_period)

            # Loop exited because node/goal is no longer active (e.g. shutdown) rather than a normal outcome
            if not (rclpy.ok() and goal_handle.is_active):
                self.tracking_active = False
                result.success = False
                result.message = "Tracking stopped: node or goal no longer active"
                return result

        except Exception as e:
            self.get_logger().error(f"Error during dynamic target locking: {str(e)}")
            self.reset_measurement_attributes()
            self.tracking_active = False
            result.success = False
            result.message = f"Execution error: {str(e)}"
            goal_handle.abort()
            return result

    def generate_Q_matrix(self, dt, varAcc):
        W = np.zeros((6, 3))
        W[0:3, 0:3] = np.eye(3) * 0.5 * dt * dt
        W[3:6, 0:3] = np.eye(3) * dt

        Q = varAcc * np.dot(W, W.T)
        return Q


def main(args=None):
    rclpy.init(args=args)

    lock_target_dynamic_server = LockTargetDynamicServer()

    executor = MultiThreadedExecutor()
    executor.add_node(lock_target_dynamic_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        lock_target_dynamic_server.get_logger().info(
            "Shutting down lock target dynamic server..."
        )
    finally:
        lock_target_dynamic_server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
