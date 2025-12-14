import rclpy
import numpy as np
import math

from rclpy.node import Node
from mpc_controller.mpc_controller import MPCController
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped
from athena_custom_msgs_pkg.msg import PathPoint
from athena_custom_msgs_pkg.srv import SetWaypoints


class MPCControllerNode(Node):
    def __init__(self):
        super().__init__("mpc_controller")
        self.get_logger().info("MPC Controller Node has been started.")

        self.mpc = MPCController(self.get_logger)

        self.dt = 0.02  # 50 Hz
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.has_imu = False
        self.has_path = False

        self.last_imu = None
        self.world_path = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_path_idx = 0  # first loop

        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.path_cli = self.create_client(SetWaypoints, "get_path")
        while not self.path_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Wait for Path service...")

        self.world_path = None
        self.origin = None
        self.has_initialized_position = False

        self.request_path()

        self.imu_sub = self.create_subscription(
            Imu, "/sensors/imu", self.imu_callback, qos_policy
        )

        self.ackermann_pub = self.create_publisher(
            AckermannDriveStamped, "/ackermann_cmd", qos_policy
        )

    def request_path(self):
        self.get_logger().info("Send service request for fixed path...")
        req = SetWaypoints.Request()
        future = self.path_cli.call_async(req)
        future.add_done_callback(self.handle_path_response)

    def handle_path_response(self, future):
        try:
            response = future.result()
            self.world_path = response.path
            self.has_path = True
            self.get_logger().info("Path received.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def imu_callback(self, msg_imu: Imu):
        """
        Save the latest imu data received from the VESC controller.

        Args:
            msg_imu (Imu): Incoming Imu message.
        """
        self.last_imu = msg_imu
        self.has_imu = True

    def control_loop(self):
        if not self.has_imu or not self.has_path:
            return

        x0 = self.compute_state()
        if x0 is None:
            return

        self.current_path_idx = self.find_closest_waypoint(x0)

        x_ss_traj = self.compute_preview_reference()

        try:
            u = self.mpc.step(x0, x_ss_traj)
            self.publish_control(u)
        except Exception as e:
            self.get_logger().warn(f"MPC failed: {e}")
            self.publish_zero()

    def compute_state(self):
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = self.transform_to_yaw(tf.transform.rotation)

        return np.array([x, y, yaw])

    def compute_preview_reference(self):
        N = self.mpc.N

        x_ss = np.zeros((N + 1, 3))

        for k in range(N + 1):
            idx = min(self.current_path_idx + k, len(self.world_path) - 1)
            wp = self.world_path[idx]

            yaw_ref = self.compute_path_yaw(idx)
            x_ss[k, :] = [wp.x, wp.y, yaw_ref]

        return x_ss

    def compute_path_yaw(self, idx):
        if idx == 0:
            wp = self.world_path[0]
            wp_next = self.world_path[1]
        elif idx < len(self.world_path) - 1:
            wp = self.world_path[idx]
            wp_next = self.world_path[idx + 1]
        else:
            wp = self.world_path[idx - 1]
            wp_next = self.world_path[idx]

        return math.atan2(wp_next.y - wp.y, wp_next.x - wp.x)

    def transform_to_yaw(self, q) -> float:
        """
        geometry_msgs/msg/Quaternion orientation is in quaternion so
        there is a transforamtion needed into euler angle.

        Quaternion to Euler angles (in 3-2-1 sequence) conversion
        Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        """

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def publish_control(self, u):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(u[0])
        msg.drive.steering_angle = float(u[1])
        self.ackermann_pub.publish(msg)

    def publish_zero(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.ackermann_pub.publish(msg)

    def find_closest_waypoint(self, x0):
        px, py = x0[0], x0[1]

        dists = [(wp.x - px) ** 2 + (wp.y - py) ** 2 for wp in self.world_path]

        return int(np.argmin(dists))
