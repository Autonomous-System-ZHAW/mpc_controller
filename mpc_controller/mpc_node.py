import time

import numpy as np
import rclpy
import yaml
import math

from . import globals


from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker

from rclpy.qos import qos_profile_sensor_data
from mpc_controller.MPC import MPC
from mpc_controller.model import ReferencePath, simple_bycicle_model
from nav_msgs.msg._odometry import Odometry
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc")
        self.get_logger().info("MPC Node has been started.")

        self.waypoints = []
        self.reference_path = None
        self.vehicle = None
        self.controller = None
        self.x0 = None
        self.last_odom_time_stamp = None
        self.latest_odom_time = None
        self.latest_odom_dt = None
        self.mpc_period = 0.025

        self.load_from_yaml()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.current_waypoint_pub = self.create_publisher(
            Marker,
            "/debug/current_waypoint",
            marker_qos,
        )

        self.waypoint_pub = self.create_publisher(
            Marker,
            "/debug/waypoints",
            marker_qos,
        )

        self.environment_setup()
        self.MPC_Problem_setup()

        self.set_initial_state()

        self.publish_waypoints()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.vehicle_state_callback,
            qos_profile_sensor_data,
        )

        self.mpc_timer = self.create_timer(
            self.mpc_period,
            self.mpc_callback,
        )

        self.ackermann_pub = self.create_publisher(
            AckermannDriveStamped,
            "/ackermann_cmd",
            10,
        )

    def vehicle_state_callback(self, msg: Odometry):
        """Update the vehicle state self.x0 = [x, y, psi, v, 0] in the World frame.

        Triggered by every /odom message:
          - pose (x, y, psi) from TF World -> base_link (same source as Foxglove)
          - speed v from the odometry message (forward speed in the car frame)
          - dt between two odometry messages, for the distance update of the MPC
        """
        current_time = Time.from_msg(msg.header.stamp)

        if self.last_odom_time_stamp is None:
            self.last_odom_time_stamp = current_time
            return

        dt_odom = (current_time - self.last_odom_time_stamp).nanoseconds / 1e9

        self.last_odom_time_stamp = current_time

        # car pose in World from TF: World -> odom (start pose, published by Isaac)
        # combined with odom -> base_link (odometry). Same source as Foxglove, so the
        # MPC and the display always agree. Time() = latest available transform.
        try:
            tf = self.tf_buffer.lookup_transform("World", "base_link", Time())
        except Exception as error:
            self.get_logger().warn(
                f"no TF World -> base_link yet: {error}", throttle_duration_sec=2.0
            )
            return

        x_map = tf.transform.translation.x
        y_map = tf.transform.translation.y

        # based on https://www.vcalc.com/wiki/quaternion-to-roll-pitch-yaw
        # convert a quaternion into yaw
        q = tf.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        psi_map = math.atan2(siny_cosp, cosy_cosp)

        # forward speed in the car frame (from the odometry message)
        velocity = msg.twist.twist.linear.x

        self.x0 = np.array(
            [
                x_map,
                y_map,
                psi_map,
                velocity,
                0.0,
            ]
        )

        self.latest_odom_dt = dt_odom
        self.latest_odom_time = self.get_clock().now()

    def load_from_yaml(self):
        with open(
            "athena-simulation-isaac-sim/src/mpc_controller/config/waypoints-stage-2-track.yaml",
            "r",
        ) as stream:
            data = yaml.full_load(stream)
            self.waypoints = data["waypoints"]  # -> list of [x, y] pairs

    def publish_waypoints(self):
        marker = Marker()

        marker.header.frame_id = "World"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.1
        marker.scale.y = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # for x, y in self.waypoints:
        #     p = Point()
        #     p.x = float(x)
        #     p.y = float(y)
        #     p.z = 0.0

        #     marker.points.append(p)

        for waypoint in self.reference_path.waypoints:
            p = Point()
            p.x = float(waypoint.x)
            p.y = float(waypoint.y)
            p.z = 0.0
            marker.points.append(p)

        self.waypoint_pub.publish(marker)

    def publish_current_waypoint(self, current_x, current_y):
        marker = Marker()

        marker.header.frame_id = "World"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "current_waypoint"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Größe der Punkte
        marker.scale.x = 0.5
        marker.scale.y = 0.5

        # Rot
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        p = Point()
        p.x = float(current_x)
        p.y = float(current_y)
        p.z = 0.0

        marker.points.append(p)

        self.current_waypoint_pub.publish(marker)

    def MPC_Problem_setup(
        self,
        ay_max=4.0,
        a_min=-1,
        a_max=1,
    ):
        """
        Get configured do-mpc modules.
        """

        # Model setup
        self.vehicle = simple_bycicle_model(
            # length=0.28,
            # width=0.14,
            length=0.26,
            width=0.17,
            reference_path=self.reference_path,
            # Ts=0.05,
        )
        self.vehicle.model_setup()

        self.controller = MPC(self.vehicle)

        self.controller.constraints_setup()

        # Compute speed profile
        SpeedProfileConstraints = {
            "a_min": a_min,
            "a_max": a_max,
            "v_min": 0.1,
            "v_max": 1.0,
            "ay_max": ay_max,
        }

        self.vehicle.reference_path.compute_speed_profile(SpeedProfileConstraints)

    def environment_setup(self):
        path_resolution = 0.01

        # Create smoothed reference path
        self.reference_path = ReferencePath(
            self.waypoints,
            path_resolution,
            smoothing_distance=10,
            max_width=0.2,
            circular=True,
        )

    def set_initial_state(self):
        if self.x0 is None:
            return

        # Set initial state of the vehicle
        self.x0 = np.array(
            [
                self.vehicle.reference_path.waypoints[0].x,
                self.vehicle.reference_path.waypoints[0].y,
                self.vehicle.reference_path.waypoints[0].psi,
                0.3,
                0,
            ]
        )

        self.controller.mpc.x0 = self.x0

        self.controller.mpc.set_initial_guess()

    def find_closest_waypoint(self, x, y):
        min_distance = float("inf")
        closest_wp_id = 0

        for i, waypoint in enumerate(self.vehicle.reference_path.waypoints):
            distance = np.sqrt((waypoint.x - x) ** 2 + (waypoint.y - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_wp_id = i

        return closest_wp_id

    def mpc_callback(self):
        if self.x0 is None:
            return

        # 1. which waypoint is the car closest to (index)
        closest_wp_id = self.find_closest_waypoint(self.x0[0], self.x0[1])

        # 2. translate index -> meters along the path
        sum_length = np.concatenate(
            ([0.0], np.cumsum(self.vehicle.reference_path.segment_lengths))
        )
        globals.s = sum_length[closest_wp_id]

        # 3. pass the vehicle state, as before
        u, current_x, current_y = self.controller.get_control(self.x0)

        self.publish_current_waypoint(
            current_x,
            current_y,
        )

        # self.controller.distance_update(self.x0, self.latest_odom_dt)

        acc = float(u[0])
        delta = float(u[1])

        target_velocity = float(self.x0[3]) + acc * self.mpc_period

        target_velocity = np.clip(
            target_velocity,
            0.0,
            1.0,
        )

        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.speed = float(target_velocity)
        ackermann_msg.drive.acceleration = acc
        ackermann_msg.drive.steering_angle = delta

        self.ackermann_pub.publish(ackermann_msg)

        self.get_logger().info(
            f"psi car: {self.x0[2]:.2f}  psi ref: {self.controller.vehicle.current_waypoint.psi:.2f}"
        )


def main():
    rclpy.init()

    node = MPCNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
