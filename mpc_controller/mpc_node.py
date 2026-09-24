import time

import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
import yaml
import math

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
            self.odom_callback,
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

    def odom_callback(self, msg: Odometry):
        current_time = Time.from_msg(msg.header.stamp)

        if self.last_odom_time_stamp is None:
            self.last_odom_time_stamp = current_time
            return

        dt_odom = (current_time - self.last_odom_time_stamp).nanoseconds / 1e9

        self.last_odom_time_stamp = current_time

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # based on https://www.vcalc.com/wiki/quaternion-to-roll-pitch-yaw
        # convert a quanternium into yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        psi = math.atan2(siny_cosp, cosy_cosp)

        velocity = msg.twist.twist.linear.x

        theta = -np.pi / 2
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_map = 1.75 + cos_theta * x - sin_theta * y
        y_map = 5.25 + sin_theta * x + cos_theta * y
        psi_map = np.arctan2(
            np.sin(psi + theta),
            np.cos(psi + theta),
        )

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
            "athena-simulation-isaac-sim/src/mpc_controller/config/waypoints.yaml",
            "r",
        ) as stream:
            data = yaml.full_load(stream)
            self.waypoints = data["waypoints"]  # -> list of [x, y] pairs

    def publish_waypoints(self):
        marker = Marker()

        marker.header.frame_id = "world"
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

        marker.header.frame_id = "world"
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
            width=0.135,
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

    def mpc_callback(self):
        if self.x0 is None:
            return

        u, current_x, current_y = self.controller.get_control(self.x0)

        self.publish_current_waypoint(
            current_x,
            current_y,
        )

        self.controller.distance_update(self.x0, self.latest_odom_dt)

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


def main():
    rclpy.init()

    node = MPCNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
