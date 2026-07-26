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


class MPCNode(Node):
    def __init__(self):
        super().__init__("mpc")
        self.get_logger().info("MPC Node has been started.")

        self.waypoints = []
        self.reference_path = None
        self.vehicle = None
        self.controller = None
        self.x0 = None

        self.load_from_yaml()
        # self.get_logger().info(f"Loaded waypoints: {self.waypoints}")

        marker_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.waypoint_pub = self.create_publisher(
            Marker,
            "/debug/waypoints",
            marker_qos,
        )

        self.current_waypoint_pub = self.create_publisher(
            Marker,
            "/debug/current_waypoint",
            marker_qos,
        )

        self.publish_waypoints()

        self.environment_setup()
        self.MPC_Problem_setup()

        self.set_initial_state()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self.mpc_timer = self.create_timer(
            0.025,
            self.mpc_callback,
        )

        self.waypoint_timer = self.create_timer(
            0.05,
            self.publish_waypoints,
        )

        self.ackermann_pub = self.create_publisher(
            AckermannDriveStamped,
            "/ackermann_cmd",
            10,
        )

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # based on https://www.vcalc.com/wiki/quaternion-to-roll-pitch-yaw
        # convert a quanternium into yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        psi = math.atan2(siny_cosp, cosy_cosp)

        velocity = msg.twist.twist.linear.x

        self.x0 = np.array(
            [
                x + 1.75,  # offset of the map in x
                y + 5.25,  # offset of the map in y
                psi - np.pi / 2,  # because of 90° offset, rotation of the map
                velocity,
                0.0,
            ]
        )

        print(f"x0: ", self.x0)

        # self.get_logger().info(f"New x0: {self.x0}")

        # self.mpc_callback()

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

        # Größe der Punkte
        marker.scale.x = 0.1
        marker.scale.y = 0.1

        # Rot
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for x, y in self.waypoints:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0

            marker.points.append(p)

        self.current_waypoint_pub.publish(marker)

    def publish_current_waypoints(self, current_x, current_y):
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

        self.waypoint_pub.publish(marker)

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
        path_resolution = 0.1

        # Create smoothed reference path
        self.reference_path = ReferencePath(
            self.waypoints,
            path_resolution,
            smoothing_distance=5,
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

        # print("NODE x0 BEFORE MPC:", self.x0)
        # print("MPC INTERNAL x0:", self.controller.mpc.x0)

        u, current_x, current_y = self.controller.get_control(self.x0)

        # print("MPC INPUT:", self.x0)
        # print("MPC OUTPUT u:", u)
        # print("current reference:", current_x, current_y)

        self.publish_current_waypoints(
            current_x,
            current_y,
        )

        self.controller.distance_update(self.x0)

        self.controller.constraints_setup()

        dt = 0.025

        acc = float(u[0])
        delta = float(u[1])

        current_velocity = self.x0[3]

        target_velocity = current_velocity + acc * dt

        # Dein Speed Profile hat v_min=0 und v_max=1
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

        print(f"steering angle: ", delta)
        self.ackermann_pub.publish(ackermann_msg)


def main():
    rclpy.init()

    node = MPCNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
