import rclpy
from rclpy.node import Node
from mpc_controller.mpc_controller import MPCController
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Path, Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from athena_custom_msgs_pkg.msg import PathPoint
from athena_custom_msgs_pkg.srv import SetWaypoints


class MPCControllerNode(Node):
    def __init__(self):
        super().__init__("mpc_controller")
        self.get_logger().info("MPC Controller Node has been started.")

        self.calculate = MPCController(self.get_logger)
        self.tf_buffer = Buffer()
        self.emergency = False

        self.tf_listener = TransformListener(self.tf_buffer, self)
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

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_policy
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

            self.get_logger().info("Path received.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def odom_callback(self, msg_odom: Odometry):
        """
        Save the latest odometry data received from the VESC controller.

        Args:
            msg_odom (Odometry): Incoming Odometry message.
        """

        self.last_odom = msg_odom
