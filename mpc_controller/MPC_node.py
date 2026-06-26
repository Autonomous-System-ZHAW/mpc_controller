import rerun as rr
import yaml
import time
import rerun.blueprint as rrb

# from rclpy.node import Node


class MPCNode:

    def __init__(self):
        # super().__init__("mpc")

        self.waypoints = []

        rr.init("follow_the_gap", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self.load_from_yaml()
        self.visualization_path()

    def visualization_path(self):
        rr.set_time("sim_time", timestamp=time.time())
        print("hello")
        rr.log(
            "path", rr.Points2D(self.waypoints), radii=[0.1, 0.3], colors=[0, 0, 255]
        )

    def load_from_yaml(self):
        with open("config/waypoints.yaml", "r") as stream:
            self.waypoints = yaml.full_load(stream)
