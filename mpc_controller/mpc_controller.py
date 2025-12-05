from rclpy.impl.rcutils_logger import RcutilsLogger

import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from mpc_controller.bicycle_model import BicycleModel


class MPCController:
    def __init__(self, logger: RcutilsLogger):
        super().__init__("mpc_controller")

        self.get_logger = logger
        self.get_logger().info("MPC Controller calculation started.")

        self.bicycle_model = BicycleModel().model

        ocp = AcadosOcp()
        ocp.model = self.model

        ocp.dims.N = 10
        ocp.solver_options.integrator_type = "DISCRETE"

        # add cost, constraints, bounds ... idk

        # build solver
        self.solver = AcadosOcpSolver(ocp, json_file="bicycle_mpc.json")

    def step(self, x0):
        # set initial state
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        status = self.solver.solve()
        return self.solver.get(0, "u")
