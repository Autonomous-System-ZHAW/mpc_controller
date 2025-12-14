from rclpy.impl.rcutils_logger import RcutilsLogger

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_controller.bicycle_model import BicycleModel

MAX_STEERING_ANGLE_RADIANS = 0.37  # rad
MIN_STEERING_ANGLE_RADIANS = -0.37  # rad
MAX_SPEED = 2.0  # m/s
MIN_SPEED = 0.0  # m/s


class MPCController:
    def __init__(self, logger: RcutilsLogger):
        super().__init__("mpc_controller")

        self.get_logger = logger
        self.get_logger().info("MPC Controller calculation started.")

        self.model = BicycleModel().model

        ocp = AcadosOcp()
        ocp.model = self.model

        self.N = 10
        ocp.dims.N = self.N
        ocp.solver_options.integrator_type = "DISCRETE"

        # constraints
        ocp.constraints.lbu = np.array([MIN_SPEED, MIN_STEERING_ANGLE_RADIANS])
        ocp.constraints.ubu = np.array([MAX_SPEED, MAX_STEERING_ANGLE_RADIANS])
        ocp.constraints.idxbu = np.array([0, 1])

        # cost for cost function
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]

        Q = np.diag([10.0, 10.0, 5.0])  # X, Y, yaw
        R = np.diag([0.1, 0.1])  # v, delta

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"  # terminal cost k = N

        ocp.cost.W = np.block(
            [[Q, np.zeros((self.nx, self.nu))], [np.zeros((self.nu, self.nx)), R]]
        )
        ocp.cost.W_e = Q

        ocp.cost.Vx = np.hstack([np.eye(self.nx), np.zeros((self.nx, self.nu))])
        ocp.cost.Vu = np.hstack([np.zeros((self.nu, self.nx)), np.eye(self.nu)])
        ocp.cost.Vx_e = np.eye(self.nx)

        ocp.cost.yref = np.zeros(self.nx + self.nu)
        ocp.cost.yref_e = np.zeros(self.nx)

        self.solver = AcadosOcpSolver(ocp, json_file="bicycle_mpc.json")

    def step(self, x0, x_ss_traj):
        """
        x0         : current state [x, y, yaw]
        x_ss_traj  : reference states over horizon (N+1, nx)
        """

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        for k in range(self.N):
            yref = np.hstack([x_ss_traj[k], np.zeros(self.nu)])
            self.solver.set(k, "yref", yref)

        self.solver.set(self.N, "yref_e", x_ss_traj[self.N])  # last state in horizont

        status = self.solver.solve()
        if status != 0:
            self.get_logger().warn(f"MPC solver returned status {status}")

        return self.solver.get(0, "u")
