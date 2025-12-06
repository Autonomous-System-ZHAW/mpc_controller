from rclpy.impl.rcutils_logger import RcutilsLogger

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from mpc_controller.bicycle_model import BicycleModel

MAX_STEERING_ANGLE_RADIANS = 0.37  # rad
MIN_STEERING_ANGLE_RADIANS = -0.37  # rad
MAX_SPEED = 4.0  # m/s
MIN_SPEED = 0.0  # m/s


class MPCController:
    def __init__(self, logger: RcutilsLogger):
        super().__init__("mpc_controller")

        self.get_logger = logger
        self.get_logger().info("MPC Controller calculation started.")

        ocp.model = self.bicycle_model

        ocp = AcadosOcp()
        ocp.model = self.model

        ocp.dims.N = 10
        ocp.solver_options.integrator_type = "DISCRETE"

        # constraints
        ocp.constraints.lbu = np.array([MIN_SPEED, MIN_STEERING_ANGLE_RADIANS])
        ocp.constraints.ubu = np.array([MAX_SPEED, MAX_STEERING_ANGLE_RADIANS])
        ocp.constraints.idxbu = np.array([0, 1])

        # cost for cost function
        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]

        Q = np.diag([10.0, 10.0, 5.0])  # X, Y, v
        R = np.diag([0.1, 0.1])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"  # terminal cost k = N

        ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
        ocp.cost.W_e = Q

        ocp.cost.Vx = np.hstack([np.eye(nx), np.zeros((nx, nu))])  # no impact of u
        ocp.cost.Vu = np.hstack([np.zeros((nu, nx)), np.eye(nu)])  # no impact of x
        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        self.solver = AcadosOcpSolver(ocp, json_file="bicycle_mpc.json")

    def step(self, x0):
        # set initial state
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        status = self.solver.solve()
        return self.solver.get(0, "u")
