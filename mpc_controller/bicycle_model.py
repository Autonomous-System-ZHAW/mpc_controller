from casadi import SX, vertcat
from acados_template import AcadosModel

MAX_STEERING_ANGLE_RADIANS = 0.37  # rad
MIN_STEERING_ANGLE_RADIANS = -0.37  # rad


class BicycleModel:
    def __init__(self, L=0.33, dt=0.01):
        # car parameter
        self.L = L
        self.dt = dt
        self.steering_angle_max = MAX_STEERING_ANGLE_RADIANS
        self.steering_angle_min = MIN_STEERING_ANGLE_RADIANS

        # states
        X = SX.sym("X")
        Y = SX.sym("Y")
        psi = SX.sym("psi")
        v = SX.sym("v")
        x = vertcat(X, Y, psi, v)

        # inputs
        a = SX.sym("a")
        delta = SX.sym("delta")
        u = vertcat(a, delta)

        # continuous dynamics
        dX = v * SX.cos(psi)
        dY = v * SX.sin(psi)
        dpsi = v / self.L * SX.tan(delta)
        dv = a
        f = vertcat(dX, dY, dpsi, dv)

        # euler discretization
        x_next = x + self.dt * f

        # acados model
        model = AcadosModel()
        model.name = "bicycle_model"
        model.x = x
        model.u = u
        model.xdot = None  # not needed for DISCRETE
        model.cont_dyn_expr = f  # optional
        model.disc_dyn_expr = x_next  # REQUIRED

        self.model = model
