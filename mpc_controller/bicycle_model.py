from casadi import SX, vertcat
from acados_template import AcadosModel


class BicycleModel:
    def __init__(self, L=0.33, dt=0.01):
        # car parameter
        self.L = L
        self.dt = dt

        # states
        X = SX.sym("X")
        Y = SX.sym("Y")
        psi = SX.sym("psi")
        x = vertcat(X, Y, psi)

        # inputs
        v = SX.sym("v")
        delta = SX.sym("delta")
        u = vertcat(v, delta)

        # continuous dynamics
        dX = v * SX.cos(psi)
        dY = v * SX.sin(psi)
        dpsi = v / self.L * SX.tan(delta)

        f = vertcat(dX, dY, dpsi)

        # euler discretization
        x_next = x + dt * f

        # acados model
        model = AcadosModel()
        model.name = "bicycle_model"
        model.x = x
        model.u = u
        model.xdot = None  # not needed for DISCRETE
        model.cont_dyn_expr = f  # optional
        model.disc_dyn_expr = x_next  # REQUIRED

        self.model = model
