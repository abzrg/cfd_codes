# --------------------------------------------------------------------------- #
# 1-D Unsteady Heat Equation
# --------------------------------------------------------------------------- #
#
# Equation:
#   u_t = k u_{xx}
#
# Where:
# - k: head conduction coeff
# - u: temperature field
# - x \in [0, 1]
# - t: time
#
# Initial Conditions and Boundary Conditions:
# - u(x = 0, t) = sin(t)
# - u(x = 1, t) = 1.0
# - u(x, t = 0) = 0.5
#
# Scheme (Explicit):
# - Time: Forward Euler
# - Space: Cerntral Differencing
#
# file: heat_transfer.py
#
# --------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

# -- Transport Properties
k = 1

# -- Geometry
L = 1.0
X_min = 0.0
X_max = L
U_min = -1.0
U_max = +1.0

# -- Solution Parameters
Nx = 81
dx = L / (Nx - 1)
Nt = 5000
# 0 <= Sigma <= 0.5 (stability condition for explicit scheme)
Sigma = 0.305

# -- Initial Condition
ic = lambda: 0.5  # noqa

# -- Boundary Conditions
bc_left_corrector = lambda t: np.sin(10 * t)  # noqa
bc_right_corrector = lambda: 1.0  # noqa


def main():
    dt = Sigma * dx**2 / k
    t = 0.0

    u = ic() * np.ones(Nx)
    u[0] = bc_left_corrector(t)
    u[-1] = bc_right_corrector()
    u0 = u.copy()

    # -- Time loop
    x = np.linspace(X_min, X_max, Nx)
    for n in range(1, Nt + 1):
        t = n * dt

        un = u.copy()
        u[0] = bc_left_corrector(t)
        u[-1] = bc_right_corrector()
        u[1:-1] = un[1:-1] + k * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])

        # Plot every 1000 time-step
        rate: int
        text: str
        if n < 500:
            rate = 10
            text = ""
        else:
            rate = 100
            text = '"speed up"'

        if n % rate == 0:
            plt.clf()

            plt.title("Temperature Distribution")
            plt.ylabel("$u$")
            plt.xlabel("$x$")
            plt.ylim([U_min, U_max])
            plt.xlim([X_min, X_max])

            plt.plot(
                x,
                u0,
                "r--",
                markersize=2,
                label="Initial Condition (t = 0)",
            )

            plt.plot(
                x,
                u,
                "b-",
                markersize=2,
                label=f"Current Time ({t=:.2f})",
            )

            plt.text(x=0.7, y=-0.7, s=text)
            plt.legend(loc=(0.57, 0.01))
            plt.pause(0.1)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
