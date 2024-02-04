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
# Scheme (Implicit):
# - Time: Backward Euler
# - Space: Cerntral Differencing
#
# file: heat_transfer.py
#
# --------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import numpy as np
from linalg import gs
import time

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
Nt = 5500
Sigma = 1  # 0 <= sigma <= 0.5 (stability condition for explicit scheme)
dt = Sigma * dx**2 / k
t = 0.0

# -- Initial Condition
ic = lambda: 0.5  # noqa
u0 = ic() * np.ones(Nx)

# -- Boundary Conditions
bc_left_corrector = lambda t: np.sin(10 * t)  # noqa
bc_right_corrector = lambda: 1  # noqa
u0[0] = bc_left_corrector(t)
u0[-1] = bc_right_corrector()


def main():
    # Matrix of Coefficients
    A = np.zeros(shape=(Nx, Nx))

    # Diagonal and off-diagnoal entries
    lambda_coeff = 2 * dt / dx**2
    diagonal_entries = 1.0 + 2 * lambda_coeff * np.ones(Nx)
    upper_diagonal_entries = -1 * lambda_coeff * np.ones(Nx - 1)
    upper_diagonal_entries[0] = 0.0
    lower_diagonal_entries = -1 * lambda_coeff * np.ones(Nx - 1)
    lower_diagonal_entries[-1] = 0.0

    # Constructing the Matrix of coeffs
    A = (
        np.diag(lower_diagonal_entries, k=-1)
        + np.diag(diagonal_entries, k=0)
        + np.diag(upper_diagonal_entries, k=1)
    )
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    # Construct the rhs vector
    # (for comparing my linear solver (gs: gauss-seidel) with
    # numpy's direct linear solver)
    b_solve = u0.copy()
    b_gs = u0.copy()

    u_solve = u0.copy()
    u_gs = u0.copy()

    cpu_time_solve: float = 0.0
    cpu_time_gs: float = 0.0

    # -- Time loop
    x = np.linspace(X_min, X_max, Nx)
    for n in range(1, Nt + 1):
        t = n * dt

        # Update the old values (un)
        un_solve = u_solve.copy()
        un_gs = u_gs.copy()

        # Solve the linear system of equations
        start_time = time.time()
        u_solve = np.linalg.solve(A, b_solve)
        end_time = time.time()
        cpu_time_solve += end_time - start_time

        start_time = time.time()
        u_gs = gs(A, un_gs, b_gs, tol=1e-8)
        end_time = time.time()
        cpu_time_gs += end_time - start_time

        # Update the RHS
        b_gs[1:-1] = un_gs[1:-1] + dt * k / dx**2 * (
            un_gs[:-2] - 2 * un_gs[1:-1] + un_gs[2:]
        )
        b_gs[0] = bc_left_corrector(t)
        b_gs[-1] = bc_right_corrector()

        b_solve[1:-1] = un_solve[1:-1] + dt * k / dx**2 * (
            un_solve[:-2] - 2 * un_solve[1:-1] + un_solve[2:]
        )
        b_solve[0] = bc_left_corrector(t)
        b_solve[-1] = bc_right_corrector()

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
                u_solve,
                "b-",
                markersize=1,
                label=f"numpy.linalg.solve ({t=:.2f})",
            )

            plt.plot(
                x,
                u_solve,
                "k-",
                markersize=1,
                label=f"nmpy.linalg.gs ({t=:.2f})",
            )

            plt.text(x=0.68, y=-0.6, s=text)
            plt.legend(loc=(0.505, 0.01))

            plt.pause(0.1)

    print(
        "[Total Elapsed TIME]\n"
        f"* np.linalg.solve: {cpu_time_solve}\n"  # Almost x3 faster
        f"* nmpy.linalg.gs:  {cpu_time_gs}"
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
