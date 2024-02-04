# --------------------------------------------------------------------------- #
# 1-D Unsteady Heat Equation (Object Oriented Approach)
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
# file: heat_transfer_oop.py
#
# --------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

from typing import Callable
from dataclasses import dataclass
import numbers


# -- Main
def main() -> int:
    """'Entry point' to solve the 1-D unsteady heat equation."""
    # -- Time
    run_time = Time(start_time=0.0, end_time=2.0, delta_t=7.0e-5)

    # -- Mesh
    mesh = Mesh(domain_interval=(0.0, 1.0), n_div=81)

    # -- Transport properties
    transportProperties = TransportProperties(k=1.0)

    # -- Stability factor
    sigma = transportProperties.k * run_time.delta_t / mesh.dx**2
    if sigma > 0.5 or sigma < 0.0:
        raise ValueError(
            "Unstable value for sigma. Choose value in range [0, 0.5]"
        )

    # -- Create field, and initial condition
    u = Field(mesh, ic=lambda: 0.5)

    # -- Boundary Conditions
    bc_left_corrector = lambda t: np.sin(5 * t)  # noqa
    bc_right_corrector = lambda: 1.0  # noqa
    u[0] = bc_left_corrector(run_time.value)
    u[-1] = bc_right_corrector()

    # For plotting purposes
    u0 = u.copy()

    # -- Time loop
    while run_time.loop():
        plt.clf()

        # -- Solve for internal nodes
        un = u.copy()
        u[1:-1] = un[1:-1] + sigma * (un[2:] - 2 * un[1:-1] + un[:-2])

        # -- Correct boundary nodes
        u[0] = bc_left_corrector(run_time.value)
        u[-1] = bc_right_corrector()

        # -- Plot
        plot(mesh, u0, u, run_time)

    plt.show()

    return 0


@dataclass
class TransportProperties:
    """Data class for storing transport properties."""
    k: float


class Mesh:
    """Class representing the mesh for discretizing the 1-D domain."""
    def __init__(
        self, domain_interval: tuple[float, float], n_div: int
    ) -> None:
        self._domain = np.linspace(
            domain_interval[0], domain_interval[1], n_div
        )
        self._n_div = n_div
        self._dx = (float)(domain_interval[1] - domain_interval[0]) / (
            n_div - 1
        )

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def n_div(self) -> int:
        return self._n_div

    @property
    def domain(self) -> np.ndarray:
        return self._domain


class Singleton(type):
    """Metaclass for implementing the Singleton pattern."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class Time(metaclass=Singleton):
    """Class representing time parameters and controlling the time loop."""
    def __init__(
        self, start_time: float, end_time: float, delta_t: float
    ) -> None:
        self._start_time = start_time
        self._end_time = end_time
        self._delta_t = delta_t
        self._value: float = 0.0
        self._index: int = 0

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def value(self) -> float:
        return self._value

    @property
    def index(self) -> int:
        return self._index

    def loop(self):
        is_running: bool = self.value < (
            self._end_time - 0.5 * self._delta_t
        )  # Idea from OpenFOAM
        if is_running:
            self._value += self.delta_t
            self._index += 1
        return is_running


class Field:
    """Class representing the customf 1-D field."""
    def __init__(
        self,
        mesh: Mesh,
        ic: Callable,
        # bc_left_corrector: Callable,
        # bc_right_corrector: Callable,
    ) -> None:
        self._mesh = mesh
        self._ic = ic
        # self._bc_left_corrector = bc_left_corrector
        # self._bc_right_corrector = bc_right_corrector
        self._data: np.ndarray = self._ic() * np.ones(self._mesh.n_div)
        # self.correct_boundary_conditions()

    # def correct_boundary_conditions(self, ??): # use partial?
    #     self._bc_left_corrector()
    #     self._bc_right_corrector()

    def asarray(self):
        return np.asarray(self._data)

    def copy(self):
        return self._data.copy()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __add__(self, other):
        return Field(self._data + other._data)

    # TODO: radd?

    def __sub__(self, other):
        return Field(self._data + other._data)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Field(other * self._data)
        else:
            raise ValueError("Vector Multiplication not supported...yet")

    # TODO
    def __repr__() -> str:
        pass

    def __str__() -> str:
        pass


@dataclass
class PlotMeta:
    """Data class containing plot metadata for visualization."""

    u_range: tuple[float] = (-1.0, 1.0)
    x_range: tuple[float] = (0.0, 1.0)
    title: str = "Temperature Distribution"
    x_label: str = "$x$"
    y_label: str = "$u$"
    markersize: int = 2


def plot(mesh: Mesh, u0: Field, u: Field, run_time: Time):
    """A helper function to plot the results and to compare with initial
    field."""
    plt.clf()  # Clear previous figure

    rate: int  # Rate at which data at different times is shown
    text: str  # To show when we speed up the plot by increasing rate
    if run_time.index < 5000:
        rate = 100  # An arbitrary "slow" rate
        text = ""
    else:
        rate = 500  # An arbitrary "fast" rate
        text = '"speed up (x5)"'

    if run_time.index % rate == 0:
        # Title, labels and limits
        plt.title(PlotMeta.title)
        plt.ylabel(PlotMeta.y_label)
        plt.xlabel(PlotMeta.x_label)
        plt.ylim([*PlotMeta.u_range])
        plt.xlim([*PlotMeta.x_range])

        # Initial condition
        plt.plot(
            mesh.domain,
            u0,
            "r--",
            markersize=PlotMeta.markersize,
            label="Initial Condition (t = 0)",
        )

        # Current time step
        plt.plot(
            mesh.domain,
            u.asarray(),
            "b-",
            markersize=PlotMeta.markersize,
            label=f"Current Time "
            f"(time = {run_time.value:.2f}, index = {run_time.index})",
        )

        plt.text(x=0.7, y=-0.7, s=text)
        plt.legend(loc=(0.2, 0))
        plt.pause(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
