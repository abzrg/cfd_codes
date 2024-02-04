# From https://github.com/abzrg/nmpy

import numpy as np


def tdma(A: np.ndarray, b: np.ndarray):
    """TDMA (Tri-Diagonal Matrix Algorithm) for a special type of linear system
    of equations.

    Positional arguments:
        A -- Coefficient matrix
        b -- Right-hand-side vector

    Returns:
        Solution vector of the linear system of equations
    """

    ud_entries = np.diag(A, +1).copy()
    d_entries = np.diag(A, 0).copy()
    ld_entries = np.diag(A, -1).copy()

    grid_size = len(d_entries)

    # Define solution field
    x = np.zeros(grid_size)

    # Step 1: Forward elimination
    for i in range(1, grid_size):
        d_entries[i] = (
            d_entries[i]
            - ud_entries[i - 1] * ld_entries[i - 1] / d_entries[i - 1]
        )
        b[i] = b[i] - b[i - 1] * ld_entries[i - 1] / d_entries[i - 1]

    # Step 2: Backward substitution
    x[grid_size - 1] = b[grid_size - 1]
    for i in range(grid_size - 2, -1, -1):
        x[i] = (b[i] - ud_entries[i] * x[i + 1]) / d_entries[i]

    return x


def jacobi(
    A: np.ndarray,
    x_0: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int = 50,
) -> np.ndarray:
    """Jacobi iterative method for linear system of equations

    Positional arguments:
        A   -- matrix coefficient
        x_0 -- initial guess
        b   -- right hand side vector
        tol -- absolute tolerance

    Keyword arguments:
        max_iter -- maximum number of iterations

    Returns:
        Solution to the linear system of equations
    """
    # Make sure A is a Numpy array
    A = np.array(A)
    num_row, num_col = A.shape

    if num_row != num_col:
        raise ValueError(f"A[{num_row} x {num_col}] is not a a square matrix.")

    if num_col != len(b):
        raise ValueError(f"Incompatible dimension: A[{A.shape}], b[{b.shape}]")

    if not max_iter > 0:
        raise ValueError(
            f"Maximum number of iteration must be greater than 0: {max_iter=}"
        )

    # Main diagonal vector of matrix A
    d: np.ndarray = np.diag(A, 0)

    # Initialize solution vectors
    x_curr: np.ndarray = np.zeros(num_row)  # Current iteration (Solution)
    x_prev: np.ndarray = x_0  # Previous iteration

    # Initialize error (absolute approximate error)
    residual = b - A @ x_0  # Residual vector
    error: float = np.linalg.norm(residual, 2)

    k: int = 0  # Iteration counter
    while error > tol and k < max_iter:
        x_curr = x_prev + (1.0 / d) * (b - A @ x_prev)

        # Compute the (absolute) approximate error
        error = np.linalg.norm(x_curr - x_prev, 2)

        # Update the guess
        x_prev = x_curr.copy()

        k += 1

    return x_curr


def gs(
    A: np.ndarray,
    x_0: np.ndarray,
    b: np.ndarray,
    tol: float,
    omega: float = 1.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Gauss-Seidel iterative method for linear system of equations.

    Positional arguments:
        A   -- Coefficient matrix
        x_0 -- Initial guess
        b   -- Right hand side vector
        tol -- Absolute tolerance

    Keyword arguments:
        max_iter -- Maximum number of iterations
        omega    -- Relaxation parameter (sor method) 0 < omega < 2

    Returns:
        Solution vector of the linear system of equations
    """
    # Make sure A is a Numpy array
    A = np.array(A)
    num_row, num_col = A.shape

    if num_row != num_col:
        raise ValueError(f"A[{num_row} x {num_col}] is not a a square matrix.")

    if num_col != len(b):
        raise ValueError(f"Incompatible dimension: A[{A.shape}], b[{b.shape}]")

    if not 0 < omega < 2:
        raise ValueError(f"Invalid {omega=}. 0 < omega < 2")

    if not max_iter > 0:
        raise ValueError(
            f"Maximum number of iteration must be greater than 0: {max_iter=}"
        )

    # Main diagonal vector of matrix A
    d: np.ndarray = np.diag(A, 0)

    # Initialize solution vectors
    x_curr: np.ndarray = np.zeros(num_row)  # Current iteration (Solution)
    x_prev: np.ndarray = x_0  # Previous iteration

    # Initialize error (absolute approximate error)
    residual = b - A @ x_0  # Residual vector
    error: float = np.linalg.norm(residual, 2)

    k: int = 0  # Iteration counter
    while error > tol and k < max_iter:
        x_curr = x_prev + (omega / d) * (b - A @ x_curr)

        # Compute the (absolute) approximate error
        error = np.linalg.norm(x_curr - x_prev, 2)

        # Update the guess
        x_prev = x_curr.copy()

        k += 1

    return x_curr
