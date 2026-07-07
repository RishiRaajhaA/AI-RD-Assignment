"""
AI R&D Assignment: Parameter Estimation for a Parametric Curve

This script estimates the unknown parameters theta, M, and X from xy_data.csv.

Given curve:
    x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
    y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)

Parameter ranges:
    0 < theta < 50 degrees
    -0.05 < M < 0.05
    0 < X < 100
    6 <= t <= 60

Method:
    Differential Evolution is used to minimize a symmetric nearest-neighbour
    L1 distance between the points in xy_data.csv and the predicted curve.
    This is suitable because the CSV points are not assumed to be ordered by t.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree


T_MIN = 6.0
T_MAX = 60.0
DATA_PATH = Path("xy_data.csv")
PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("results")


def generate_curve(theta_deg: float, m_value: float, x_shift: float, n_points: int = 1500):
    """Generate points on the parametric curve for t uniformly sampled in [6, 60]."""
    theta_rad = np.deg2rad(theta_deg)
    t = np.linspace(T_MIN, T_MAX, n_points)

    exp_term = np.exp(m_value * np.abs(t))
    sine_term = np.sin(0.3 * t)

    x = t * np.cos(theta_rad) - exp_term * sine_term * np.sin(theta_rad) + x_shift
    y = 42 + t * np.sin(theta_rad) + exp_term * sine_term * np.cos(theta_rad)

    return t, x, y


def symmetric_l1_loss(params: np.ndarray, given_points: np.ndarray) -> float:
    """
    Compute symmetric nearest-neighbour L1 loss.

    The assignment evaluates L1 distance. Since xy_data.csv may not be sorted by
    the parameter t, nearest-neighbour matching is used instead of direct row-wise
    comparison. p=1 in cKDTree means Manhattan / L1 distance.
    """
    theta_deg, m_value, x_shift = params

    _, x_pred, y_pred = generate_curve(
        theta_deg=theta_deg,
        m_value=m_value,
        x_shift=x_shift,
        n_points=len(given_points),
    )
    predicted_points = np.column_stack((x_pred, y_pred))

    predicted_tree = cKDTree(predicted_points)
    data_to_curve_distance, _ = predicted_tree.query(given_points, k=1, p=1)

    data_tree = cKDTree(given_points)
    curve_to_data_distance, _ = data_tree.query(predicted_points, k=1, p=1)

    return float(np.mean(data_to_curve_distance) + np.mean(curve_to_data_distance))


def estimate_parameters(given_points: np.ndarray):
    """Estimate theta, M, and X using Differential Evolution."""
    bounds = [
        (0.0, 50.0),       # theta in degrees
        (-0.05, 0.05),     # M
        (0.0, 100.0),      # X
    ]

    result = differential_evolution(
        symmetric_l1_loss,
        bounds=bounds,
        args=(given_points,),
        seed=42,
        maxiter=800,
        tol=1e-10,
        polish=True,
        updating="immediate",
        workers=1,
    )

    return result


def make_submission_equation(theta_rad: float, m_value: float, x_shift: float) -> str:
    """Create the final equation string in LaTeX/Desmos-friendly format."""
    return (
        rf"\left(t\cos({theta_rad:.6f})"
        rf"-e^{{{m_value:.6f}\left|t\right|}}\sin(0.3t)\sin({theta_rad:.6f})"
        rf"+{x_shift:.6f},"
        rf"42+t\sin({theta_rad:.6f})"
        rf"+e^{{{m_value:.6f}\left|t\right|}}\sin(0.3t)\cos({theta_rad:.6f})\right)"
    )


def save_plots(data: pd.DataFrame, theta_deg: float, m_value: float, x_shift: float) -> None:
    """Save given-data, fitted-curve, and comparison plots."""
    PLOTS_DIR.mkdir(exist_ok=True)

    _, x_fit, y_fit = generate_curve(
        theta_deg=theta_deg,
        m_value=m_value,
        x_shift=x_shift,
        n_points=1500,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(data["x"], data["y"], s=8, label="Given xy_data.csv points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Given Points")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "given_points.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(x_fit, y_fit, linewidth=2, label="Predicted fitted curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted Fitted Parametric Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predicted_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(data["x"], data["y"], s=8, label="Given xy_data.csv points")
    plt.plot(x_fit, y_fit, linewidth=2, label="Predicted fitted curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Given Points vs Fitted Parametric Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fitted_curve.png", dpi=300)
    plt.close()


def save_results(result, theta_deg: float, theta_rad: float, m_value: float, x_shift: float) -> None:
    """Save final parameter values, optimization statistics, and final equation."""
    RESULTS_DIR.mkdir(exist_ok=True)

    submission_equation = make_submission_equation(theta_rad, m_value, x_shift)

    estimated_text = f"""AI R&D Assignment - Final Estimated Parameters

Theta (degrees): {theta_deg:.8f}
Theta (radians): {theta_rad:.8f}
M              : {m_value:.8f}
X              : {x_shift:.8f}
L1 Loss        : {float(result.fun):.10f}

Optimization statistics:
Iterations           : {result.nit}
Function evaluations : {result.nfev}
Optimization success : {result.success}
Optimizer message    : {result.message}

Rounded final answer:
Theta = {theta_deg:.2f} degrees
M     = {m_value:.2f}
X     = {x_shift:.2f}
"""

    (RESULTS_DIR / "estimated_parameters.txt").write_text(estimated_text, encoding="utf-8")
    (RESULTS_DIR / "final_equation.txt").write_text(submission_equation, encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "xy_data.csv was not found. Place main.py in the same folder as xy_data.csv."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = {"x", "y"}
    if not required_columns.issubset(data.columns):
        raise ValueError("xy_data.csv must contain columns named 'x' and 'y'.")

    given_points = data[["x", "y"]].to_numpy()
    result = estimate_parameters(given_points)

    theta_deg, m_value, x_shift = result.x
    theta_rad = np.deg2rad(theta_deg)
    submission_equation = make_submission_equation(theta_rad, m_value, x_shift)

    print("Recovered parameters")
    print("--------------------")
    print(f"theta = {theta_deg:.8f} degrees")
    print(f"theta = {theta_rad:.8f} radians")
    print(f"M     = {m_value:.8f}")
    print(f"X     = {x_shift:.8f}")
    print(f"Loss  = {result.fun:.10f}")

    print("\nOptimization statistics")
    print("-----------------------")
    print(f"Iterations           : {result.nit}")
    print(f"Function evaluations : {result.nfev}")
    print(f"Optimization success : {result.success}")
    print(f"Optimizer message    : {result.message}")

    print("\nRounded final answer")
    print("--------------------")
    print(f"theta = {theta_deg:.2f} degrees")
    print(f"M     = {m_value:.2f}")
    print(f"X     = {x_shift:.2f}")

    print("\nDesmos / LaTeX submission format")
    print("--------------------------------")
    print(submission_equation)

    save_plots(data, theta_deg, m_value, x_shift)
    save_results(result, theta_deg, theta_rad, m_value, x_shift)

    print("\nSaved outputs")
    print("-------------")
    print("plots/given_points.png")
    print("plots/predicted_curve.png")
    print("plots/fitted_curve.png")
    print("results/estimated_parameters.txt")
    print("results/final_equation.txt")


if __name__ == "__main__":
    main()
