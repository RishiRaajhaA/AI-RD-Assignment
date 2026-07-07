# AI R&D Assignment — Parametric Curve Parameter Estimation

## Problem Statement

The assignment asks us to find the unknown parameters of the given parametric curve:

\[
x(t)=t\cos(\theta)-e^{M|t|}\sin(0.3t)\sin(\theta)+X
\]

\[
y(t)=42+t\sin(\theta)+e^{M|t|}\sin(0.3t)\cos(\theta)
\]

The unknown parameters are:

- \(\theta\)
- \(M\)
- \(X\)

The given parameter ranges are:

- \(0^\circ < \theta < 50^\circ\)
- \(-0.05 < M < 0.05\)
- \(0 < X < 100\)
- \(6 \leq t \leq 60\)

The file `xy_data.csv` contains points that lie on the curve. The objective is to recover the values of \(\theta\), \(M\), and \(X\).

---

## Method Used

This problem is treated as a parameter estimation problem.

The steps followed are:

1. Load the given `(x, y)` data from `xy_data.csv`.
2. Define the given parametric curve in Python.
3. Uniformly sample values of `t` from 6 to 60.
4. Generate predicted curve points for a trial set of parameters \((\theta, M, X)\).
5. Compare the predicted curve with the given data using a symmetric nearest-neighbour L1 distance.
6. Use global optimization to minimize the L1 distance.
7. Extract the best values of \(\theta\), \(M\), and \(X\).
8. Save plots and result files for verification.

I used `scipy.optimize.differential_evolution` because:

- the equation is nonlinear,
- the parameters are bounded,
- it does not require gradients,
- it is suitable for global optimization problems.

---

## L1 Loss Function

The assignment evaluates the L1 distance between uniformly sampled expected and predicted curves. Since the data points may not necessarily be ordered by the parameter `t`, the code uses a symmetric nearest-neighbour L1 distance.

For a predicted curve point set \(P\) and the given data point set \(D\), the loss is:

\[
L = \text{mean}_{d \in D}\min_{p \in P}\|d-p\|_1 + \text{mean}_{p \in P}\min_{d \in D}\|p-d\|_1
\]

This compares the given points to the predicted curve and also compares the predicted curve back to the given points.

---

## Final Recovered Parameters

The optimizer recovered:

```text
theta = 29.99956345 degrees
theta = 0.52359116 radians
M     = 0.03000128
X     = 54.99855163
L1 Loss = 0.0400858583
```

Rounded final answer:

```text
theta = 30 degrees
M     = 0.03
X     = 55
```

---

## Optimization Statistics

```text
Iterations           : 90
Function evaluations : 4155
Optimization success : True
Optimizer message    : Optimization terminated successfully.
```

---

## Final Parametric Curve

Using the rounded values:

\[
x(t)=t\cos(0.523599)-e^{0.03|t|}\sin(0.3t)\sin(0.523599)+55
\]

\[
y(t)=42+t\sin(0.523599)+e^{0.03|t|}\sin(0.3t)\cos(0.523599)
\]

where:

\[
6 \leq t \leq 60
\]

---

## Desmos / Submission Format

Rounded final submission equation:

```latex
\left(t\cos(0.523599)-e^{0.03\left|t\right|}\sin(0.3t)\sin(0.523599)+55,42+t\sin(0.523599)+e^{0.03\left|t\right|}\sin(0.3t)\cos(0.523599)\right)
```

Equation generated directly from the optimizer:

```latex
\left(t\cos(0.523591)-e^{0.030001\left|t\right|}\sin(0.3t)\sin(0.523591)+54.998552,42+t\sin(0.523591)+e^{0.030001\left|t\right|}\sin(0.3t)\cos(0.523591)\right)
```

---

## Files Included

```text
AI_RD_Assignment_Final.ipynb
main.py
README.md
requirements.txt
xy_data.csv
plots/given_points.png
plots/predicted_curve.png
plots/fitted_curve.png
results/estimated_parameters.txt
results/final_equation.txt
```

---

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then either open and run:

```text
AI_RD_Assignment_Final.ipynb
```

or run the standalone Python script:

```bash
python main.py
```

Both the notebook and the script load the data, estimate the parameters, plot the fitted curve, save the results, and print the final values.
