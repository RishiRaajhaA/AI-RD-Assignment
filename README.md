# AI R&D Assignment – Parametric Curve Parameter Estimation

## Overview

This repository contains the solution for the AI Research & Development assignment. The objective is to estimate the unknown parameters of a nonlinear parametric curve using the given dataset (`xy_data.csv`).

The parameters are estimated using **Differential Evolution**, a global optimization algorithm, by minimizing the **L1 distance** between the predicted curve and the given data points.

---

# Problem Statement

The given parametric curve is:

```text
x(t) = t*cos(theta) - exp(M|t|)*sin(0.3t)*sin(theta) + X

y(t) = 42 + t*sin(theta) + exp(M|t|)*sin(0.3t)*cos(theta)
```

Unknown parameters:

- θ (theta)
- M
- X

Parameter ranges:

- 0° < θ < 50°
- -0.05 < M < 0.05
- 0 < X < 100
- 6 ≤ t ≤ 60

The file `xy_data.csv` contains sampled points from the curve. The objective is to recover the unknown parameters by minimizing the L1 distance between the predicted and observed points.

---

# Methodology

1. Load the provided `xy_data.csv`.
2. Implement the given parametric equations.
3. Uniformly sample the parameter `t` from 6 to 60.
4. Generate the predicted curve.
5. Compute the symmetric nearest-neighbour L1 distance between the predicted curve and the given data.
6. Use **SciPy Differential Evolution** to estimate the unknown parameters.
7. Plot the recovered curve against the given data.

---

# Optimization Algorithm

Algorithm Used:
- Differential Evolution (`scipy.optimize.differential_evolution`)

Objective Function:
- Symmetric Nearest-Neighbour L1 Distance

---

# Results

Recovered Parameters (rounded):

| Parameter | Value |
|-----------|-------|
| θ | **30°** |
| M | **0.03** |
| X | **55** |

Recovered Parameters (optimizer output):

| Parameter | Value |
|-----------|----------------|
| θ | 29.99956345° |
| M | 0.03000128 |
| X | 54.99855163 |

---

# Final Parametric Equation

```text
x(t) = t*cos(0.523599)
       - exp(0.03|t|)*sin(0.3t)*sin(0.523599)
       + 55

y(t) = 42
       + t*sin(0.523599)
       + exp(0.03|t|)*sin(0.3t)*cos(0.523599)
```

---

# Desmos / LaTeX Submission Format

```latex
\left(
t*\cos(0.523599)
-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.523599)
+55,
42+t*\sin(0.523599)
+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.523599)
\right)
```

---

# Repository Structure

```
AI_RD_Assignment/
│
├── AI_RD_Assignment.ipynb
├── main.py
├── README.md
├── requirements.txt
├── xy_data.csv
├── plots/
│   ├── fitted_curve.png
│   ├── given_points.png
│   └── predicted_curve.png
└── results/
    ├── estimated_parameters.txt
    └── final_equation.txt
```

---

# Requirements

```
numpy
pandas
matplotlib
scipy
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python main.py
```

or open

```
AI_RD_Assignment.ipynb
```

and execute all cells.

---

# Output

The program prints:

- Estimated θ
- Estimated M
- Estimated X
- Final L1 loss

It also generates:

- Fitted curve plot
- Estimated parameter file
- Final equation file

---

# Author

**Rishi Raajha A**

B.Tech Artificial Intelligence Engineering

Amrita School of Engineering
