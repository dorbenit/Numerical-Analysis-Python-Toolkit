# Numerical Analysis Final Project – Python Implementations

This project is a comprehensive collection of five numerical analysis tasks, written in Python, as part of an academic course in Numerical Analysis.  
Each assignment showcases a different technique or algorithmic concept implemented from scratch under strict constraints (no use of `solve`, `roots`, `interpolate`, etc.), and focuses on **efficiency**, **correctness**, and **algorithmic design**.

A core emphasis throughout the project was on **high numerical precision** and **optimized runtime** — each solution was carefully planned with algorithmic thinking and iterative refinement to meet these criteria.

> In several tasks, **multiple implementation options** were developed and tested side-by-side (e.g., spline vs. barycentric interpolation, bisection vs. secant), as part of the author’s process of evaluating efficiency, accuracy, and algorithmic behavior.

---

## 📂 Structure

```
Numerical-Analysis-Project/
├── assignments/
│   ├── assignment1.py
│   ├── assignment2.py
│   ├── assignment3.py
│   ├── assignment4.py
│   └── assignment5.py
├── utils/
│   ├── commons.py
│   ├── functionUtils.py
│   └── sampleFunctions.py
└── README.md
```

---

## 🧰 Project Utilities

> ⚠️ The following utility files were provided by the course staff.  
> These are not part of the author's original code, but they are necessary for the correct execution and testing of several assignments.

- `commons.py`: Includes test functions (with and without noise), constants, and shared evaluation logic
- `functionUtils.py`: Provides decorators for noise injection, timeouts, and random seeding
- `sampleFunctions.py`: Contains function generators, shape classes (e.g. `BezierShape`, `Polygon`), and sample callable test functions

These are essential especially for **Assignment 3–5**, where function evaluation and shape estimation depend on shared behaviors across tasks.

---

## 🧠 Assignment Highlights

### 🔹 Assignment 1: Interpolation
Implements **cubic spline interpolation** (tridiagonal system, Thomas algorithm) and an optimized **barycentric interpolation** using Chebyshev nodes.  
The solution was optimized to achieve **O(n)** time complexity, with the barycentric approach outperforming splines on most real-world tests.

### 🔹 Assignment 2: Finding Intersections
Solves for the **intersection points** between two continuous functions.  
An adaptive combination of **bisection**, **secant**, and logic inspired by **Brent’s algorithm** ensures fast convergence even in noisy or non-linear cases.

> Modular design includes a simpler fallback method (pure bisection) and automated comparison of accuracy/runtime.

### 🔹 Assignment 3: Numerical Integration & Area Between Curves
Efficient implementation of **adaptive Simpson’s Rule** for definite integration, with handling of edge cases (`n=1`, odd `n`).  
Calculates area between two curves by integrating between intersection points.

> Smart reuse of previous functions + precision control with `float32` + iterative accuracy loop

### 🔹 Assignment 4: Polynomial Fitting
Performs **least squares polynomial fitting** of arbitrary degree using QR-based matrix construction and normal equations.  
Includes dynamic validation of polynomial degree against fitting errors.

> Implementation respects the constraint of not using `numpy.polyfit` or `lstsq`.

### 🔹 Assignment 5: Shape Fitting and Area Estimation
Given noisy (x,y) points from an unknown shape, the algorithm estimates area using:
- **Bezier curves** or **circle fitting**
- Time-constrained adaptive sampling
- Abstract class-based design

> Efficient handling of noise, timeout logic, and generalization to different geometric shapes

---

## ⚙️ Requirements

- Python 3.8+
- NumPy
- (optional) tqdm for progress bars during testing

> Some test cases were designed independently by the author, as the official grader used by the course was not shared in this repository.

---

## 📌 Academic Notes

- Developed under submission constraints (no use of black-box methods)
- Focused on **modularity, correctness, numerical precision**, and **runtime optimization**
- Solutions reflect **algorithmic thinking** and methodical debugging
- Several tasks include **multiple alternative implementations** for empirical comparison
---

## ✍️ Author

Developed by **Dor Benita** as part of the Numerical Analysis final project at university.  
Feel free to explore, reuse (with credit), or adapt the code for learning or academic purposes.
