"""
In this assignment you should interpolate the given function.
"""
import re
import numpy as np
import time
import random
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test
        def option_1(f, a, b, n):
            if n == 1:
                return lambda x: f(a)
            if n == 2:
                return lambda x: f(a) + (f(b) - f(a)) / (b - a) * (x - a)

            def thomas_algorithm(a, b, c, d):
                """
                Solve a tridiagonal system Ax = d where:
                - a: sub-diagonal (n-1 elements)
                - b: main diagonal (n elements)
                - c: super-diagonal (n-1 elements)
                - d: right-hand side (n elements)

                Returns:
                - x: solution to the system
                """
                n = len(d)
                # Forward elimination
                c_prime = np.zeros(n - 1)
                d_prime = np.zeros(n)

                c_prime[0] = c[0] / b[0]
                d_prime[0] = d[0] / b[0]

                for i in range(1, n - 1):
                    denom = b[i] - a[i - 1] * c_prime[i - 1]
                    c_prime[i] = c[i] / denom
                    d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

                d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / (b[n - 1] - a[n - 2] * c_prime[n - 2])

                # Back substitution
                x = np.zeros(n)
                x[-1] = d_prime[-1]

                for i in range(n - 2, -1, -1):
                    x[i] = d_prime[i] - c_prime[i] * x[i + 1]

                return x

            # Generate Chebyshev nodes
            chebyshev_nodes = np.cos((2 * np.arange(1, n + 1) - 1) / (2 * n) * np.pi)
            sample_points = 0.5 * (a + b) + 0.5 * (b - a) * chebyshev_nodes
            sample_points = np.sort(sample_points)  # Ensure points are sorted
            a_coefficients = np.array([f(x) for x in sample_points])

            # intervals: distances between consecutive sample points
            intervals = np.diff(sample_points)
            intervals = np.maximum(intervals, 1e-8)
            # Build alpha_values for the spline system (related to consecutive slopes)
            alpha_values = []
            for i in range(1, n - 1):
                alpha = 3 * ((a_coefficients[i + 1] - a_coefficients[i]) / intervals[i] - (
                            a_coefficients[i] - a_coefficients[i - 1]) / intervals[i - 1])
                alpha_values.append(alpha)

            # Solve the tridiagonal system for inner second derivatives
            lower_diagonal = intervals[:-1]
            main_diagonal = 2 * (intervals[:-1] + intervals[1:])
            upper_diagonal = intervals[1:]
            inner_coefficients = thomas_algorithm(lower_diagonal, main_diagonal, upper_diagonal, alpha_values)

            # c_coefficients array will store the second derivative at each sample point
            c_coefficients = np.zeros(n)
            c_coefficients[1:n - 1] = inner_coefficients

            # b_coefficients and d_coefficients for each interval
            b_coefficients = np.zeros(n - 1)
            d_coefficients = np.zeros(n - 1)

            # Compute b_coefficients and d_coefficients on each subinterval
            for i in range(n - 1):
                b_coefficients[i] = ((a_coefficients[i + 1] - a_coefficients[i]) / intervals[i]
                                     - intervals[i] * (c_coefficients[i + 1] + 2 * c_coefficients[i]) / 3)
                d_coefficients[i] = (c_coefficients[i + 1] - c_coefficients[i]) / (3 * intervals[i])

            # Define the final interpolating function (cubic spline piecewise)
            def interpolating_function(x):
                if x < sample_points[0]:
                    return a_coefficients[0]
                if x > sample_points[-1]:
                    return a_coefficients[-1]

                # Find the interval index by binary searching sample_points
                idx = np.searchsorted(sample_points, x) - 1

                # Here we "clamp" idx so it won't go out of [0, n-2]
                if idx < 0:
                    idx = 0
                elif idx >= n - 1:
                    idx = n - 2

                # dx is the distance from the left endpoint of the interval
                dx = x - sample_points[idx]
                # Return the cubic spline interpolation on this interval
                return (a_coefficients[idx] + b_coefficients[idx] * dx + c_coefficients[idx] * dx ** 2 + d_coefficients[
                    idx] * dx ** 3)

            return interpolating_function


        def option_2(f, a, b, n):

            # x_i = 0.5*(a + b) + 0.5*(b - a)*cos((2i-1)/(2n)*pi)  (i=1..n)
            i_vals = np.arange(1, n + 1)
            cheb_nodes = np.cos((2 * i_vals - 1) * np.pi / (2 * n))
            xs = 0.5 * (a + b) + 0.5 * (b - a) * cheb_nodes

            sort_idx = np.argsort(xs)
            xs = xs[sort_idx]
            fs = np.array([f(x) for x in xs])

            # w_i = (-1)^(i-1) * sin((2*i-1)*pi/(2*n))
            w = (-1) ** (i_vals - 1) * np.sin((2 * i_vals - 1) * np.pi / (2 * n))
            w = w[sort_idx]

            def interpolte_func(x):
                idx_equal = np.where(np.isclose(xs, x, atol=1e-14))[0]
                if len(idx_equal) > 0:
                    return fs[idx_equal[0]]

                diff = x - xs
                temp = w / diff
                numerator = np.sum(temp * fs)
                denominator = np.sum(temp)
                return numerator / denominator

            return interpolte_func
        return option_2(f, a, b, n)

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


# class TestAssignment1(unittest.TestCase):
#
#     def test_with_poly(self):
#         T = time.time()
#
#         ass1 = Assignment1()
#         mean_err = 0
#
#         d = 30
#         for i in tqdm(range(100)):
#             a = np.random.randn(d)
#
#             f = np.poly1d(a)
#
#             ff = ass1.interpolate(f, -10, 10, 100)
#
#             xs = np.random.random(200)
#             err = 0
#             for x in xs:
#                 yy = ff(x)
#                 y = f(x)
#                 err += abs(y - yy)
#
#             err = err / 200
#             mean_err += err
#         mean_err = mean_err / 100
#
#         T = time.time() - T
#         print(T)
#         print(mean_err)
#
#     def test_with_poly_restrict(self):
#         ass1 = Assignment1()
#         a = np.random.randn(5)
#         f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
#         ff = ass1.interpolate(f, -10, 10, 10)
#         xs = np.random.random(20)
#         for x in xs:
#             yy = ff(x)


##########################################################################

import unittest
from tqdm import tqdm

# class TestAssignment1(unittest.TestCase):
#
#     def test_with_poly(self):
#         T = time.time()
#
#         ass1 = Assignment1()
#         mean_err = 0
#
#         d = 30
#         for i in tqdm(range(100)):
#             a = np.random.randn(d)
#
#             f = np.poly1d(a)
#
#             ff = ass1.interpolate(f, -10, 10, 100)
#
#             # Generate random test points
#             xs = np.linspace(-10, 10, 200)
#             err = 0
#             original_vals = []
#             interpolated_vals = []
#
#             for x in xs:
#                 yy = ff(x)
#                 y = f(x)
#                 err += abs(y - yy)
#
#                 original_vals.append(y)
#                 interpolated_vals.append(yy)
#
#             # Calculate mean error
#             err = err / 200
#             mean_err += err
#
#             # Visualization for the first iteration
#             if i == 0:
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(xs, original_vals, label="Original Function", alpha=0.7)
#                 plt.plot(xs, interpolated_vals, label="Interpolated Function", linestyle='dashed', alpha=0.7)
#                 plt.legend()
#                 plt.title("Original vs Interpolated Function")
#                 plt.xlabel("x")
#                 plt.ylabel("y")
#                 plt.savefig("interpolation_plot.png")  # Save plot as image
#                 print("Plot saved as 'interpolation_plot.png'")
#
#         mean_err = mean_err / 100
#
#         T = time.time() - T
#         print("Total time:", T)
#         print("Mean error:", mean_err)
#
#     def test_with_poly_restrict(self):
#         ass1 = Assignment1()
#         a = np.random.randn(5)
#         f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
#         ff = ass1.interpolate(f, -10, 10, 10)
#         xs = np.linspace(-10, 10, 20)
#         for x in xs:
#             yy = ff(x)

# class TestAssignment1(unittest.TestCase):
#     def test_various_functions(self):
#         T = time.time()
#         ass1 = Assignment1()
#         functions = [
#             (lambda x: x**2, -10, 10, "x^2"),
#             (lambda x: x**3 - 2*x + 5, -10, 10, "x^3 - 2x + 5"),
#             (lambda x: np.sin(x), -np.pi, np.pi, "sin(x)"),
#             (lambda x: np.cos(x), -np.pi, np.pi, "cos(x)"),
#             (lambda x: np.exp(x), -2, 2, "exp(x)"),
#             (lambda x: np.exp(-x**2), -2, 2, "exp(-x^2)"),
#             (lambda x: np.log(x + 1), 0.1, 10, "ln(x+1)"),
#             (lambda x: 1 / (1 + x**2), -5, 5, "1 / (1 + x^2)"),
#             (lambda x: 5, -10, 10, "constant (y = 5)"),  # Constant function
#             (lambda x: 1 / (0.1 + x**80), -1, 1, "1 / (0.1 + x^80)"),  # Sharp edges
#             (lambda x: np.exp(np.exp(x)), -1, 1, "e^(e^x)")  # The new function
#         ]
#
#         for f, a, b, name in functions:
#             print(f"Testing function: {name}")
#             ff = ass1.interpolate(f, a, b, 100)
#
#             # Generate test points
#             xs = np.linspace(a, b, 200)
#             err = 0
#             original_vals = []
#             interpolated_vals = []
#
#             for x in xs:
#                 yy = ff(x)
#                 y = f(x)
#                 err += abs(y - yy)
#                 original_vals.append(y)
#                 interpolated_vals.append(yy)
#
#             # Calculate mean error
#             mean_err = err / 200
#             print(f"Mean error for {name}: {mean_err:.6f}")
#
#             # Clean filename
#             clean_name = re.sub(r'[\\/:*?"<>|()^]', '_', name)
#
#             # Visualization
#             plt.figure(figsize=(10, 6))
#             plt.plot(xs, original_vals, label="Original Function", alpha=0.7)
#             plt.plot(xs, interpolated_vals, label="Interpolated Function", linestyle='dashed', alpha=0.7)
#             plt.legend()
#             plt.title(f"Original vs Interpolated Function ({name})")
#             plt.xlabel("x")
#             plt.ylabel("y")
#             plt.savefig(f"spline_interpolation_{clean_name.replace(' ', '_')}.png")
#             print(f"Plot saved as 'spline_interpolation_{clean_name.replace(' ', '_')}.png'")
#
#         T = time.time() - T
#         print(f"Total testing time: {T:.2f} seconds")
if __name__ == "__main__":
    unittest.main()
