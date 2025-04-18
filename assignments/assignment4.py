"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        # if d >12:
        #     d =12
        #     print("d >12")
        start_time = time.time()
        if d == 0:
            x1, x2 = a, b
            y1, y2 = f(x1), f(x2)
            avg_y = (y1 + y2) / 2

            def constant_function(x):
                return avg_y

            return constant_function
        n = 1000 * d  # Number of sample points

        # Sampling using Chebyshev nodes
        x_samples = np.cos(np.pi * (4 * np.arange(1, n + 1) - 1) / (2 * n)) * (b - a) / 2 + (a + b) / 2
        y_samples = {x: [] for x in x_samples}  # Create a dictionary to store samples for each x

        # Allocate 80% of the time for sampling
        time_for_sampling = maxtime * 0.8  # 80% of the time for sampling

        # Start the sampling loop
        while time.time() - start_time < time_for_sampling:
            if maxtime - (time.time() - start_time) < 2:
                break
            # For each x, calculate the value and append it to the corresponding y_samples entry
            for x in x_samples:
                if time.time() - start_time > time_for_sampling :
                    break  # Stop if we've exceeded the time for sampling
                y_samples[x].append(f(x))  # Append the result of f(x) for the current x

        # Now, average the y_samples for each x
        y_samples_avg = np.array([np.mean(y_samples[x]) for x in x_samples])  # Compute average for each x

        # Normalize x_samples to reduce numerical errors
        x_min = np.min(x_samples)
        x_max = np.max(x_samples)
        x_normalized = 2 * (x_samples - x_min) / (x_max - x_min) - 1  # scale to [-1, 1]

        # Custom implementation of Vandermonde matrix
        def create_vandermonde(x, degree):
            m = len(x)
            vander = np.zeros((m, degree + 1))
            for i in range(degree + 1):
                vander[:, i] = x ** i
            return vander

        # Build Vandermonde matrix
        M = create_vandermonde(x_normalized, d)

        # Compute coefficients using numpy
        try:
            MT = M.T  # M^T
            MTM = np.dot(MT, M)  # M^T * M
            MTM_inverse = np.linalg.inv(MTM)  # (M^T * M)^-1
            MTY = np.dot(MT, y_samples_avg)  # M^T * Y
            coefficients = np.dot(MTM_inverse, MTY)  # (M^T * M)^-1 * M^T * Y
        except np.linalg.LinAlgError as e:
            print(f"Matrix inversion failed: {e}")

            # Return a function based on the data we have so far
            def g_partial(x):
                # Normalize the input x
                x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
                powers = np.array([x_normalized ** i for i in range(d + 1)])
                return 0  # Placeholder for incomplete coefficients

            return g_partial

        # Define the resulting polynomial function
        def g(x):
            # Normalize the input x
            x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
            powers = np.array([x_normalized ** i for i in range(d + 1)])
            return np.dot(coefficients, powers)

        # Ensure we respect the time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > maxtime :
            print("Function exceeded allowed runtime. Returning partial result.")
            return g  # Return the function based on the available coefficients.

        return g

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


# class TestAssignment4(unittest.TestCase):
#     def test_polynomial(self):
#         def test_polynomial(x):
#             return 2 * x ** 4 - 3 * x ** 3 + x ** 2 - 5 * x + 7
#
#         # יצירת מופע של המחלקה Assignment4
#         ass4 = Assignment4()
#
#         # הגדרת הפרמטרים
#         a = -10  # תחילת הטווח
#         b = 10  # סוף הטווח
#         d = 4  # דרגת הפולינום
#         maxtime = 10  # זמן מקסימלי (בשניות)
#
#         # הפעלת fit לקבלת הפונקציה המוערכת
#         fitted_function = ass4.fit(test_polynomial, a, b, d, maxtime)
#
#         # בדיקת תוצאות על מספר ערכים
#         test_points = [-10, -5, 0, 5, 10]
#         print("Testing fitted polynomial:")
#         for x in test_points:
#             original = test_polynomial(x)
#             fitted = fitted_function(x)
#             print(f"x = {x}, Original = {original}, Fitted = {fitted:.4f}, Error = {abs(original - fitted):.4e}")
#
#         # יצירת גרף
#         x_vals = np.linspace(a, b, 1000)
#         original_vals = [test_polynomial(x) for x in x_vals]
#         fitted_vals = [fitted_function(x) for x in x_vals]
#
#         plt.figure(figsize=(10, 6))
#         plt.plot(x_vals, original_vals, label="Original Polynomial", color="blue", linestyle="--")
#         plt.plot(x_vals, fitted_vals, label="Fitted Polynomial", color="red")
#         plt.scatter(test_points, [test_polynomial(x) for x in test_points], color="blue", label="Original Test Points",
#                     zorder=5)
#         plt.scatter(test_points, [fitted_function(x) for x in test_points], color="red", label="Fitted Test Points",
#                     zorder=5)
#
#         plt.title("Comparison of Original and Fitted Polynomials")
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#     def test_return(self):
#         f = NOISY(0.01)(poly(1,1,1))
#         ass4 = Assignment4()
#         T = time.time()
#         shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         self.assertLessEqual(T, 5)
#
#     def test_delay(self):
#         f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
#
#         ass4 = Assignment4()
#         T = time.time()
#         shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         self.assertGreaterEqual(T, 5)
#
#     def test_err(self):
#         f = poly(1,1,1)
#         nf = NOISY(1)(f)
#         ass4 = Assignment4()
#         T = time.time()
#         ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         mse=0
#         for x in np.linspace(0,1,1000):
#             self.assertNotEquals(f(x), nf(x))
#             mse+= (f(x)-ff(x))**2
#         mse = mse/1000
#         print(mse)
#
#
#



if __name__ == "__main__":
    unittest.main()
